# Quick (and dirty) script to load a model and generate parses for PTB
# With sharding, so we can run multiple evals in parallel
# Usage: cd eval_utils.py; python eval_ptb.py --shard_id 0 --num_shards 1 --dir_name ptb_results
# The script will write a pickle file containing model predictions and
# ground truth parses (processed such that words are GPT2-tokenized and everything is CNFed)

import sys
from pathlib import Path

# This line adds the parent directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))


import hydra
import yaml
import os, pickle
from train_transformers import get_base_transformer_lm
import argparse
from tqdm import tqdm
from transformers import GPT2Tokenizer
import torch
from nltk import Tree
import pdb
import torch.nn.functional as F
from regularizer.regularizer_main import Chart
import numpy as np
import matplotlib.pyplot as plt
import pushdown_util
from eval_pushdown_model import convert_tree_to_tuple_and_collate, flatten, get_parsing_accuracy


# util to convert nltk tree into nested tuples
def convert_into_tuple(nltk_tree):
    """
    Convert NLTK tree into a tuple of tuples
    """

    def helper(tree):
        if type(tree) == str:
            return tree
        elif len(tree) == 1:
            return helper(tree[0])
        else:
            return tuple(helper(x) for x in tree)

    return helper(nltk_tree)

def tree_to_parse_decisions(parse, start, parse_dict):
    # Builds a dict containing index of gold split for all gold subtrees
    # Accumulates results in parse_dict
    if (len(parse.leaves()) <= 1):
        return len(parse.leaves())
    
    # We know that subwords are never split between constituents. 
    # If all tokens after the first one in the current span do not start with 'G.', it should not be split further (single word)
    single_word = True
    for idx, word in enumerate(parse.leaves()):
        if (idx == 0):
            continue
        if (word[0] == 'Ġ'):
            single_word = False
            break
    
    if single_word:
        return len(parse.leaves())
    
    s1 = len(parse[0].leaves())
    s2 = len(parse[1].leaves()) 

    tree_to_parse_decisions(parse[0], start, parse_dict)
    tree_to_parse_decisions(parse[1], start + s1, parse_dict)

    parse_dict[str(start) + ' ' + str(start + s1 + s2)] = start + s1

    return s1 + s2


# Load test set
class ParserPipeline:
    def __init__(self):
        self.gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    def post_process_op(self, tree):
        """
        Input: nltk tree where leaf nodes are strings
        Output: nltk tree but use gpt_tokenizer to tokenize leaf nodes and create a subtree corresponding to it.
        """

        def fix(t, is_first, label=None):
            if type(t) == str:
                tokenized = self.gpt_tokenizer.tokenize(
                    t, add_prefix_space=not is_first
                )
                if len(tokenized) == 1:
                    return Tree(label, [tokenized[0]])
                else:
                    return Tree(label, [Tree(label, [tok]) for tok in tokenized])
            elif len(t) == 1:
                return fix(t[0], is_first=is_first, label=t.label())
            else:
                return Tree(
                    t.label(),
                    [fix(c, is_first=is_first and idx == 0) for idx, c in enumerate(t)],
                )

        fixed_tree = fix(tree, is_first=True)
        fixed_tree.chomsky_normal_form()
        return fixed_tree

    def process(self, parse):
        ptree = Tree.fromstring(parse)
        ptree.chomsky_normal_form()
        return self.post_process_op(ptree)
        # return ptree

    def __call__(self, parse):
        return self.process(parse)
    
def get_gold_score(parse_dict, regularizer, chart):
    score = 0
    for key in parse_dict:
        [s,e] = [int(_) for _ in key.split(" ")]
        g = parse_dict[key]
        score += regularizer.get_penalty(chart[0], s, g-1, e-1, cky=True)

    return score

def display_parse(parse):
    def create_dummy_tree(parse_string):
        if type(parse_string) == str:
            return Tree('X', [parse_string])
        else:
            return Tree('X', [create_dummy_tree(parse_string[0]), create_dummy_tree(parse_string[1])])
        
    tree = create_dummy_tree(parse)
    tree.pretty_print()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_load_path", type=str, default="/nlp/scr/ananjan/bllip/state_final_model.pt",
                        help="Path of model to resume training, or ")
    parser.add_argument("--vec_dim", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--encoder_n_layers", type=int, default=16)
    parser.add_argument("--embedding_dropout", type=float, default=-1.0)
    parser.add_argument("--output_dropout", type=float, default=-1.0)
    parser.add_argument("--lm_with_token_labels", action="store_true")
    parser.add_argument("--relative", action="store_true")
    parser.add_argument("--rotary", action="store_true")
    parser.add_argument("--dataset", type=str, default="bllip-md")
    parser.add_argument("--use_difference", action="store_true", help="Use difference to compute vectors in SCI score")
    parser.add_argument("--use_gumbel", action="store_true", help="Use gumbel softmax to encourage exploration")
    parser.add_argument("--layer_id", type=int, default=-1, help="Depth limited SCI computation")
    parser.add_argument("--start_relax_layer", type=int, default=0, help="Context-free lower layers")
    parser.add_argument("--end_relax_layer", type=int, default=0, help="Context-free later layers")
    parser.add_argument("--retain_positions", action="store_true", help="Compute phrase representation in-position")
    parser.add_argument("--causal_only", action="store_true", help="Use version of code that only makes causal calls to the LM")
    parser.add_argument("--reg_sample_num", type=int, default=-1, help="Number of phrases sampled for regularization")
    parser.add_argument("--reg_sample_len", type=int, default=10, help="Length of phrases sampled for regularization")
    parser.add_argument("--sample_subtrees", action="store_true", help="Sample entire subtrees of parse tree instead of random phrases")
    parser.add_argument("--use_orthogonal", action="store_true", help="SCI score = norm of orthogonal component")
    parser.add_argument("--reg_single", action="store_true", help="Only top level decision for SCI")
    parser.add_argument("--margin", type=int, default=3)
    parser.add_argument("--neg_samples", type=int, default=-1, help="Number of negative samples to penalize SCI for bad constituents")
    parser.add_argument("--neg_rel_wt", type=float, default=0., help="Relative weight of negative samples to penalize SCI for bad constituents")
    parser.add_argument("--orth_single", action="store_true", help="Use endpoint of span for SCI computation")
    parser.add_argument("--orth_comp", action="store_true", help="Use only cosine for SCI computation")
    parser.add_argument("--orth_diff", action="store_true", help="Use difference for SCI computation")
    parser.add_argument("--orth_prev", action="store_true", help="Use representations from before span for SCI computation")
    parser.add_argument("--proj", action="store_true", help="Projections in SCI computation")
    parser.add_argument("--ce", action="store_true", help="Use cross entropy in (gold) SCI computation")
    parser.add_argument("--sci_heads", type=float, default=-1., help="Proportion of heads for SCI computation")
    parser.add_argument("--cky", action="store_true", help="Train tree reg using CKY scores. Always use single for this.")
    parser.add_argument("--neg_only", action="store_true", help="SCI where random negatives are sampled for training")
    parser.add_argument("--cky_dec", action="store_true", help="Decision level CKY.")
    parser.add_argument("--balance", action="store_true", help="New idea for orthogonal SCI")
    parser.add_argument("--orth_bidir", action="store_true", help="Bidirectional SCI computation")
    parser.add_argument("--pack", action="store_true", help="Implement packing to max_seq_len.")
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--hf", action="store_true", help="Use a model from HuggingFace.")
    parser.add_argument("--hf_model_name", type=str, default="princeton-nlp/Sheared-LLaMA-1.3B")
    
    args = parser.parse_args()
    

    in_vocab = pickle.load(open('/afs/cs.stanford.edu/u/ananjan/grokking-330/structural-grokking-330/data_utils/blimp_vocab.pkl', 'rb'))
    lm, interface = get_base_transformer_lm(
            args, in_vocab, model_load_path=args.model_load_path)

    device = torch.device("cuda:0")
    lm.to(device)
    lm.eval()

    pipeline = ParserPipeline()
    # PTB_DATA_PATH = "/nlp/scr/horatio/data/constituency/en_ptb3-revised_test.mrg"
    PTB_DATA_PATH = "/u/scr/smurty/pushdown-lm/data_utils/bllip-lg-depth/test.txt"
    # PTB_DATA_PATH = "eval_utils/ptb_samples.mrg"
    with open(PTB_DATA_PATH, "r") as reader:
        if 'bllip' in PTB_DATA_PATH:
            data = [Tree.fromstring(l.strip()) for l in reader.readlines()]
        else:
            data = [pipeline(l.strip()) for l in reader.readlines()]
    
    regularizer = Chart(lambda x1, x2: F.cosine_similarity(x1, x2, dim=-1), in_vocab, True, args)
    # evaluate
    ground_truth_parses = []
    greedy_parses = []
    beam_parses = []
    cky_parses = []
    nexcept = 0
    diffs = []
    perfs = []
    lengths = []
    diff_percent = []
    logprobs = []
    ppl = []
    for ex in tqdm(data):
        try:
            sentence = flatten(ex, add_eos=False)
            # if ('-' in sentence) or ('*' in sentence):
            #     continue
            # print(sentence)
            # if len(sentence.split(" ")) > 40:
            #     continue
            # print(sentence)
            chart, _, _, _ = regularizer.build_scores([sentence], lm, 0, batch=True, use_gold=False, eval=True)
            predicted_parse_1, score_1 = regularizer.get_parse_beam([sentence], chart, topk=3)
            beam_parses.append(predicted_parse_1[0])
            predicted_parse_2, score_2, _ = regularizer.get_parse([sentence], chart, False)
            greedy_parses.append(predicted_parse_2[0])
            predicted_parse_3, score_3, cky_charts = regularizer.get_parse_cky([sentence], chart)
            cky_parses.append(predicted_parse_3[0])

            gold_brackets = convert_tree_to_tuple_and_collate(ex)
            ground_truth_parses.append(gold_brackets)
            gold_parse = {}
            _ = tree_to_parse_decisions(ex, 0, gold_parse)
            gold_score = get_gold_score(gold_parse, regularizer, chart)
            diffs.append(score_2[0].item() - gold_score.item())
            diff_percent.append((score_2[0].item() - gold_score.item())/gold_score.item())
            perfs.append(get_parsing_accuracy(predicted_parse_2, [gold_brackets], split="ptb")["f1"])
            lengths.append(len(sentence.split(" ")))

            curr_logprobs = pushdown_util.make_preds_base_model(
                        lm, in_vocab, [sentence]
                    )
            # pdb.set_trace()
            # remove EOS token
            total_logprobs = np.sum(curr_logprobs[0][:-1])
            # print(curr_logprobs)
            # print(len(sentence.split(" ")))
            logprobs.append(total_logprobs)
            curr_ppl = np.exp(-total_logprobs / len(sentence.split(" ")))
            # if curr_ppl > 1000:
            #     beam_parses.pop()
            #     greedy_parses.pop()
            #     cky_parses.pop()
            #     diffs.pop()
            #     perfs.pop()
            #     lengths.pop()
            #     ground_truth_parses.pop()
            #     continue
            ppl.append(curr_ppl)
            # print(ppl)

            # print(perfs)
            if perfs[-1] < 0.7:
            # if 'ĠWall ' in sentence:
                # print(perfs)
                print('Gold')
                display_parse(gold_brackets)
                print('Predicted')
                display_parse(predicted_parse_2[0])
                logit_dict = {}
                for idx, _ in enumerate(sentence.split(" ")):
                    logit_dict[_] = curr_logprobs[0][idx]
                print(logit_dict)
                # pdb.set_trace()
        except:
            nexcept += 1
    print(f'PPL: {np.exp(-sum(logprobs) / sum(lengths))}')
    beam_parsing_acc = get_parsing_accuracy(beam_parses, ground_truth_parses, split="ptb")
    greedy_parsing_acc = get_parsing_accuracy(greedy_parses, ground_truth_parses, split="ptb")
    cky_parsing_acc = get_parsing_accuracy(cky_parses, ground_truth_parses, split="ptb")
    print(beam_parsing_acc)
    print(greedy_parsing_acc)
    print(cky_parsing_acc)
    print(nexcept)
    # diffs = np.array(diffs)
    # print(np.mean(diffs), np.var(diffs))
    # diff_percent = np.array(diff_percent)
    # print(np.mean(diff_percent), np.var(diff_percent))

    num_bins = 20
    bin_edges = np.linspace(0, 200, num_bins + 1)

    avg_perfs = []
    avg_var = []
    bin_centers = []
    perfs = np.array(perfs)
    lengths = np.array(lengths)
    print(np.mean(lengths), np.var(lengths))

    for i in range(num_bins):
        mask = (lengths >= bin_edges[i]) & (lengths < bin_edges[i+1])
        if np.sum(mask) == 0:
            avg_perfs.append(0)
            avg_var.append(0)
        else:
            avg_perf = np.mean(perfs[mask])
            avg_perfs.append(avg_perf)
            avg_var.append(np.sqrt(np.var(perfs[mask])))
        bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)

    plt.bar(bin_centers, avg_perfs, yerr=avg_var, capsize=5, width=(bin_edges[1] - bin_edges[0]), align='center')
    plt.xlabel("Length of Sentence")
    plt.ylabel("Average Parseval F1s")
    plt.title("Average Parseval F1s for Sentences in length ranges")
    plt.grid(axis='y')
    plt.savefig('parseval_length_ptb.png')
    plt.close()

    # print(len(perfs), len(ppl))
    # plt.scatter(ppl, perfs)
    # plt.xlabel("Perplexity")
    # plt.ylabel("Parseval F1")
    # plt.title("Perplexity vs Parseval F1s")
    # plt.savefig('parseval_corr.png')


if __name__ == "__main__":
    main()
