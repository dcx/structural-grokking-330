from transformers import GPT2Tokenizer
import numpy as np
import re
import json
from tqdm import tqdm
import math
import hydra
import argparse

from data_utils.text_helpers import build_datasets_pushdown
from train_transformers import get_base_transformer_lm, get_base_transformer_hf
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import torch
import os
import pickle
from beam_search_util import BeamSearchDepthBased, logsumexp
from regularizer.regularizer_main import Chart
import pdb
from pprint import pprint

from pushdown_util import (
    make_preds_base_model
)
from nltk import Tree


class PreProcessor:
    def __init__(self):
        ### tokenize with GPT2 tokenizer
        self.gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    def process(self, ex, add_eos=False):
        ### if last word has question mark, remove it

        if ex[-1] == "?":
            ex = ex[:-1] + " ?"
        tokenized = self.gpt_tokenizer.tokenize(ex)
        if add_eos:
            joined_ex = " ".join(tokenized) + " <eos>"
        else:
            joined_ex = " ".join(tokenized)
        return joined_ex


def eval_math_expr(expr):
    try:
        return eval(expr)
    except:
        return math.nan


def dummy_tree(parse):
    def helper(p):
        if type(p) == str:
            return "(X {})".format(p) if p not in ["(", ")"] else "(X paren)"
        else:
            out = " ".join(helper(x) for x in p)
            return "(X {})".format(out)

    return helper(parse)


class TestSuiteParser:
    def __init__(self, test_suite_file):
        self.test_suite_file = test_suite_file
        self.read_test_suite()
        self.answers = [0 for _ in range(len(self.meta_data["data"]))]
        self.logprobs = [0 for _ in range(len(self.meta_data["data"]))]

    def read_test_suite(self):
        data_file = "../data_utils/sg_test_suites/{}.json".format(self.test_suite_file)
        with open(self.test_suite_file, "r") as f:
            data = json.load(f)
        self.meta_data = {
            "formula": data["predictions"][0]["formula"],
            "data": self.get_sents(data),
        }

    def get_sents(self, data):
        all_ex = []
        for item in data["items"]:
            curr_ex = {}
            for cond in item["conditions"]:
                regions = [x["content"] for x in cond["regions"]]
                curr_ex[cond["condition_name"]] = regions
            all_ex.append(curr_ex)
        return all_ex

    def extract_formulas(self, surprisal_dict):
        formula = self.meta_data["formula"]
        keys = re.findall(r"%([\w|-]+)%", formula)
        keys = set(keys)
        for key in keys:
            positions = set(re.findall(r"\((\d+);%{}%".format(key), formula))
            for position in positions:
                formula = formula.replace(
                    "({};%{}%)".format(position, key),
                    str(surprisal_dict[key][int(position)]),
                )
        ### replace [ with ( and ] with ) to make it a valid math expression

        formula = formula.replace("[", "(")
        formula = formula.replace("]", ")")
        return formula

    def get_example(self, idx):
        return self.meta_data["data"][idx]

    def evaluate_example(self, idx, evaluator, verbose=False):
        examples = self.get_example(idx)
        phen2surprisals = {}
        for phen in examples:
            target_surprisals, logprobs, target_idxs = evaluator.get_surprisals(
                examples[phen]
            )
            if verbose:
                print("Regions: {}".format(examples[phen]))
                print(logprobs)
            phen2surprisals[phen] = [0] + target_surprisals

        extracted_formula = self.extract_formulas(phen2surprisals)
        self.answers[idx] = extracted_formula
        self.logprobs[idx] = logprobs

    def evaluate_all(self, evaluator):
        for idx in range(len(self.meta_data["data"])):
            self.evaluate_example(idx, evaluator)
        return


class Evaluator:
    def __init__(
        self,
        lm,
        beam_obj,
        tokenizer,
        non_incremental=False,
        stack_type_info=None,
        hf=False
    ):
        ### lm: language model, beam_obj: beam search object, tokenizer: tokenizer corresponding to LM
        self.lm = lm
        self.beam_obj = beam_obj
        self.preprocessor = PreProcessor()
        self.gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer = tokenizer
        self.stack_type_info = stack_type_info
        self.non_incremental = non_incremental
        self.hf = hf

    def run_beam_search(self, sent_processed, get_surprisal=True):
        if get_surprisal:
            logprobs, best_incremental_parses, beams = self.beam_obj(
                self.lm, self.tokenizer, sent_processed, 0, get_surprisal=get_surprisal
            )
            return logprobs, best_incremental_parses, beams
        else:
            beams = self.beam_obj(
                self.lm, self.tokenizer, sent_processed, 0, get_surprisal=get_surprisal
            )
            return beams

    def marginalize(self, beams):
        num_words = len(beams[0][2])

        partial_logprob = [0.0]
        prev_marginalized_logprob = 0.0
        marginalized_logprobs = []
        for i in range(num_words):
            curr_logprob_set = []
            seen_parses = set()
            for beam in beams:
                partial_parse = tuple(beam[1][: i + 1])
                if partial_parse not in seen_parses:
                    seen_parses.add(partial_parse)
                    curr_logprob = np.sum(beam[2][:i])
                    if beam[1][i] != i:
                        curr_logprob += beam[2][i][0]
                    else:
                        # shift action!
                        curr_logprob += beam[2][i][0] + beam[2][i][1]
                    curr_logprob_set.append(curr_logprob)
            curr_marginalized_logprob = logsumexp(curr_logprob_set)
            marginalized_logprobs.append(
                curr_marginalized_logprob - prev_marginalized_logprob
            )
            prev_marginalized_logprob = curr_marginalized_logprob
        return marginalized_logprobs

    def get_target_idxs(self, regions):
        sent = " ".join([r.lstrip().rstrip() for r in regions if len(r) > 0])

        sent_processed = self.preprocessor.process(sent + " .")
        region_subword_lens = [
            len(
                self.gpt_tokenizer.tokenize(
                    x.lstrip().rstrip(),
                    add_prefix_space=idx != 0 and len(x.lstrip().rstrip()) > 0,
                )
            )
            for idx, x in enumerate(regions)
        ]
        cumulative_region_subword_lens = [0] + list(np.cumsum(region_subword_lens))
        all_target_idxs = []
        for idx, region in enumerate(regions):
            t_start = cumulative_region_subword_lens[idx]
            t_end = cumulative_region_subword_lens[idx + 1]
            all_target_idxs.append((t_start, t_end))
        return all_target_idxs

    def get_surprisals(self, regions, verbose=False):
        """
        regions: a list of regions which when concatenated with a period and
        processed by the preprocessor, gives a valid input to the language model
        but some regions can be empty, so we need to take care of that
        """

        sent = " ".join([r.lstrip().rstrip() for r in regions if len(r) > 0])
        if self.hf:
            sent_processed = sent + " ."
            region_subword_lens = [
                    len(self.tokenizer.tokenize(
                        x.lstrip().rstrip()
                    ))
                for idx, x in enumerate(regions)
            ]
        else:
            sent_processed = self.preprocessor.process(sent + " .")
            region_subword_lens = [
                len(
                    self.gpt_tokenizer.tokenize(
                        x.lstrip().rstrip(),
                        add_prefix_space=idx != 0 and len(x.lstrip().rstrip()) > 0,
                    )
                )
                for idx, x in enumerate(regions)
            ]
        cumulative_region_subword_lens = [0] + list(np.cumsum(region_subword_lens))
        all_target_idxs = []
        for idx, region in enumerate(regions):
            t_start = cumulative_region_subword_lens[idx]
            t_end = cumulative_region_subword_lens[idx + 1]
            all_target_idxs.append((t_start, t_end))

        if self.beam_obj:
            if not self.non_incremental:
                logprobs, best_incremental_parses, _ = self.run_beam_search(
                    sent_processed
                )
            else:
                beams = self.run_beam_search(sent_processed, get_surprisal=False)
                # marginalize over all beams now.
                logprobs = self.marginalize(beams)
                best_incremental_parses = None
        else:
            all_sent_logprobs = make_preds_base_model(
                self.lm, self.tokenizer, [sent_processed], no_tqdm=True, hf=self.hf
            )
            logprobs = all_sent_logprobs[0]
        target_surprisals = [
            -1.0 * np.sum(logprobs[st:en]) for st, en in all_target_idxs
        ]

        if verbose:
            words = sent_processed.split()
            ### pretty print word and corresponding logprob to 2 decimal places
            print(["{}: {:.2f}".format(w, l) for w, l in zip(words, logprobs)])

        return target_surprisals, logprobs, all_target_idxs

    def get_sent_logprob(self, regions, verbose=False):
        sent = " ".join([r.lstrip() for r in regions if len(r) > 0])
        sent_processed = self.preprocessor.process(sent + " .")
        beams = self.run_beam_search(sent_processed, get_surprisal=False)

        total_logprob = [b[0] for b in beams]

        return logsumexp(total_logprob), beams

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

    if args.hf:
        in_vocab = AutoTokenizer.from_pretrained(args.hf_model_name)
        lm, interface = get_base_transformer_hf(
            args, in_vocab, model_load_path=args.model_load_path
        )
    else:
        in_vocab = pickle.load(open('/afs/cs.stanford.edu/u/ananjan/grokking-330/structural-grokking-330/data_utils/blimp_vocab.pkl', 'rb'))
        lm, interface = get_base_transformer_lm(
            args, in_vocab, model_load_path=args.model_load_path)

    device = torch.device(f"cuda:0")
    lm.to(device)
    lm.eval()

    regularizer = Chart(lambda x1, x2: F.cosine_similarity(x1, x2, dim=-1), in_vocab, True, args)

    beam_obj = None
    eval_obj = Evaluator(lm, beam_obj, in_vocab, non_incremental=False, hf=args.hf)

    avg_acc = 0
    parse_acc = 0
    tot = 0
    for filename in os.listdir('/afs/cs.stanford.edu/u/ananjan/grokking-330/structural-grokking-330/data_utils/sg_test_suites'):
        print(filename)
        f = os.path.join('/afs/cs.stanford.edu/u/ananjan/grokking-330/structural-grokking-330/data_utils/sg_test_suites', filename)
        test_suite_parser = TestSuiteParser(f)
        test_suite_parser.evaluate_all(eval_obj)
        
        # print(test_suite_parser.logprobs)

        acc = 0.0
        for formula in test_suite_parser.answers:
            acc += eval_math_expr(formula)

        if 'fgd' in filename or 'center_embed' in filename:
            nexcept = 0
            for ex in test_suite_parser.meta_data['data']:
                try:
                    sents = ex.values()

                    for sent in sents:
                        transformed_sent = (" ".join(sent)).split(" ")
                        transformed_sent = [transformed_sent[0]] + ['Ġ' + _ for _ in transformed_sent[1:] if _ != ''] + ['Ġ.']
                        chart, _, _, _ = regularizer.build_scores([" ".join(transformed_sent)], lm, 0, batch=True, use_gold=False, eval=True)
                        predicted_parse, score, _ = regularizer.get_parse([" ".join(transformed_sent)], chart, False)
                        display_parse(predicted_parse[0])
                except:
                    nexcept += 1

        avg_acc += acc / len(test_suite_parser.answers)
        tot += 1
        print(acc / len(test_suite_parser.answers))
        # print(curr_parse_acc/curr_tot)

        # parse_acc += curr_parse_acc/curr_tot

    print('Aggregated Score')
    print(avg_acc/tot)
    # print(parse_acc/tot)

if __name__ == "__main__":
    main()
