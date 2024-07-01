# evaluate perplexity of a pushdown model, using either beam search or gold parses

import sys
from pathlib import Path
import pdb

# This line adds the parent directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))


from transformers import GPT2Tokenizer
import torch
import numpy as np
import argparse
import yaml
import random, pickle
import pushdown_util
import hydra
import collate

from data_utils.text_helpers import binarize_tree, flatten, ParserPipeline, reformat_tree, reformat_sentence
from transformer_helpers import create_lm
from util import set_seed, convert_tree_to_tuple
from tqdm import tqdm
# Include loading logic from tree projection experiments
from data_utils.text_helpers import binarize_tree
import pickle
import matplotlib.pyplot as plt

from nltk import Tree


class PreProcessor:
    def __init__(self):
        # tokenize with GPT2 tokenizer
        self.gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    def process(self, ex, add_eos=True):
        # if last word has question mark, remove it

        if ex[-1] == "?":
            ex = ex[:-1] + " ?"
        tokenized = self.gpt_tokenizer.tokenize(ex)
        if add_eos:
            joined_ex = " ".join(tokenized) + " <eos>"
        else:
            joined_ex = " ".join(tokenized)
        return joined_ex


def get_parsing_accuracy(predicted_parses, gold_parses, split):
    """Compute parsing scores for predicted parses."""

    def get_brackets(parse):
        p_set = set()

        def get_brackets_helpers(t, st):
            if type(t) == str:
                return 1
            else:
                l1_len = get_brackets_helpers(t[0], st)
                l2_len = get_brackets_helpers(t[1], st + l1_len)
                p_set.add((st, st + l1_len + l2_len - 1))
                return l1_len + l2_len

        get_brackets_helpers(parse, 0)
        return p_set

    gold_brackets = [get_brackets(parse) for parse in gold_parses]
    pred_brackets = [get_brackets(parse) for parse in predicted_parses]

    def get_score(set_1, set_2):
        score = 0.0
        lengthwise = {}
        for p in set_2:
            if p in set_1:
                score += 1
                if (p[1] - p[0]) not in lengthwise:
                    lengthwise[p[1] - p[0]] = 0
                lengthwise[p[1] - p[0]] += 1
        return (score, lengthwise)
    
    def get_base_lengthwise_stats(brackets):
        lengthwise = {}
        for b in brackets:
            if (len(b) == 0):
                continue
            max_length = -1
            for p in b:
                if (p[1] - p[0]) not in lengthwise:
                    lengthwise[p[1] - p[0]] = 0
                max_length = max(max_length, p[1] - p[0])
                lengthwise[p[1] - p[0]] += 1
            # remove full sentence bracket
            lengthwise[max_length] -= 1
        return lengthwise

    precision = sum(
        [get_score(gold, pred)[0] for gold, pred in zip(gold_brackets, pred_brackets)]
    )
    recall = sum(
        [get_score(pred, gold)[0] for gold, pred in zip(gold_brackets, pred_brackets)]
    )
    precision /= 1.0 * sum(len(b) for b in pred_brackets)
    recall /= 1.0 * sum(len(b) for b in gold_brackets)

    pred_lengthwise = {}
    for gold, pred in zip(gold_brackets, pred_brackets):
        _, lengthwise = get_score(gold, pred)
        if (len(lengthwise.keys()) == 0):
            continue
        max_length = max(lengthwise.keys())
        for key in lengthwise:
            if key not in pred_lengthwise:
                pred_lengthwise[key] = 0
            # remove full sentence bracket
            if key != max_length:
                pred_lengthwise[key] += lengthwise[key]

    gold_lengthwise = get_base_lengthwise_stats(gold_brackets)

    lengthwise_stats = {}
    for key in gold_lengthwise:
        if (gold_lengthwise[key] == 0):
            continue
        if (key not in pred_lengthwise):
            lengthwise_stats[key] = 0
        else:
            lengthwise_stats[key] = pred_lengthwise[key]/gold_lengthwise[key]

    paired_lists = []
    for key in lengthwise_stats:
        paired_lists.append((key, lengthwise_stats[key]))
    paired_lists.sort()

    plt.plot([_[0] + 1 for _ in paired_lists], [_[1] for _ in paired_lists])
    plt.savefig(f'{split}.png')
    plt.close()

    return {
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / (precision + recall + 1e-10),
    }

def convert_tree_to_tuple_and_collate(tree):
    """Convert NLTK tree to a tuple representation. Collapse subwords"""
    def fix(t):
        if type(t) == str:
            return t
        elif len(t) == 1:
            return fix(t[0])
        else:
            all_children = [c for c in t]
            span_subwords = t.leaves()
            one_word = True
            for idx in range(1, len(span_subwords)):
                if span_subwords[idx][0] == 'Ġ':
                    one_word=False
                    break
            if one_word:
                return "".join(span_subwords)
            else:
                return (fix(all_children[0]), fix(tuple(all_children[1:])))

    return fix(tree)

def load_data(args):
    if "bllip" in args.dataset:
        with open("../data_utils/{}/{}.txt".format(args.dataset, args.data_split)) as f:
            data = [Tree.fromstring(l.strip()) for l in f.readlines()]
        examples = [(d, flatten(d, add_eos=True)) for d in data]
        return examples
    elif args.dataset == "blimp":
        with open("../data_utils/blimp_processed.pkl", "rb") as f:
            data = [Tree.fromstring(t) for t in pickle.load(f)]
        # we do not use any of trees from BLIMP as part of the model,
        # but just to compute parsing accuracy for analysis.
        examples = [(convert_tree_to_tuple(d), flatten(d, add_eos=True)) for d in data]
        return examples
    else:
        preprocessor = PreProcessor()
        with open(args.dataset, "r") as reader:
            data = [preprocessor.process(l.strip()) for l in reader.readlines()]
        return data

def callback_pushdown(lm, in_vocab, split, regularizer, args, data_ratio=1.0, data_folder_given=None, override=False, hf=False):
    """Callback function on BLIMP for pushdown lm training."""
    if data_folder_given:
        folder_dir = data_folder_given
    else:
        folder_dir = "bllip-lg-depth"
    
    with open("{}/{}.txt".format(folder_dir, split)) as f:
        if (split == 'train'):
            data = [Tree.fromstring(l.strip()) for l in f.readlines()[:min(int(data_ratio * 1755715), 3000)]]
        else:
            data = [Tree.fromstring(l.strip()) for l in f.readlines()]

    device = torch.device("cuda:0")
    if hf:
        examples = [(reformat_tree(d, in_vocab, True), flatten(d, add_eos=False, clean=True)) for d in data]
    else:
        examples = [(d, flatten(d, add_eos=False)) for d in data]

    # get parsevals
    actual = 0
    if regularizer is not None:
        gold_parses = []
        predicted_parses = []
        for d, sentence in tqdm(examples):
            actual += 1
            lm.eval()
            if hf:
                sentence = reformat_sentence(sentence, in_vocab) # add in spaces to match tokenizer expected format
            chart, _, _, _ = regularizer.build_scores([sentence], lm, 0, batch=True, use_gold=False, eval=True)
            predicted_parse, _, _ = regularizer.get_parse([sentence], chart, override=override)
            predicted_parses.append(predicted_parse[0])
            gold_parses.append(convert_tree_to_tuple_and_collate(d))
            # pdb.set_trace()
        parsing_acc = get_parsing_accuracy(predicted_parses, gold_parses, split)

    sent_ppl, _ = eval_base_model(lm, examples, in_vocab, 0, hf=hf)
    
    # Get accs for BLIMP
    if hf:
        with open("data_utils/blimp.pkl", "rb") as f:
            blimp_data = [reformat_tree(Tree.fromstring(t), in_vocab, True) for t in pickle.load(f)]
        blimp_examples = [flatten(d, add_eos=False, clean=True) for d in blimp_data]
    else:
        with open("data_utils/blimp.pkl", "rb") as f:
            blimp_data = [Tree.fromstring(t) for t in pickle.load(f)]
        blimp_examples = [flatten(d, add_eos=False) for d in blimp_data]

    # Rewrite
    all_sent_logprobs = pushdown_util.make_preds_base_model(
            lm, in_vocab, blimp_examples, hf=hf
        )

    num_pairs = len(blimp_examples)//2
    acc = 0
    for num in range(num_pairs):
        good_prop = np.sum(all_sent_logprobs[num])/len(all_sent_logprobs[num])
        bad_prop = np.sum(all_sent_logprobs[num + num_pairs])/len(all_sent_logprobs[num + num_pairs])

        if (good_prop >= bad_prop):
            acc += 1

    acc /= num_pairs

    print(acc)

    if regularizer is not None:
        return {"ppl": sent_ppl,
                "blimp_acc": acc,
                "parsing_acc": parsing_acc["f1"]}
    else:
        return {"ppl": sent_ppl, "blimp_acc": acc}
    
def callback_ptb(lm, in_vocab, split, regularizer, args, data_folder_given=None, override=False):
    """Callback function on BLIMP for pushdown lm training."""
    if data_folder_given:
        folder_dir = data_folder_given
    else:
        folder_dir = "bllip-lg-depth"
    
    with open("{}/en_ptb3-revised_{}.mrg".format(folder_dir, split)) as f:
        pipeline = ParserPipeline()
        if (split == 'train'):
            data = [pipeline(l.strip()) for l in f.readlines()[:3000]]
        else:
            data = [pipeline(l.strip()) for l in f.readlines()]

    device = torch.device("cuda:0")
    examples = []
    exceptions = 0
    for d in data:
        try:
            tokens = in_vocab(flatten(d, add_eos=False))
            examples.append((d, flatten(d, add_eos=False)))
        except:
            exceptions += 1

    # get parsevals
    actual = 0
    if regularizer is not None:
        gold_parses = []
        predicted_parses = []
        for d, sentence in tqdm(examples):
            actual += 1
            lm.eval()
            chart, _, _, _ = regularizer.build_scores([sentence], lm, 0, batch=True, use_gold=False, eval=True)
            predicted_parse, _, _ = regularizer.get_parse([sentence], chart, override=override)
            predicted_parses.append(predicted_parse[0])
            gold_parses.append(convert_tree_to_tuple_and_collate(d))
            # pdb.set_trace()
        parsing_acc = get_parsing_accuracy(predicted_parses, gold_parses, split)

    sent_ppl, _ = eval_base_model(lm, examples, in_vocab, 0)

    # Get accs for BLIMP
    with open("data_utils/blimp.pkl", "rb") as f:
        blimp_data = [Tree.fromstring(t) for t in pickle.load(f)]
    blimp_examples = [flatten(d, add_eos=False) for d in blimp_data]

    # Rewrite
    all_sent_logprobs = pushdown_util.make_preds_base_model(
            lm, in_vocab, blimp_examples
        )

    num_pairs = len(blimp_examples)//2
    acc = 0
    for num in range(num_pairs):
        good_prop = np.sum(all_sent_logprobs[num])/len(all_sent_logprobs[num])
        bad_prop = np.sum(all_sent_logprobs[num + num_pairs])/len(all_sent_logprobs[num + num_pairs])

        if (good_prop >= bad_prop):
            acc += 1

    acc /= num_pairs

    print(acc)

    if regularizer is not None:
        return {"ppl": sent_ppl,
                "blimp_acc": acc,
                "parsing_acc": parsing_acc["f1"]}
    else:
        return {"ppl": sent_ppl, "blimp_acc": acc}

def tokenizer_helper(
    tokenizer,
    data_collator,
    inp_slice
    ):

    inp_list = [tokenizer(s) for s in inp_slice]
    in_lens = [len(s) for s in inp_list]
    inp_to_collate = [{"in": x, "in_len": y} for x, y in zip(inp_list, in_lens)]
    inp = data_collator(inp_to_collate)
    in_len = inp["in_len"].long()
    return (
        inp["in"].transpose(0, 1),
        in_len,
    )


def compute_perplexity_from_logprobs(all_logprobs):
    """
    Compute perplexity from logprobs. works for both torch and numpy arrays.
    Also works if we want to marginalize parses
    """
    if type(all_logprobs[0]) == torch.Tensor:
        total_logprob = np.sum([torch.sum(p).item() for p in all_logprobs])
        total_len = np.sum([len(s) for s in all_logprobs])
    elif len(all_logprobs[0]) == 2:
        ### sent logprb, length
        total_len = np.sum([_len for logprob_set, _len in all_logprobs])
        total_logprob = np.sum(
            [logsumexp(logprob_set) for logprob_set, _len in all_logprobs]
        )
    else:
        total_logprob = np.sum([np.sum(p) for p in all_logprobs])
        total_len = np.sum([len(s) for s in all_logprobs])
    return np.exp(-total_logprob / total_len)


def eval_base_model(lm, examples, tokenizer, device, hf=False):
    """Evaluate a standard transformer LM (no pushdown / external stack)."""
    all_sent_logprobs = pushdown_util.make_preds_base_model(
        lm, tokenizer, [s for p, s in examples], hf=hf
    )
    sent_ppl = compute_perplexity_from_logprobs([x for x in all_sent_logprobs])
    return sent_ppl, None


def logsumexp(x):
    x = np.array(x)
    max_x = np.max(x)
    return max_x + np.log(np.sum(np.exp(x - max_x)))
