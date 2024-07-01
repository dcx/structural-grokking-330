### phrase sampler for the regularizer module

from joblib import parallel_backend
import torch
from tree_projections.tree_projections_utils import get_all_hidden_states_scratch, get_masking_info, get_pre_tokenized_info
from tqdm import tqdm
from torch.optim import Adam
import torch.nn.functional as F
import collate
from data_utils.dyck_helpers import build_datasets_dyck, eval_callback_dyck
from data_utils.lm_dataset_helpers import build_datasets_lm
from transformer_helpers import create_lm
from random import randint, sample
import random
import time
import pdb
import numpy as np

random.seed(42)

class PhraseSampler(torch.nn.Module):
    def __init__(self, spaces, args):
        super().__init__()

        # Initialize the SCI chart
        self.spaces = spaces # False for AddMult!
        self.dataset = args.dataset

        # phrase sampler version of regularizer: sample small phrases from the input and calculate
        self.sample_num = args.reg_sample_num
        self.sample_len = args.reg_sample_len
        self.sample_subtrees = args.sample_subtrees

        # top-layer decision idea
        self.single = args.reg_single


    def get_phrases(self, str):
        # Sample phrases from given string
        phrases = []
        starting_indices = []

        # It is guaranteed that str is larger than phrase len
        for iter in range(self.sample_num):
            if (self.dataset in ["ds-addmult-mod10", "dyck"]):
                # ensure phrases are the largest nested subtree
                if (self.dataset == "dyck"):
                    curr_str = str.split(" ")
                else:
                    curr_str = str
                start = randint(0, len(curr_str) - self.sample_len - 1)
                start_brack = -1
                for pos in range(start, len(curr_str)):
                    if (self.dataset == "dyck"):
                        if (curr_str[pos][0] == '('):
                            start_brack = pos
                            break
                    else:
                        if (curr_str[pos] == '('):
                            start_brack = pos
                            break
                if (start_brack == -1):
                    continue
                depth = 0
                end_brack = -1
                for pos in range(start_brack, len(curr_str)):
                    if (self.dataset == "dyck"):
                        if (curr_str[pos][0] == '('):
                            depth += 1
                        elif (curr_str[pos][-1] == ')'):
                            depth -= 1
                    else:
                        if (curr_str[pos] == '('):
                            depth += 1
                        elif (curr_str[pos] == ')'):
                            depth -= 1
                    if (depth == 0):
                        end_brack = pos
                        break
                if (end_brack == -1):
                    continue
                # allow some relaxation in phrase length, as most well-formed phrases will be small
                if (end_brack - start_brack > 2*self.sample_len):
                    continue
                if (self.dataset == "dyck"):
                    phrase = " ".join(curr_str[start : start + self.sample_len + 1])
                    starting_indices.append(start)
                else:
                    phrase = curr_str[start_brack : end_brack + 1]
                    starting_indices.append(start_brack)
                phrases.append(phrase)
            else:
                # simple sampling (phrases are random spans in the input)
                split_str = str.split(" ")
                start = randint(0, len(split_str) - self.sample_len - 1)
                # Is it better to sample lengths here?
                phrase = " ".join(split_str[start : start + self.sample_len + 1])
                phrases.append(phrase)
                starting_indices.append(start)
        
        return phrases, starting_indices
    
    def get_subtree_sizes(self, parse_dict, size_dict, st, en):
        # get sizes of all subtrees
        if f'{st} {en}' not in parse_dict:
            # full word
            return 0

        split = parse_dict[f'{st} {en}']
        s1 = self.get_subtree_sizes(parse_dict, size_dict, st, split)
        s2 = self.get_subtree_sizes(parse_dict, size_dict, split, en)

        size_dict[(st, en)] = s1+s2+1
        return s1 + s2 + 1
    
    def get_subtree_depths(self, parse_dict, depth_dict, st, en, d):
        # get sizes of all subtrees
        if f'{st} {en}' not in parse_dict:
            # full word
            return 0

        split = parse_dict[f'{st} {en}']
        s1 = self.get_subtree_depths(parse_dict, depth_dict, st, split, d+1)
        s2 = self.get_subtree_depths(parse_dict, depth_dict, split, en, d+1)

        depth_dict[(st, en)] = d
        return s1 + s2 + 1

    def get_full_subtree_spans(self, parse_dict, spans, depths, st, en, d):
        # get all spans in tree
        if f'{st} {en}' not in parse_dict:
            # full word
            return
        
        spans.append((st, en))
        depths.append(d)
        split = parse_dict[f'{st} {en}']

        self.get_full_subtree_spans(parse_dict, spans, depths, st, split, d+1)
        self.get_full_subtree_spans(parse_dict, spans, depths, split, en, d+1)

        return

    def get_gold_parse_phrases(self, curr_str, parse):
        # Only sample phrases that are subtrees of the gold parse tree
        gold_phrases = [tuple([int(_) for _ in phrase.split(" ")]) for phrase in list(parse.keys())]

        # sample with probability proportional to phrase length
        # can change this in non-gold version as well
        filtered_phrases = [phrase for phrase in gold_phrases if (phrase[1] - phrase[0]) <= self.sample_len]

        if len(filtered_phrases) == 0:
            return [], [], [], []

        if self.sample_subtrees:
            # Choose largest subtree that does not exceed self.sample_num spans and max span length does not exceed self.sample_len
            subtree_sizes = {}
            self.get_subtree_sizes(parse, subtree_sizes, 0, len(curr_str.split(" ")))
            candidate = ("", 0)
            # prioritize length?
            for phrase in subtree_sizes:
                (st, en) = phrase
                if subtree_sizes[phrase] <= self.sample_num and en - st <= self.sample_len:
                    if en - st >= candidate[1]:
                        # update root node
                        candidate = (f'{st} {en}', en - st)
            if candidate[1] == 0:
                return [], [], [], []

            sampled_spans = []
            sampled_depths = []
            [c_st, c_en] = [int(_) for _ in candidate[0].split(" ")]
            self.get_full_subtree_spans(parse, sampled_spans, sampled_depths, c_st, c_en, 0)
        else:
            # Per class probabilities for uniform distribution over lengths
            phrase_lengths = {}
            for phrase in filtered_phrases:
                curr_len = phrase[1] - phrase[0]
                if curr_len not in phrase_lengths:
                    phrase_lengths[curr_len] = 0
                phrase_lengths[curr_len] += 1
            prob_dist = [1/len(phrase_lengths) * 1/phrase_lengths[phrase[1] - phrase[0]] for phrase in filtered_phrases]

            depth_dict = {}
            self.get_subtree_depths(parse, depth_dict, 0, len(curr_str.split(" ")), 0)

            # Forcibly make them sum to 1
            prob_dist = [_/sum(prob_dist) for _ in prob_dist]
            prob_dist[-1] += 1 - sum(prob_dist)

            sampled_spans_indices = np.random.choice(len(filtered_phrases), min(self.sample_num, len(filtered_phrases)), p=prob_dist, replace=False)
            sampled_spans = [filtered_phrases[idx] for idx in sampled_spans_indices]
            sampled_depths = [depth_dict[_] for _ in sampled_spans]

        phrases = []
        split_str = curr_str.split(" ")
        starting_indices = []

        new_parses = []
        for span in sampled_spans:
            (s,e) = span
            if (e-s > self.sample_len):
                continue
            phrases.append(" ".join(split_str[s:e]))
            starting_indices.append(s)

            # all relevant parses need to be filtered and reindexed
            relevant_parses = {}
            for split in parse:
                [c_s,c_e] = split.split(" ")
                c_s = int(c_s)
                c_e = int(c_e)
                if (c_s >= s and c_e <= e):
                    # this is a subtree of the sampled subtree
                    # Reindex so that start of phrase is index 0
                    relevant_parses[str(c_s - s) + " " + str(c_e - s)] = parse[split] - s
            new_parses.append(relevant_parses)

        return phrases, new_parses, starting_indices, sampled_depths

    def phrase_split(self, input_str, parses, batch, use_gold, eval):
        # Split input into smaller phrases to get past O(n^2) computation

        if batch:
            outer_input = input_str
        else:
            outer_input = [input_str]

        # for each phrase: which outer_input, starting index
        index_info = []
        sampled_parses = []
        sampled_depths = []
        if (self.sample_num != -1 and not eval):
            # perform sampling
            sampled_input = []
            for idx, str in enumerate(outer_input):
                if (self.dataset in ["ds-addmult-mod10"]):
                    if (len(str) <= self.sample_len):
                        # No need for sampling, sentence is smaller than max phrase length
                        curr_str = [str]
                        index_info.append((idx, 0))
                    else:
                        curr_str, starting_indices = self.get_phrases(str)
                    if (use_gold and self.single):
                        # In regularization based on gold parses, brackets and operators are stripped.
                        for curr_idx, sampled in enumerate(curr_str):
                            sampled_input.append(sampled[2:-1])
                            index_info.append((idx, starting_indices[curr_idx] + 2))
                    else:
                        sampled_input += curr_str
                        for _ in starting_indices:
                            index_info.append((idx, _))
                else:
                    split_str = str.split(" ")
                    curr_str = str
                    if (len(split_str) <= 2):
                        # no point in using this example
                        continue
                    if (len(split_str) <= self.sample_len and parses is None):
                        # No need for sampling, sentence is smaller than max phrase length
                        sampled_input += [curr_str]
                        index_info.append((idx, 0))
                    else:
                        if (parses is not None):
                            new_input, new_parses, starting_indices, new_depths = self.get_gold_parse_phrases(curr_str, parses[idx])
                            sampled_input += new_input
                            sampled_parses += new_parses
                            sampled_depths += new_depths
                            for _ in starting_indices:
                                index_info.append((idx, _))
                        else:
                            curr_str, starting_indices = self.get_phrases(curr_str)
                            for _ in starting_indices:
                                index_info.append((idx, _))
                            sampled_input += curr_str
        else:
            # No sampling performed
            sampled_input = outer_input
            for idx, string in enumerate(sampled_input):
                index_info.append((idx, 0))
            if (parses is not None):
                sampled_parses = parses

        # Init SCI chart
        if (not batch and self.sample_num == -1):
            scores = {}
        else:
            scores = [{} for _ in sampled_input]

        return outer_input, sampled_input, index_info, sampled_parses, sampled_depths, scores