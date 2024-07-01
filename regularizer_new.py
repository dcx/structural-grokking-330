### regularizes a model to encourage structural generalization
### becoming too large, split into multiple classes/scorers

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

class Chart(torch.nn.Module):
    def __init__(self, sim_metric, tokenizer, spaces, args):
        """
        Initialize the SCI score chart.

        Args:
            sim_metric: Distance metric for similarity calculation.
            tokenizer: Tokenizer object.
            spaces: Words in all datasets other than SimPL are space-separated. Set to False for SimPL.
            hinge_const: If hinge loss is used, maximum desired SCI score at any tree node.
            dataset: Current dataset.
            sample_num: If not -1, number of phrases sampled from each input sentence.
            sample_len: Maximum length of phrases sampled.
            single: If enabled, regularize only on decision for top-level split.
            diff: If enabled, SCI score computation is done in difference mode (span embedding = embedding of last token - embedding of token before first token)
            gumbel: If enabled, use gumbel softmax in computation of SCI score.
            linear: Shikhar's linear time tree regularization idea.
        """
        super().__init__()

<<<<<<< HEAD
=======
class Chart():
    def __init__(self, sim_metric, tokenizer, spaces, hinge_const, dataset, 
                 sample_num = -1, sample_len = 8, single = False, diff = False, gumbel = False, linear=False):
        """
        Initialize the SCI score chart.

        Args:
            sim_metric: Distance metric for similarity calculation.
            tokenizer: Tokenizer object.
            spaces: Words in all datasets other than SimPL are space-separated. Set to False for SimPL.
            hinge_const: If hinge loss is used, maximum desired SCI score at any tree node.
            dataset: Current dataset.
            sample_num: If not -1, number of phrases sampled from each input sentence.
            sample_len: Maximum length of phrases sampled.
            single: If enabled, regularize only on decision for top-level split.
            diff: If enabled, SCI score computation is done in difference mode (span embedding = embedding of last token - embedding of token before first token)
            gumbel: If enabled, use gumbel softmax in computation of SCI score.
            linear: Shikhar's linear time tree regularization idea.
        """
        
>>>>>>> b3ce4da34f2346bdda5d7496d402133864818eee
        # Initialize the SCI chart
        self.sim_metric = sim_metric

        self.tokenizer = tokenizer
        self.train_data_collator = collate.VarLengthCollate(None)
        self.spaces = spaces # False for AddMult!
        self._cache = {}
        self.dataset = args.dataset
        self.diff = args.use_difference
        self.gumbel = args.use_gumbel
        self.layer_id = args.layer_id
        if self.layer_id == -1:
            self.layer_id = args.encoder_n_layers
        self.start_relax_layer = args.start_relax_layer
        self.end_relax_layer = args.end_relax_layer
        self.retain_positions = args.retain_positions
        self.causal_only = args.causal_only

        # loss things
        self.margin = args.margin
        self.ce = args.ce
        self.neg_samples = args.neg_samples
        self.neg_rel_wt = args.neg_rel_wt

        # depth-structured tree reg
        self.depth_dep = args.depth_dep

        # phrase sampler version of regularizer: sample small phrases from the input and calculate
<<<<<<< HEAD
        self.sample_num = args.reg_sample_num
        self.sample_len = args.reg_sample_len
        self.sample_subtrees = args.sample_subtrees

        # other versions of tree reg
        self.linear = args.use_linear
        self.orthogonal = args.use_orthogonal
        self.orth_single = args.orth_single
        self.orth_comp = args.orth_comp
        self.orth_diff = args.orth_diff
        self.orth_prev = args.orth_prev
        self.bilinear = args.bilinear
        self.sci_heads = args.sci_heads
        self.proj = args.proj
        if self.bilinear:
            self.b_transform = torch.nn.Bilinear(args.vec_dim, args.vec_dim, args.vec_dim)
        if self.proj:
            self.proj_M = torch.nn.Linear(args.vec_dim, args.vec_dim)

        self.cky = args.cky
        self.cky_dec = args.cky_dec
        self.neg_only = args.neg_only
        if self.cky:
            self.orth_single = True

        # top-layer decision idea
        self.single = args.reg_single
    
    def get_phrases(self, str):
        # Sample phrases from given string
        phrases = []
        starting_indices = []
=======
        self.sample_num = sample_num
        self.sample_len = sample_len

        # linear pass idea (Shikhar)
        self.linear = linear

        # top-layer decision idea
        self.single = single

    def relax_cond(self, mask, relax_mask, start_relax_layer, num_layers):
        ### relax mask only masks padded stuff
        ### mas masked everything
        #### relax mask from 0 ... start_relax_layer-1, 
        #### relax_mask from num_layers - end_relax_layer to num_layers - 1
        #### mask from start_relax_layer to num_layers - end_layers - 1  
        return [relax_mask]*start_relax_layer + [mask]*(num_layers - start_relax_layer)

    def cache(self, input_str, parse_split=None):
        # Could be used in the future: cache masking information
        if input_str not in self._cache:
            sentence2idx_tuple, masked_strs, input_masks = get_masking_info(self.tokenizer, [input_str])
            slice_dict = {idx: key for idx, key in sentence2idx_tuple[0]}
            self._cache[input_str] = (slice_dict, masked_strs, input_masks)
        
        return self._cache[input_str]
    
    def get_phrases(self, str):
        # Sample phrases from given string
        phrases = set()
>>>>>>> b3ce4da34f2346bdda5d7496d402133864818eee

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
<<<<<<< HEAD
                    starting_indices.append(start)
                else:
                    phrase = curr_str[start_brack : end_brack + 1]
                    starting_indices.append(start_brack)
                phrases.append(phrase)
=======
                else:
                    phrase = curr_str[start_brack : end_brack + 1]
                phrases.add(phrase)
>>>>>>> b3ce4da34f2346bdda5d7496d402133864818eee
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
<<<<<<< HEAD
                        curr_str = [str]
                        index_info.append((idx, 0))
=======
                        curr_str =  [str]
>>>>>>> b3ce4da34f2346bdda5d7496d402133864818eee
                    else:
                        curr_str, starting_indices = self.get_phrases(str)
                    if (use_gold and self.single):
                        # In regularization based on gold parses, brackets and operators are stripped.
<<<<<<< HEAD
                        for curr_idx, sampled in enumerate(curr_str):
=======
                        for sampled in curr_str:
>>>>>>> b3ce4da34f2346bdda5d7496d402133864818eee
                            sampled_input.append(sampled[2:-1])
                            index_info.append((idx, starting_indices[curr_idx] + 2))
                    else:
                        sampled_input += curr_str
                        for _ in starting_indices:
                            index_info.append((idx, _))
                else:
                    split_str = str.split(" ")
<<<<<<< HEAD
                    curr_str = str
                    if (len(split_str) <= 2):
                        # no point in using this example
                        continue
                    if (len(split_str) <= self.sample_len and parses is None):
                        # No need for sampling, sentence is smaller than max phrase length
                        sampled_input += [curr_str]
                        index_info.append((idx, 0))
=======
                    if (len(split_str) <= self.sample_len):
                        # No need for sampling, sentence is smaller than max phrase length
                        sampled_input += [str]
>>>>>>> b3ce4da34f2346bdda5d7496d402133864818eee
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
<<<<<<< HEAD
            # No sampling performed
            sampled_input = outer_input
            for idx, string in enumerate(sampled_input):
                index_info.append((idx, 0))
            if (parses is not None):
                sampled_parses = parses
=======
            if (self.dataset in ["ds-addmult-mod10"]):
                sampled_input = []
                for str in outer_input:
                    sampled_input.append(str)
                outer_input = sampled_input
>>>>>>> b3ce4da34f2346bdda5d7496d402133864818eee

        # Init SCI chart
        if (not batch and self.sample_num == -1):
            scores = {}
        else:
            scores = [{} for _ in sampled_input]

<<<<<<< HEAD
        return outer_input, sampled_input, index_info, sampled_parses, sampled_depths, scores
=======
        return outer_input, scores
>>>>>>> b3ce4da34f2346bdda5d7496d402133864818eee
    
    def get_fully_contextual_vectors(self, all_vector_idxs, st, en):
        # Get in-context span representations
        if (self.diff):
            if (st == 0):
                fully_contextual_vectors = all_vector_idxs[en]
            else:
                fully_contextual_vectors = all_vector_idxs[en] - all_vector_idxs[st-1]
            fully_contextual_vectors = fully_contextual_vectors.squeeze()
        elif (self.linear):
            fully_contextual_vectors = all_vector_idxs[st : en + 1].mean(
                axis=0
            )
        else:
            fully_contextual_vectors = all_vector_idxs[st : en + 1].sum(
                axis=0
            )
        return fully_contextual_vectors
<<<<<<< HEAD
    
    def get_l2_norm(self, vec):
        return torch.sqrt(torch.sum(vec*vec, dim=-1))
    
    def get_orthogonal_score(self, all_vector_idxs, st, en):
        # Score for a span is average(norm of orthogonal component to st)
        # Better way of distilling information from before span?

        if st == 0:
            # Left to right LM, always context-free
            return 0
        else:
            # normalize for stability
            if self.sci_heads == -1:
                proportion = all_vector_idxs.shape[1]
            else:
                proportion = int(self.sci_heads * all_vector_idxs.shape[1])

            if self.orth_prev:
                previous_info = F.normalize(all_vector_idxs[:st - 1, :proportion], dim=-1) # S, D
            else:
                previous_info = F.normalize(all_vector_idxs[st - 1, :proportion], dim=-1).unsqueeze(0) 

            # Alternative: trying mean of everything that came before
            # previous_info = F.normalize(torch.mean(all_vector_idxs[:st - 1], dim=0), dim=-1)

            # v_perp = v - (v.v1)v1
            # E-S, D   E-S, S  S, D
            # Original
            span_vectors = all_vector_idxs[st:en+1, :proportion] # (E-S), D

            if self.bilinear:
                # apply bilinear transform
                return self.b_transform(previous_info.expand(span_vectors.shape[0], -1), span_vectors)

            components_along = span_vectors @ previous_info.T

            if self.orth_comp:
                orthogonals = -components_along.squeeze(-1)
            elif self.orth_diff:
                orthogonals = span_vectors - previous_info
            else:
                orthogonals = span_vectors - (components_along @ previous_info)
            
            # Changed
            # orthogonals = -components_along.squeeze(-1)

            # Changed
            # orthogonals = all_vector_idxs[st:en+1] - previous_info.unsqueeze(0)

            # aggregate magnitudes of orthogonals
            return orthogonals
        
    def get_batched_orthogonal_scores(self, outer_vector, slice_dict):
        # Batch computation by starting indices st
        # Compute max endpoint required for all st
        ends_dict = {}
        for (st, en) in slice_dict.values():
            if st not in ends_dict:
                ends_dict[st] = en
            else:
                ends_dict[st] = max(en, ends_dict[st])

        # why wait, compute these scores immediately after they are available
        ends_req_dict = {}
        for key in slice_dict.keys():
            (st, en) = slice_dict[key]
            if st not in ends_req_dict:
                ends_req_dict[st] = []
            ends_req_dict[st].append(key)

        # Get all required norms and scores for a particular st (batched)
        score_dict = {}
        for st in ends_dict:
            orthogonals = self.get_orthogonal_score(outer_vector, st, ends_dict[st])
            for key in ends_req_dict[st]:
                st, en = slice_dict[key]
                if st == 0:
                    score_dict[key] = torch.tensor(0, requires_grad = True, dtype = torch.float).to(outer_vector[0].device) 
                else:
                    if self.orth_comp:
                        if self.orth_single:
                            score_dict[key] = orthogonals[en-st]
                        else:
                            score_dict[key] = torch.mean(orthogonals[:en-st+1])
                    else:
                        if self.orth_single:
                            score_dict[key] = torch.norm(orthogonals[en-st])
                        else:
                            score_dict[key] = torch.mean(torch.norm(orthogonals[:en-st+1], dim=-1))

        return score_dict
=======
>>>>>>> b3ce4da34f2346bdda5d7496d402133864818eee

    def build_scores(self, input_str, model, parses=None, batch=True, use_gold=False, eval=False):
        # Build SCI chart
        # use_gold: Tree regularization guided by gold parses

        def tokenizer_helper(s, add_special_tokens=True):
            # Wrapper to handle LM SOS
            if add_special_tokens:
                return [curr_model.encoder_sos] + self.tokenizer(s)
            else:
                return self.tokenizer(s)
            
        if batch:
            scores = [{} for _ in input_str]
        else:
            scores = {}
        device = torch.device("cuda")
<<<<<<< HEAD
        if (type(model) != torch.nn.Module and not eval):
=======
        if (type(model) != torch.nn.Module):
>>>>>>> b3ce4da34f2346bdda5d7496d402133864818eee
            curr_model = model.model
        else:
            curr_model = model

<<<<<<< HEAD
        outer_input, sampled_input, index_info, parses, depths, scores = self.phrase_split(input_str, parses, batch, use_gold, eval)
        # pdb.set_trace()
=======
        outer_input, scores = self.phrase_split(input_str, batch, use_gold)
>>>>>>> b3ce4da34f2346bdda5d7496d402133864818eee

        batch_sent_tokens, batch_idxs = [], []
        for curr_str in outer_input:
            sent_tokens, idxs = get_pre_tokenized_info(curr_str, tokenizer_helper, spaces = self.spaces)
            batch_sent_tokens.append(sent_tokens)
            batch_idxs.append(idxs)

        outer_context_vecs = {}
        if self.depth_dep:
            for l in range(2, self.layer_id + 1):
                outer_context_vecs_l = get_all_hidden_states_scratch(curr_model,
                                                            tokenizer_helper,
                                                            outer_input,
                                                            tqdm_disable=True,
                                                            pre_tokenized=(batch_sent_tokens, batch_idxs),
                                                            layer_id=l,
                                                            start_relax_layer=0,
                                                            end_relax_layer=0,
                                                            is_lm=True,
                                                            compute_grad=True,
                                                            spaces = self.spaces
                                                        )
                outer_context_vecs[l] = outer_context_vecs_l
        else:
            outer_context_vecs = get_all_hidden_states_scratch(curr_model,
                                                            tokenizer_helper,
                                                            outer_input,
                                                            tqdm_disable=True,
                                                            pre_tokenized=(batch_sent_tokens, batch_idxs),
                                                            layer_id=self.layer_id,
                                                            start_relax_layer=0,
                                                            end_relax_layer=0,
                                                            is_lm=True,
                                                            compute_grad=True,
                                                            spaces = self.spaces
                                                        )

        if eval or self.neg_samples != -1:
            slice_dict, masked_strs, input_masks = get_masking_info(tokenizer_helper, sampled_input, spaces=self.spaces, single=False)
        else:
            slice_dict, masked_strs, input_masks = get_masking_info(tokenizer_helper, sampled_input, spaces=self.spaces, single=self.single)

        # remove duplicate calls
        fin_slice_dict = []
        fin_masked_strs = []
        fin_input_masks = []

        # map values in slice_dict to their indices in the new output
        map_back = {}

        # number of values that are being skipped
        skip_offset = 0

        # encountered slices for each string with corresponding idx
        encountered = [{} for _ in range(len(outer_input))]

        # potential change: better to compute using shorter string
        for idx, slices in enumerate(slice_dict):
            curr_slice_dict = []
            for curr_slice in slices:
                (base_str_idx, offset) = index_info[idx]
                (other_idx, (s,e)) = curr_slice
                if (s+offset, e+offset) in encountered[base_str_idx]:
                    # We are already going to use this split
                    skip_offset += 1
                    map_back[other_idx] = encountered[base_str_idx][(s+offset, e+offset)]
                else:
                    # This is a new split
                    map_back[other_idx] = other_idx - skip_offset
                    encountered[base_str_idx][(s+offset, e+offset)] = other_idx - skip_offset
                    curr_slice_dict.append((other_idx - skip_offset, (s,e)))

                    if self.retain_positions:
                        current_str = masked_strs[other_idx]
                        current_input_mask = input_masks[other_idx]
                        # SOS + masked_prefix + phrase
                        fin_input_masks.append([False] + [True]*offset + current_input_mask[1:])
                        if self.spaces:
                            base_str = outer_input[base_str_idx].split(" ")
                            if offset != 0:
                                fin_masked_strs.append(" ".join(base_str[:offset]) + " " + current_str)
                            else:
                                fin_masked_strs.append(current_str)
                        else:
                            fin_masked_strs.append(base_str[:offset] + current_str)
                    elif self.causal_only:
                        # collapse all [MASKS] before first true token
                        # takes advantage of Torch acceleration with causal masks
                        fin_input_masks.append([False] + [False]*(e-s+1))
                        if self.spaces:
                            base_str = outer_input[base_str_idx].split(" ")
                            fin_masked_strs.append(" ".join(base_str[s+offset:e+offset+1]))
                        else:
                            fin_masked_strs.append(base_str[:offset] + current_str)
                    else: 
                        fin_masked_strs.append(masked_strs[other_idx])
                        fin_input_masks.append(input_masks[other_idx])
            fin_slice_dict.append(curr_slice_dict)

        if (not batch and self.sample_num == -1):
            pre_tokenized_construct = (batch_sent_tokens * len(masked_strs), batch_idxs * len(masked_strs))
        else:
            str_to_slice = {}

            str_num = 0
            for slice in fin_slice_dict:
                for idx in slice:
                    str_to_slice[idx[0]] = str_num
                str_num += 1

<<<<<<< HEAD
            if self.retain_positions or self.causal_only:
                # recompute tokenization
                batch_sent_tokens, batch_idxs = [], []
                for curr_str in fin_masked_strs:
                    sent_tokens, idxs = get_pre_tokenized_info(curr_str, tokenizer_helper, spaces = self.spaces)
                    batch_sent_tokens.append(sent_tokens)
                    batch_idxs.append(idxs)
                pre_tokenized_construct = tuple([batch_sent_tokens, batch_idxs])
            else:
                # tokenization according to the sampled parses, since those are the input now
                batch_sent_tokens, batch_idxs = [], []
                for curr_str in sampled_input:
                    sent_tokens, idxs = get_pre_tokenized_info(curr_str, tokenizer_helper, spaces = self.spaces)
                    batch_sent_tokens.append(sent_tokens)
                    batch_idxs.append(idxs)

                pre_tokenized_construct = [[], []]
                for _ in range(str_num):
                    num_req = len(fin_slice_dict[_])
                    pre_tokenized_construct[0] += [batch_sent_tokens[_]] * num_req
                    pre_tokenized_construct[1] += [batch_idxs[_]] * num_req
                pre_tokenized_construct = tuple(pre_tokenized_construct)

        if not (self.linear or self.orthogonal):
            inner_context_vecs = get_all_hidden_states_scratch(
                            curr_model,
                            tokenizer_helper,
                            fin_masked_strs,
                            fin_input_masks,
                            sum_all=True,
                            tqdm_disable=True,
                            pre_tokenized=pre_tokenized_construct,
                            layer_id=self.layer_id,
                            start_relax_layer=self.start_relax_layer,
                            end_relax_layer=self.end_relax_layer,
                            is_lm=True,
                            compute_grad=True,
                            diff=self.diff,
                            slice_dict=fin_slice_dict
                        )
=======
            pre_tokenized_construct = [[], []]
            for _ in range(str_num):
                num_req = len(slice_dict[_])
                pre_tokenized_construct[0] += [batch_sent_tokens[_]] * num_req
                pre_tokenized_construct[1] += [batch_idxs[_]] * num_req
            pre_tokenized_construct = tuple(pre_tokenized_construct)

            if not self.linear:
                inner_context_vecs = get_all_hidden_states_scratch(
                                curr_model,
                                tokenizer_helper,
                                masked_strs,
                                input_masks,
                                sum_all=True,
                                start_relax_layer=start_relax_layer,
                                tqdm_disable=True,
                                pre_tokenized=pre_tokenized_construct,
                                layer_id=-1,
                                is_lm=True,
                                compute_grad=True,
                                diff=self.diff,
                                slice_dict=slice_dict
                            )
>>>>>>> b3ce4da34f2346bdda5d7496d402133864818eee

        if (not batch and self.sample_num == -1):
            all_vector_idxs = outer_context_vecs[0][-1][1:]
            for idx, key in slice_dict[0]:
                st, en = key

<<<<<<< HEAD
                if self.orthogonal:
                    scores[key] = torch.mean(self.get_orthogonal_score(all_vector_idxs, st, en))
                else:
=======
                fully_contextual_vectors = self.get_fully_contextual_vectors(all_vector_idxs, st, en)
                if (self.linear):
                    if (st == 0):
                        scores[key] = 0
                        continue
                    else:
                        inner_context_vectors = all_vector_idxs[st]
                else:
                    inner_context_vectors = inner_context_vecs[idx][-1]
                scores[key] = self.sim_metric(
                    inner_context_vectors, fully_contextual_vectors
                )
        else:
            for str_idx in range(str_num):
                all_vector_idxs = outer_context_vecs[str_idx][-1]
                for idx, key in slice_dict[str_idx]:
                    st, en = key

>>>>>>> b3ce4da34f2346bdda5d7496d402133864818eee
                    fully_contextual_vectors = self.get_fully_contextual_vectors(all_vector_idxs, st, en)
                    if (self.linear):
                        # problem, constituents will have dependencies before them in the sentence
                        if (st == 0):
                            scores[key] = 0
                            continue
                        else:
                            inner_context_vectors = all_vector_idxs[st - 1]
                    else:
                        inner_context_vectors = inner_context_vecs[idx][-1]
                    scores[key] = self.sim_metric(
                        inner_context_vectors, fully_contextual_vectors
                    )
        else:
            if self.depth_dep:
                scores = [{l:_ for l in range(2, self.layer_id + 1)} for _ in scores]
            if self.orthogonal:
                # First, for each string, precompute what ranges are required
                new_slice_dicts = {}
                curr_outer_idx = 0
                for str_idx in range(str_num):
                    (outer_idx, offset) = index_info[str_idx]
                    if outer_idx != curr_outer_idx:
                        # Do the computation now
                        # print(new_slice_dicts)
                        if self.depth_dep:
                            for l in range(2, self.layer_id):
                                all_vector_idxs = outer_context_vecs[l][curr_outer_idx][-1]
                                if self.proj:
                                    all_vector_idxs = self.proj_M(all_vector_idxs)
                                score_dict = self.get_batched_orthogonal_scores(all_vector_idxs, new_slice_dicts)
                                for (in_str_idx, in_key) in score_dict.keys():
                                    scores[in_str_idx][l][in_key] = score_dict[(in_str_idx, in_key)]
                        else:
                            all_vector_idxs = outer_context_vecs[curr_outer_idx][-1]
                            if self.proj:
                                    all_vector_idxs = self.proj_M(all_vector_idxs)
                            score_dict = self.get_batched_orthogonal_scores(all_vector_idxs, new_slice_dicts)
                            for (in_str_idx, in_key) in score_dict.keys():
                                scores[in_str_idx][in_key] = score_dict[(in_str_idx, in_key)]
                        curr_outer_idx = outer_idx
                        new_slice_dicts = {}
                    # Update the slice dict
                    for idx, key in slice_dict[str_idx]:
                        st, en = key
                        # required to recover the final scores
                        new_slice_dicts[(str_idx, key)] = (st + offset, en + offset)
                if self.depth_dep:
                    for l in range(2, self.layer_id):
                        all_vector_idxs = outer_context_vecs[l][curr_outer_idx][-1]
                        if self.proj:
                            all_vector_idxs = self.proj_M(all_vector_idxs)
                        score_dict = self.get_batched_orthogonal_scores(all_vector_idxs, new_slice_dicts)
                        for (str_idx, key) in score_dict.keys():
                            scores[str_idx][l][key] = score_dict[(str_idx, key)]
                else:
                    all_vector_idxs = outer_context_vecs[curr_outer_idx][-1]
                    if self.proj:
                        all_vector_idxs = self.proj_M(all_vector_idxs)
                    score_dict = self.get_batched_orthogonal_scores(all_vector_idxs, new_slice_dicts)
                    for (str_idx, key) in score_dict.keys():
                        scores[str_idx][key] = score_dict[(str_idx, key)]
            else:
                for str_idx in range(str_num):
                    # corresponding outer_input index and offset
                    (outer_idx, offset) = index_info[str_idx]
                    all_vector_idxs = outer_context_vecs[outer_idx][-1]
                    for idx, key in slice_dict[str_idx]:
                        st, en = key
                        fully_contextual_vectors = self.get_fully_contextual_vectors(all_vector_idxs, st + offset, en + offset)
                        if (self.linear):
                            # problem, constituents will have dependencies before them in the sentence
                            if (st == 0):
                                scores[str_idx][key] = 0
                                continue
                            else:
                                inner_context_vectors = all_vector_idxs[st]
                        else:
                            # get the correct index in the computed array
                            inner_context_vectors = inner_context_vecs[map_back[idx]][-1]
                        scores[str_idx][key] = self.sim_metric(
                            inner_context_vectors, fully_contextual_vectors
                        )

        return scores, sampled_input, parses, depths
    
    def get_best_and_mean(self, scores, tau, st):
        # for linear, choose the minimum here
        # take abs of everything
        # objective becomes to minimize abs(min) - mean(abs(scores))
        if self.linear:
            scores = torch.abs(scores)
            if (self.gumbel):
                # Soft split decision
                softmaxed = F.gumbel_softmax(-scores, tau=tau, hard=True)
                best = torch.argmax(softmaxed) + st
                best_score = torch.sum(softmaxed*scores)
                tot_score = torch.sum(scores)
                tot_score = tot_score - best_score
            else:
                # Greedy split decision
                best = torch.argmin(scores) + st
                best_score = torch.min(scores)
                tot_score = torch.sum(scores)
                tot_score = tot_score - best_score
        else:
            if (self.gumbel):
                # Soft split decision
                softmaxed = F.gumbel_softmax(scores, tau=tau, hard=True)
                best = torch.argmax(softmaxed) + st
                best_score = torch.sum(softmaxed*scores)
                tot_score = torch.sum(scores)
                tot_score = tot_score - best_score
            else:
                # Greedy split decision
                best = torch.argmax(scores) + st
                best_score = torch.max(scores)
                tot_score = torch.sum(scores)
                tot_score = tot_score - best_score

<<<<<<< HEAD
        return best, best_score, tot_score

    def recurse(self, dataset, chart, curr_str, st, en, tau, mean, use_gold, print_parse, device, depth):
        # Helper for tree regularization score computation
        # Contains logic for normal computation as well as gold parse computation for SimPL
        if (en - st <= 2):
            return 0, 0
=======
    def recurse(self, chart, curr_str, st, en, tau, mean, use_gold, print_parse, device):
        # Helper for tree regularization score computation
        if (st == en):
            return 0
>>>>>>> b3ce4da34f2346bdda5d7496d402133864818eee
        else:
            if (use_gold):
                # enforce LR parsing (addmult only)
                if (len(curr_str) == 3):
                    # single digit
<<<<<<< HEAD
                    return 0, 0
=======
                    return 0
>>>>>>> b3ce4da34f2346bdda5d7496d402133864818eee
                curr_phrase = curr_str[st : en+1]
                if (print_parse):
                    print(curr_phrase)
                if (len(curr_phrase) <= 5):
                    # depth 1 subexpression, don't need to enforce
<<<<<<< HEAD
                    return 0, 0
=======
                    return 0
>>>>>>> b3ce4da34f2346bdda5d7496d402133864818eee
                # atleast one non-trivial bracket left
                if (curr_phrase[2] == '('):
                    # (+(...)...)
                    end_brack = -1
                    enc = 1
                    for idx in range(3,len(curr_phrase)):
                        if (curr_phrase[idx] == ')'):
                            end_brack = idx
                            enc -= 1
                        elif (curr_phrase[idx] == '('):
                            enc += 1
                        if (enc == 0):
                            break
                else:
                    # (+d(...))
                    end_brack = 2

<<<<<<< HEAD
                if self.depth_dep:
                    curr_chart = chart[max(2, self.layer_id - depth)]
                else:
                    curr_chart = chart

                tot_score = 0
                for k in range(st+2, en-1):
                    cand_score = curr_chart[(st+2,k)] + curr_chart[(k+1,en-1)]
                    if self.linear:
                        tot_score += torch.abs(cand_score)
                    else:
                        tot_score += cand_score

                s1, n1 = self.recurse(dataset, chart, curr_str, st+2, st+end_brack, tau, mean, use_gold, print_parse, device, depth + 1)
                s2, n2 = self.recurse(dataset, chart, curr_str, st + end_brack + 1, en-1, tau, mean, use_gold, print_parse, device, depth + 1)
                
                des_score = curr_chart[(st+2, st+end_brack)] - curr_chart[(st + end_brack + 1, en-1)]
                if self.linear:
                    des_score = torch.abs(des_score)

                tot_score = tot_score - des_score
                
                if mean:
                    norm_score = des_score - (tot_score/(en-st-4))
                else:
                    k_rand = randint(st+2, en-3)
                    if self.linear:
                        norm_score = des_score - torch.abs(curr_chart[(st+2,k_rand)] + curr_chart[(k_rand+1,en-2)])
                    else:
                        norm_score = des_score - curr_chart[(st+2,k_rand)] - curr_chart[(k_rand+1,en-2)]
            else:
                # Normal computation
                scores = []

                word_list = curr_str.split(" ")
                # Check if it is actually one word
                if dataset in ["bllip-lg", "bllip-md", "bllip-int"]:
                    one_word = True
                    for k in range(st+1,en+1):
                        if word_list[k][0] == 'Ġ':
                            one_word = False
                            break
                    if one_word:
                        return 0, 0
                
                if self.depth_dep:
                    curr_chart = chart[max(2, self.layer_id - depth)]
                else:
                    curr_chart = chart

                offset = 0
                maint_offsets = []
                for k in range(st, en):
                    cand_score = curr_chart[(st,k)] + curr_chart[(k+1,en)]

                    # Ensure that words are not split! 
                    if dataset in ["bllip-lg", "bllip-md", "bllip-int"]:
                        # start of second split should have 'G.'
                        if word_list[k+1][0] != 'Ġ':
                            offset += 1
                            continue
                        maint_offsets.append(offset)

                    scores.append(cand_score)
                
                if len(scores) < 2:
                    return 0, 0
                
                scores = torch.stack(scores)
                
                best, best_score, tot_score = self.get_best_and_mean(scores, tau, st)

                if (print_parse):
                    if (self.spaces):
                        word_list = curr_str.split(" ")
                        print((" ").join(word_list[st:best+1]), (" ").join(word_list[best+1:en]))
                    else:
                        print(curr_str[st:best+1], curr_str[best+1:en])

                s1, n1 = self.recurse(dataset, chart, curr_str, st, best.item() + maint_offsets[best.item() - st], tau, mean, use_gold, print_parse, device, depth + 1)
                s2, n2 = self.recurse(dataset, chart, curr_str, best.item() + maint_offsets[best.item() - st] + 1, en, tau, mean, use_gold, print_parse, device, depth + 1)
                
                if mean:
                    tot_score = tot_score - best_score
                    norm_score = best_score - (tot_score/(en-st-1))
                    if self.margin == -1:
                        tot_score = tot_score - best_score
                        norm_score = best_score - (tot_score/(en-st-1))
                    # print(st, en, norm_score, best_score, best - st + maint_offsets[best - st])
                    else:
                    # max margin
                        second_best = scores[torch.topk(scores, 2).indices[1]]
                        norm_score = torch.min(best_score - second_best, torch.tensor(self.margin, requires_grad=True, dtype = torch.float).to(scores.device))
                else:
                    k_rand = randint(st, en-1)
                    if self.linear:
                        norm_score = best_score - torch.abs(curr_chart[(st,k_rand)] + curr_chart[(k_rand+1,en)])
                    else:
                        norm_score = best_score - curr_chart[(st,k_rand)] - curr_chart[(k_rand+1,en)]

                if self.neg_samples != -1:
                    # sample some other wrong constituents and drive their SCI scores down
                    samples = set()
                    for _ in range(self.neg_samples):
                        # inter-constituent phrases are always wrong
                        start_phrase = randint(0, best.item() + maint_offsets[best.item() - st])
                        end_phrase = randint(best.item() + maint_offsets[best.item() - st] + 1, len(word_list)-1)
                        samples.add((start_phrase, end_phrase))

                    sub_score = []
                    for phrase in samples:
                        (st, en) = phrase
                        sub_score.append(curr_chart[(st, en)])

                    norm_score = norm_score - self.neg_rel_wt*torch.mean(torch.stack(sub_score))

            return s1 + s2 + norm_score, n1 + n2 + 1
        
    def gold_recurse(self, chart, curr_str, parse, st, en, mean, print_parse, device, depth):
        # Helper for tree regularization score computation given gold decisions
        # If enforce is True, tree reg encourages the correct split decision as well
        if (en - st <= 1):
            return 0, 0, 0
        else:
            scores = []

            word_list = curr_str.split(" ")
            # Check if it is actually one word
            one_word = True
            for k in range(st+1,en+1):
                if word_list[k][0] == 'Ġ':
                    one_word = False
                    break
            if one_word:
                return 0, 0, 0
            
            if self.depth_dep:
                curr_chart = chart[max(2, self.layer_id - depth)]
            else:
                curr_chart = chart

            offset = 0
            maint_offsets = []
            for k in range(st, en):
                cand_score = curr_chart[(st,k)] + curr_chart[(k+1,en)]

                # Ensure that words are not split! 
                # start of second split should have 'G.'
                if word_list[k+1][0] != 'Ġ':
                    offset -= 1
                    maint_offsets.append(offset)
                    continue
                
                maint_offsets.append(offset)
                scores.append(cand_score)
            
            if len(scores) < 2:
                return 0, 0, 0

            scores = torch.stack(scores)
            
            curr_span = str(st) + " " + str(en + 1)
            best = parse[curr_span] - 1
            best_score = scores[best - st + maint_offsets[best - st]] # parse value 2 for [1,4] means [1,1] [2,4]

            # implement an accuracy measure here for the decisions
            # need to see where it derails
            is_best = False

            if self.linear:
                best_score = torch.abs(best_score)
                if best_score == torch.min(torch.abs(scores)):
                    is_best = True
                tot_score = torch.sum(torch.abs(scores))
            else:
                if best_score == torch.max(scores):
                    is_best = True
                tot_score = torch.sum(scores)

            if (print_parse):
                if (self.spaces):
                    word_list = curr_str.split(" ")
                    print((" ").join(word_list[st:best+1]), (" ").join(word_list[best+1:en]))
                else:
                    print(curr_str[st:best+1], curr_str[best+1:en])

            s1, n1, r1 = self.gold_recurse(chart, curr_str, parse, st, best, mean, print_parse, device, depth + 1)
            s2, n2, r2 = self.gold_recurse(chart, curr_str, parse, best+1, en, mean, print_parse, device, depth + 1)

            if mean:
                if self.margin == -1:
                    tot_score = tot_score - best_score
                    norm_score = best_score - (tot_score/(en-st-1))
                # print(st, en, norm_score, best_score, best - st + maint_offsets[best - st])
                else:
                # max margin
                    if is_best:
                        second_best = scores[torch.topk(scores, 2).indices[1]]
                        norm_score = torch.min(best_score - second_best - torch.tensor(self.margin, requires_grad=True, dtype = torch.float).to(scores.device), 0)[0]
                    else:
                        norm_score = best_score - torch.max(scores) - torch.tensor(self.margin, requires_grad=True, dtype = torch.float).to(scores.device)
            elif self.ce:
                norm_score = -F.cross_entropy(scores, torch.tensor(best - st + maint_offsets[best - st], dtype = torch.int64).to(scores.device))
            else:
                k_rand = randint(st, en-1)
                if self.linear:
                    norm_score = best_score - torch.abs(curr_chart[(st,k_rand)] + curr_chart[(k_rand+1,en)])
                else:
                    norm_score = best_score - curr_chart[(st,k_rand)] - curr_chart[(k_rand+1,en)]

            if self.neg_samples != -1:
                # sample some other wrong constituents and drive their SCI scores down
                samples = set()
                for _ in range(self.neg_samples):
                    # inter-constituent phrases are always wrong
                    start_phrase = randint(0, best)
                    end_phrase = randint(best + 1, len(word_list)-1)
                    samples.add((start_phrase, end_phrase))

                sub_score = []
                for phrase in samples:
                    (st, en) = phrase
                    sub_score.append(curr_chart[(st, en)])

                norm_score = norm_score - self.neg_rel_wt*torch.mean(torch.stack(sub_score))

        return s1 + s2 + norm_score, n1 + n2 + 1, r1 + r2 + int(is_best)
    
    def get_score_single(self, dataset, chart, curr_str, end, device, tau=1, mean=True, use_gold=False):
        # Only root-level decision
        if use_gold:
            # enforce LR parsing (addmult only)
            curr_phrase = curr_str
            # print(curr_phrase)
            if (len(curr_phrase) <= 2):
                # depth 1 subexpression, don't need to enforce
                score = torch.tensor(0, requires_grad = True, dtype = torch.float).to(device)
                return score
            # atleast one non-trivial bracket left
            if (curr_phrase[0] == '('):
                # (+(...)...)
                end_brack = -1
                enc = 1
                for idx in range(1,len(curr_phrase)):
                    if (curr_phrase[idx] == ')'):
                        end_brack = idx
                        enc -= 1
                    elif (curr_phrase[idx] == '('):
                        enc += 1
                    if (enc == 0):
                        break
            else:
                # (+d(...))
                end_brack = 0

            tot_score = 0
            for k in range(0, end-1):
                cand_score = chart[(0,k)] + chart[(k+1,end)]
                if self.linear:
                    tot_score += torch.abs(cand_score)
                else:
                    tot_score += cand_score

            des_score = chart[(0, end_brack)] - chart[(end_brack + 1, end)]
            if self.linear: 
                des_score = torch.abs(des_score)

            tot_score = tot_score - des_score
            
            if mean:
                norm_score = des_score - (tot_score/(end - 1))
            else:
                k_rand = randint(0, end-1)
                if self.linear:
                    norm_score = des_score - torch.abs(chart[(0,k_rand)] + chart[(k_rand+1,end)])
                else:
                    norm_score = des_score - chart[(0,k_rand)] - chart[(k_rand+1,end)]
        else:
            curr_phrase = curr_str.split(" ")
            # Check if it is actually one word
            if dataset in ["bllip-lg", "bllip-md", "bllip-int"]:
                one_word = True
                for k in range(0,end+1):
                    if curr_phrase[k][0] == 'Ġ':
                        one_word = False
                        break
                if one_word:
                    return torch.tensor(0, requires_grad = True, dtype = torch.float).to(device)
                    
            if (len(curr_phrase) <= 2):
                # depth 1 subexpression, don't need to enforce
                score = torch.tensor(0, requires_grad = True, dtype = torch.float).to(device)
                return score
            
            # depth computation not possible
            curr_chart = chart

            curr_scores = []
            for k in range(0, end):
                cand_score = curr_chart[(0,k)] + curr_chart[(k+1,end)]

                # Ensure that words are not split! 
                if dataset in ["bllip-lg", "bllip-md", "bllip-int"]:
                    # start of second split should have 'G.'
                    if curr_phrase[k+1][0] != 'Ġ':
                        continue

                curr_scores.append(cand_score)

            if len(curr_scores) < 2:
                return torch.tensor(0, requires_grad = True, dtype = torch.float).to(device)
            
            curr_scores = torch.stack(curr_scores)
            _, best_score, tot_score = self.get_best_and_mean(curr_scores, tau, 0)

            if mean:
                if self.margin == -1:
                    norm_score = best_score - (tot_score/(end - 1))
                else:
                    second_best = curr_scores[torch.topk(curr_scores, 2).indices[1]]
                    norm_score = torch.min(best_score - second_best, torch.tensor(self.margin, requires_grad=True, dtype = torch.float).to(curr_scores.device))
            else:
                k_rand = randint(0, end-1)
                if self.linear:
                    norm_score = best_score - torch.abs(curr_chart[(0,k_rand)] + curr_chart[(k_rand+1,end)])
                else:
                    norm_score = best_score - curr_chart[(0,k_rand)] - curr_chart[(k_rand+1,end)]

            # Not implemented right now, needs some additional tooling
            # if self.neg_samples != -1:
            #     # sample some other wrong constituents and drive their SCI scores down
            #     if (en - st) >= 4: 
            #         # no need to do if consituents aren't large enough
            #         samples = set()
            #         for _ in range(self.neg_samples):
            #             # inter-constituent phrases are always wrong
            #             start_phrase = randint(st, best.item() + maint_offsets[best.item() - st])
            #             end_phrase = randint(best.item() + maint_offsets[best.item() - st] + 1, en-1)
            #             samples.add((start_phrase, end_phrase))

            #         sub_score = []
            #         for phrase in samples:
            #             (st, en) = phrase
            #             sub_score.append(curr_chart[(st, en)])

            #         norm_score = norm_score - torch.mean(torch.stack(sub_score))

        score = norm_score
        return score
        
    def get_score_single_gold(self, chart, parse, curr_str, end, depth, device, mean=True):
        # These are no longer in-position in string, is this the issue?
        # Only root-level decision with gold parses
        curr_phrase = curr_str.split(" ")
        # Check if it is actually one word
        one_word = True
        for k in range(0,end+1):
            # use this only for certain datasets
            if curr_phrase[k][0] == 'Ġ':
                one_word = False
                break
        if one_word:
            return torch.tensor(0, requires_grad = True, dtype = torch.float).to(device), -1
                
        if (len(curr_phrase) <= 2):
            # depth 1 subexpression, don't need to enforce
            score = torch.tensor(0, requires_grad = True, dtype = torch.float).to(device)
            return score, -1
        
        if self.depth_dep:
            curr_chart = chart[max(2, self.layer_id - depth)]
        else:
            curr_chart = chart

        curr_scores = []
        offsets = []
        offset = 0
        for k in range(0, end):
            cand_score = curr_chart[(0,k)] + curr_chart[(k+1,end)]

            # start of second split should have 'G.'
            if curr_phrase[k+1][0] != 'Ġ':
                offset -= 1
                offsets.append(offset)
                continue

            offsets.append(offset)
            curr_scores.append(cand_score)

        if len(curr_scores) < 2:
            return torch.tensor(0, requires_grad = True, dtype = torch.float).to(device), -1

        curr_scores = torch.stack(curr_scores)

        curr_span = "0 {}".format(end+1)
        best = parse[curr_span] - 1 
        best_score = curr_scores[best + offsets[best]] # parse value 1 for [0,4] means [0,0] [1,4]

        is_best = False

        if self.linear:
            best_score = torch.abs(best_score)
            if best_score == torch.min(torch.abs(curr_scores)):
                is_best = True
            tot_score = torch.sum(torch.abs(curr_scores))
        else:
            if best_score == torch.max(curr_scores):
                is_best = True
            tot_score = torch.sum(curr_scores)
        tot_score = tot_score - best_score

        if mean:
            # margin instead?
            if self.margin == -1:
                norm_score = best_score - (tot_score/(end - 1))
            # print(end, norm_score, best_score, parse[curr_span] + offsets[parse[curr_span] - 1] - 1)
            else:
                if is_best:
                    second_best = curr_scores[torch.topk(curr_scores, 2).indices[1]]
                    norm_score = torch.min(best_score - second_best, torch.tensor(self.margin, requires_grad=True, dtype = torch.float).to(curr_scores.device))
                else:
                    norm_score = best_score - torch.max(curr_scores)
        elif self.ce:
                norm_score = -F.cross_entropy(curr_scores, torch.tensor(best + offsets[best], dtype = torch.int64).to(curr_scores.device))
        else:
            k_rand = randint(0, end-1)
            if self.linear:
                norm_score = best_score - torch.abs(curr_chart[(0,k_rand)] + curr_chart[(k_rand+1,end)])
            else:
                norm_score = best_score - curr_chart[(0,k_rand)] - curr_chart[(k_rand+1,end)]

        if self.neg_samples != -1:
            # sample some other wrong constituents and drive their SCI scores down
            if end >= 4 and best > 0 and end-2 > best + 1: 
                # no need to do if consituents aren't large enough
                samples = set()
                for _ in range(self.neg_samples):
                    # inter-constituent phrases are always wrong
                    start_phrase = randint(0, best)
                    end_phrase = randint(best + 1, end - 2)
                    samples.add((start_phrase, end_phrase))

                sub_score = []
                for phrase in samples:
                    (st, en) = phrase
                    sub_score.append(curr_chart[(st, en)])
                norm_score = norm_score - self.neg_rel_wt*torch.mean(torch.stack(sub_score))
    
        return norm_score, int(is_best)

    def get_gold_parse_score_right(self, chart, word_list, parse, st, en):
        # Just follow the gold parse and return appropriate scores
        # Positive score
        # word_list is split curr_str
        if st == en:
            return 0
        
        one_word = True
        for k in range(st+1,en+1):
            if word_list[k][0] == 'Ġ':
                one_word = False
                break
        if one_word:
            return 0
        
        if en == st + 1:
            # trivial decision
            # print((st, en, en))
            return chart[(en, en)]

        curr_span = str(st) + " " + str(en + 1)
        best = parse[curr_span] - 1

        s1 = self.get_gold_parse_score_right(chart, word_list, parse, st, best)
        s2 = self.get_gold_parse_score_right(chart, word_list, parse, best+1, en)

        # scored by right side split
        # print((st, best + 1, en))
        return s1 + s2 + chart[(best + 1, en)]

    def cky_decision_level_loss(self, cky_scores, chart, curr_str, parse, device):
        # instead of a global margin, influence cky scores at each decision

        word_list = curr_str.split(" ")
        score = 0

        for key in parse:
            [s, e] = [int(_) for _ in key.split(" ")]
            g = parse[key]

            best_score = chart[(g, e-1)] + cky_scores[s][g-1] + cky_scores[g][e-1]

            curr_scores = []
            for k in range(s+1,e):
                if word_list[k][0] != 'Ġ' or k == g:
                    continue
                curr_scores.append(chart[(k, e-1)] + cky_scores[s][k-1] + cky_scores[k][e-1])
            
            if len(curr_scores) == 0:
                # score += best_score
                continue

            if self.margin == -1:
                curr_scores = torch.stack(curr_scores)
                norm_score = best_score - (torch.sum(curr_scores)/len(curr_scores))
            elif self.ce:
                curr_scores = torch.stack([best_score] + curr_scores)
                norm_score = -F.cross_entropy(curr_scores, torch.tensor(0, dtype = torch.int64).to(curr_scores.device))
            else:
                curr_scores = torch.stack(curr_scores)
                if torch.max(curr_scores).item() < best_score:
                    norm_score = torch.min(best_score - torch.max(curr_scores), torch.tensor(self.margin, requires_grad=True, dtype = torch.float).to(curr_scores.device))
                else:
                    norm_score = best_score - torch.max(curr_scores)

            score += norm_score

        return score


    def cky_reg(self, chart, curr_str, parse, end, device):
        # Version of tree reg that uses margin loss on CKY-style parsesW

        def decode_cky_parse_list(chart, word_list, cky_decisions, st, en, decision_list):
            # return bracketed parse from parse_decisions
            if cky_decisions[st][en] == -1:
                # leaf of parse
                decision_list.append((st, en+1))
                return 0
            else:
                split = cky_decisions[st][en]
                s1 = decode_cky_parse_list(chart, word_list, cky_decisions, st, split, decision_list)
                s2 = decode_cky_parse_list(chart, word_list, cky_decisions, split+1, en, decision_list)
                decision_list.append((st, en+1))
                return s1 + s2 + chart[(split+1, en)]

        word_list = curr_str.split(" ")
        positive_score = self.get_gold_parse_score_right(chart, word_list, parse, 0, end)
        new_chart = {}
        for _ in chart:
            new_chart[_] = chart[_].item()
        # new_chart = chart
        cky_scores, cky_decisions = self.run_cky(new_chart, word_list, end, gpu=False)
        
        if self.cky_dec:
            gpu_cky_scores = [[0 for _ in range(end + 1)] for _ in range(end + 1)]

            for span_length in range(1, end+1):
                for start in range(end - span_length + 1):
                    st = start
                    en = start + span_length

                    one_word = True
                    for k in range(st+1,en+1):
                        if word_list[k][0] == 'Ġ':
                            one_word = False
                            break
                    if one_word:
                        # the score stays 0, do not split here
                        continue
                    
                    split_point = cky_decisions[st][en]
                    gpu_cky_scores[st][en] = chart[(split_point+1,en)] + gpu_cky_scores[st][split_point] + gpu_cky_scores[split_point+1][en]

        tot_terms = len(parse.keys())
        cky_parse = []
        negative_score = decode_cky_parse_list(chart, word_list, cky_decisions, 0, end, cky_parse)
        tot_correct = 0
        for key in parse.keys():
            st, en = key.split(" ")
            if (int(st), int(en)) in cky_parse:
                tot_correct += 1
        
        if self.cky_dec:
            score = self.cky_decision_level_loss(gpu_cky_scores, chart, curr_str, parse, device)
            # score = self.cky_decision_level_loss(cky_scores, chart, curr_str, parse, device)
        else:
            score = positive_score - self.neg_rel_wt*negative_score

        return score, tot_terms, tot_correct

    def negative_sample_reg(self, chart, curr_str, parse, end, device):
        # just sample negative spans to train tree reg, nothing fancy
        word_list = curr_str.split(" ")
        positive_score = self.get_gold_parse_score_right(chart, word_list, parse, 0, end)
        
        # sample constituents that can never be part of the parse score
        negative_score = 0
        if self.margin != -1:
            # get topk samples and treat them as negatives
            filtered_scores_dict = {}
            tot_correct = 0
            for key in chart:
                (s,e) = key
                # if f"{s} {e+1}" in parse:
                #     tot_correct += 1
                #     # continue
                filtered_scores_dict[chart[key].item()] = f"{s} {e+1}"
                # filtered_scores.append(chart[key])
            # filtered_scores = torch.stack(filtered_scores)
            filtered_scores = torch.stack(list(chart.values()))
            topk = torch.topk(filtered_scores, min(self.neg_samples, len(filtered_scores)))
            negative_score = torch.sum(topk.values)
            cons = 0
            for val in topk.values:
                cons += 1
                if cons > len(parse.keys()):
                    break
                if filtered_scores_dict[val.item()] in parse:
                    tot_correct += 1
        else:
            for _ in range(self.neg_samples):
                cand_s = randint(0, end - 1)
                cand_e = randint(cand_s+1, end)
                if f"{cand_s} {cand_e+1}" in parse:
                    continue
                negative_score += chart[(cand_s, cand_e)]
                tot_correct = -1
        tot_terms = len(parse.keys())

        return positive_score - self.neg_rel_wt*negative_score, tot_terms, tot_correct

    def get_score(self, score_chart, dataset, parses=None, depths=None, tau=1, mean=True, input_str=None, use_gold=False, print_parse=False):
=======
                tot_score = 0
                for k in range(st+2, en-1):
                    cand_score = chart[(st+2,k)] + chart[(k+1,en-1)]
                    tot_score += cand_score

                s1 = self.recurse(chart, curr_str, st+2, st+end_brack, tau, mean, use_gold, print_parse, device)
                s2 = self.recurse(chart, curr_str, st + end_brack + 1, en-1, tau, mean, use_gold, print_parse, device)
                
                if mean:
                    norm_score = chart[(st+2, st+end_brack)] + chart[(st + end_brack + 1, en-1)] - (tot_score/(en-st-3))
                else:
                    k_rand = randint(st+2, en-3)
                    norm_score = chart[(st+2, st+end_brack)] + chart[(st + end_brack + 1, en-1)] - chart[(st+2,k_rand)] - chart[(k_rand+1,en-2)]
            else:
                # Normal computation
                scores = []
                for k in range(st, en):
                    cand_score = chart[(st,k)] + chart[(k+1,en)]
                    scores.append(cand_score)
                
                scores = torch.tensor(scores, requires_grad = True, dtype = torch.float).to(device)
                if (self.gumbel):
                    # Soft split decision
                    softmaxed = F.gumbel_softmax(scores, tau=tau, hard=True)
                    best = (torch.argmax(softmaxed) + st).item()
                    best_score = torch.sum(softmaxed*scores)
                    tot_score = torch.sum(scores)
                else:
                    # Greedy split decision
                    best = (torch.argmax(scores) + st).item()
                    best_score = torch.max(scores)
                    tot_score = torch.sum(scores)

                if (print_parse):
                    if (self.spaces):
                        word_list = curr_str.split(" ")
                        print((" ").join(word_list[st:best+1]), (" ").join(word_list[best+1:en]))
                    else:
                        print(curr_str[st:best+1], curr_str[best+1:en])

                s1 = self.recurse(chart, curr_str, st, best, tau, mean, use_gold, print_parse, device)
                s2 = self.recurse(chart, curr_str, best+1, en, tau, mean, use_gold, print_parse, device)
                
                if mean:
                    norm_score = best_score - (tot_score/(en-st))
                else:
                    k_rand = randint(st, en-1)
                    norm_score = best_score - chart[(st,k_rand)] - chart[(k_rand+1,en)]

            norm_score = min(norm_score, self.hinge_const)
            return s1 + s2 + norm_score

    def get_score(self, score_chart, tau=1, mean=True, input_str=None, use_gold=False, print_parse=False):
>>>>>>> b3ce4da34f2346bdda5d7496d402133864818eee
        # Compute tree regularization scores
        # print_parse: when enabled, print subtrees after every split decision.
        # tau: Temparature for gumbel softmax.
        # mean: Expectation approximation using mean of scores.

        scores = []
        device = torch.device("cuda")
        tot = 0
        tot_correct = 0
        for idx, chart in enumerate(score_chart):
<<<<<<< HEAD
            if self.depth_dep:
                end = max([key[1] for key in chart[2]])
            else:
                end = max([key[1] for key in chart])
=======
            end = max([key[1] for key in chart])
>>>>>>> b3ce4da34f2346bdda5d7496d402133864818eee
            curr_str = input_str[idx]
            if (print_parse):
                print(curr_str)
            if (self.single):
<<<<<<< HEAD
                if (parses is not None):
                    parse = parses[idx]
                    depth = depths[idx]
                    score, right = self.get_score_single_gold(chart, parse, curr_str, end, depth, device, mean)
                    if right == -1:
=======
                # Only root-level decision
                if use_gold:
                    # enforce LR parsing (addmult only)
                    curr_phrase = curr_str
                    # print(curr_phrase)
                    if (len(curr_phrase) <= 2):
                        # depth 1 subexpression, don't need to enforce
                        score = torch.tensor(0, requires_grad = True, dtype = torch.float).to(device)
                        scores.append(score.float())
>>>>>>> b3ce4da34f2346bdda5d7496d402133864818eee
                        continue
                    else:
<<<<<<< HEAD
                        tot_correct += right
                        tot += 1
                else:
                    score = self.get_score_single(dataset, chart, curr_str, end, device, tau, mean, use_gold)
            else:
                # Run normal computation
                if (parses is not None):
                    parse = parses[idx]
                    # pdb.set_trace()
                    if self.neg_only:
                        score, tot_terms, tot_right = self.negative_sample_reg(chart, curr_str, parse, end, device)
                    elif self.cky:
                        score, tot_terms, tot_right = self.cky_reg(chart, curr_str, parse, end, device)
=======
                        # (+d(...))
                        end_brack = 0

                    tot_score = 0
                    for k in range(0, end-1):
                        cand_score = chart[(0,k)] + chart[(k+1,end)]
                        tot_score += cand_score
                    
                    if mean:
                        norm_score = chart[(0, end_brack)] + chart[(end_brack + 1, end)] - (tot_score/end)
>>>>>>> b3ce4da34f2346bdda5d7496d402133864818eee
                    else:
                        score, tot_terms, tot_right = self.gold_recurse(chart, curr_str, parse, 0, end, mean, print_parse, device, 0)
                    tot_terms = max(tot_terms,1)
                    # this is different
                    score /= tot_terms
                    tot_correct += tot_right
                    tot += tot_terms
                else:
                    score, tot_terms = self.recurse(dataset, chart, curr_str, 0, end, tau, mean, use_gold, print_parse, device, 0)
                    tot_terms = max(tot_terms,1)
                    score /= tot_terms
            if (score == 0):
                score = torch.tensor(0, requires_grad = True, dtype = torch.float).to(device)
            scores.append(score)
        # pdb.set_trace()
        if tot==0:
            return scores, None
        else:
            return scores, tot_correct/tot
    
    def get_parse(self, input_str, score_chart, override=False):
        # Only used for BLLIP
        # Just get parses from tree regularization without any approximations
        if self.cky and not override:
            # override this during analysis for greedy parse
            return self.get_parse_cky(input_str, score_chart)

        def recurse(chart, word_list, st, en, depth):
            if (st == en):
                return word_list[st], 0
            else:
                one_word = True
                for k in range(st+1,en+1):
                    if word_list[k][0] == 'Ġ':
                        one_word = False
                        break
                if one_word:
                    return "".join(word_list[st:en+1]), 0
                
                # Normal computation
                if self.depth_dep:
                    curr_chart = chart[max(2, self.layer_id - depth)]
                else:
                    curr_chart = chart

                scores = []
                offset = 0
                maint_offsets = []
                for k in range(st, en):
                    print(k, curr_chart[(st,k)], curr_chart[(k+1,en)])
                    cand_score = curr_chart[(st,k)] + curr_chart[(k+1,en)]
                    if word_list[k+1][0] != 'Ġ':
                        offset += 1
                        continue
                    maint_offsets.append(offset)
                    scores.append(cand_score)
                
                scores = torch.stack(scores)
                if self.linear:
                    scores = torch.abs(scores)
                    best_idx = torch.argmin(scores)
                else:
                    best_idx = torch.argmax(scores)
                best = best_idx.item() + st

                print(st, en, best_idx)

                p1, s1 = recurse(chart, word_list, st, best+maint_offsets[best - st], depth+1)
                p2, s2 = recurse(chart, word_list, best+1+maint_offsets[best - st], en, depth+1)

                return (p1, p2), scores[best_idx] - torch.mean(scores) + s1 + s2

        parses = []
        scores = []
        for idx, chart in enumerate(score_chart):
            # print(chart[0])
            if self.depth_dep:
                end = max([key[1] for key in chart[2]])
            else:
                end = max([key[1] for key in chart])
            curr_str = input_str[idx]
            word_list = curr_str.split(" ")
            parse, score = recurse(chart, word_list, 0, end, 0)
            parses.append(parse)
            scores.append(score)

        return parses, scores, None
    
    def get_parse_beam(self, input_str, score_chart, topk=5):
        
        def expand_state(word_list, chart, score, parse_decisions, frontier):
            is_complete = True
            candidates = []
            for split in frontier:
                (st, en, depth) = split
                if (st == en):
                    continue
                one_word = True
                for k in range(st+1,en+1):
                    if word_list[k][0] == 'Ġ':
                        one_word = False
                        break
                if one_word:
                    continue

                # there are still parsing decisions left
                is_complete = False

                if self.depth_dep:
                    curr_chart = chart[max(2, self.layer_id - depth)]
                else:
<<<<<<< HEAD
                    curr_chart = chart

                scores = []
                offset = 0
                maint_offsets = []
                for k in range(st, en):
                    cand_score = chart[(st,k)] + chart[(k+1,en)]
                    if word_list[k+1][0] != 'Ġ':
                        offset += 1
=======
                    if (len(curr_str) <= 1):
                        # depth 1 subexpression, don't need to enforce
                        score = torch.tensor(0, requires_grad = True, dtype = torch.float).to(device)
                        scores.append(score.float())
>>>>>>> b3ce4da34f2346bdda5d7496d402133864818eee
                        continue
                    maint_offsets.append(offset)
                    scores.append(cand_score.item())
                
                topk_indices = np.argpartition(scores, -min(topk, len(scores)))[-min(topk, len(scores)):]

                if len(candidates) == 0:
                    # generate new candidates
                    for idx in topk_indices:
                        new_parse_decisions = parse_decisions.copy()
                        # new parse decision for decoding final parses
                        # (st, idx) (idx, en) is new frontier
                        curr_split_idx = idx + st + 1 + maint_offsets[idx]
                        new_parse_decisions[(st, en)] = curr_split_idx
                        candidates.append((score + scores[idx], {"is_complete": False, "parse_decisions": new_parse_decisions, 
                                                                 "frontier": [(st, curr_split_idx-1, depth+1), (curr_split_idx, en, depth+1)]})) 
                else:
                    # get topk things from current candidates
                    new_candidate_generator = []
                    for c_idx, candidate in enumerate(candidates):
                        candidate_score = candidate[0]
                        for idx in topk_indices:
                            new_candidate_generator.append([candidate_score + scores[idx], (c_idx, idx)])
                    
                    np_candidate_generator = np.array([_[0] for _ in new_candidate_generator])
                    topk_cands = np.argpartition(np_candidate_generator,-min(topk, len(np_candidate_generator)))[-min(topk, len(np_candidate_generator)):]

                    # generate new candidates
                    new_candidates = []
                    for idx in topk_cands:
                        [fin_score, (c_idx, s_idx)] = new_candidate_generator[idx]
                        new_parse_decisions = candidates[c_idx][1]["parse_decisions"].copy()
                        curr_split_idx = s_idx + st + 1 + maint_offsets[s_idx]
                        new_parse_decisions[(st, en)] = curr_split_idx
                        new_candidates.append((fin_score, {
                            "is_complete": False,
                            "parse_decisions": new_parse_decisions,
                            "frontier": candidates[c_idx][1]["frontier"] + [(st, curr_split_idx-1, depth+1), (curr_split_idx, en, depth+1)]
                        }))
                    candidates = new_candidates

            return is_complete, candidates
                

        def decode_parse(word_list, parse_decisions, st, en):
            # return bracketed parse from parse_decisions
            if (st, en) not in parse_decisions:
                # leaf of parse
                return "".join(word_list[st:en+1])
            else:
                split = parse_decisions[(st, en)]
                p1 = decode_parse(word_list, parse_decisions, st, split-1)
                p2 = decode_parse(word_list, parse_decisions, split, en)
                return (p1, p2)

        def run_beam_search(chart, word_list, st, en):
            candidates = [(0, {
                "is_complete": False,
                "parse_decisions": {},
                "frontier": [(st, en, 0)]
            })] # score, [is_complete, parse_decisions, frontier of parse]
            complete_parses = []

            while len(candidates) > 0:
                new_candidates = []
                for candidate in candidates:
                    # run one step of beam search
                    is_complete, expanded_candidates = expand_state(word_list, chart, candidate[0], candidate[1]["parse_decisions"], candidate[1]["frontier"])

                    if is_complete:
                        # this parse is complete, move to complete_parses
                        complete_parses.append(candidate)
                        continue
                    else:
                        # new candidates have been created for the beam search
                        new_candidates += expanded_candidates

                    # retain topk new_candidates
                    new_candidates = np.array(new_candidates)
                    topk_indices = np.argpartition(new_candidates[:,0].squeeze(),-min(len(new_candidates),topk))[-min(len(new_candidates),topk):]
                    new_candidates = list(np.array(new_candidates)[topk_indices, :])

                # next iteration of beam search
                candidates = new_candidates 
            
            # return best parse
            top_idx = np.argmax(np.array(complete_parses)[:,0].squeeze())
            decoded_parse = decode_parse(word_list, complete_parses[top_idx][1]["parse_decisions"], st, en)
            return decoded_parse, complete_parses[top_idx][0]

        parses = []
        scores = []
        for idx, chart in enumerate(score_chart):
            # print(chart[0])
            if self.depth_dep:
                end = max([key[1] for key in chart[2]])
            else:
<<<<<<< HEAD
                end = max([key[1] for key in chart])
            curr_str = input_str[idx]
            word_list = curr_str.split(" ")
            parse, score = run_beam_search(chart, word_list, 0, end)
            parses.append(parse)
            scores.append(score)

        return parses, scores

    def run_cky(self, chart, word_list, end, gpu=False):
        # Run CKY parsing
        cky_scores = [[0 for _ in range(end + 1)] for _ in range(end + 1)]
        cky_decisions = [[-1 for _ in range(end + 1)] for _ in range(end + 1)]

        for span_length in range(1, end+1):
            for start in range(end - span_length + 1):
                st = start
                en = start + span_length

                one_word = True
                for k in range(st+1,en+1):
                    if word_list[k][0] == 'Ġ':
                        one_word = False
                        break
                if one_word:
                    # the score stays 0, do not split here
                    continue
                
                # depth-dependent computation not supported for now, run separate CKYs for each depth to implement
                # CKY[d] derived from CKY[d+1]
                curr_chart = chart

                scores = []
                offset = 0
                maint_offsets = []
                for k in range(st, en):
                    # score of split is changed here to balance number of terms
                    cand_score = curr_chart[(k+1,en)] + cky_scores[st][k] + cky_scores[k+1][en]
                    print(k, curr_chart[(k+1,en)], cky_scores[st][k], cky_scores[k+1][en])
                    if word_list[k+1][0] != 'Ġ':
                        offset += 1
                        continue
                    maint_offsets.append(offset)
                    scores.append(cand_score)
                
                if gpu:
                    scores = torch.stack(scores)
                # if self.linear:
                #     scores = torch.abs(scores)
                #     best_idx = torch.argmin(scores)
                #     best_score = torch.min(scores)
                # else:
                #     best_idx = torch.argmax(scores)
                #     best_score = torch.max(scores)
                # best = best_idx.item() + st
                if gpu:
                    best_idx = torch.argmax(scores).item()
                else:
                    best_idx = np.argmax(scores)
                if gpu:
                    best_score = torch.max(scores)
                else:
                    best_score = max(scores)
                best = best_idx + st
                best += maint_offsets[best - st] # (st, best) (best+1, en)

                print(st, en, best_score, best)
                cky_scores[st][en] = best_score
                cky_decisions[st][en] = best
        
        return cky_scores, cky_decisions

    def get_parse_cky(self, input_str, score_chart):
        
        def decode_cky_parse(word_list, cky_decisions, st, en):
            # return bracketed parse from parse_decisions
            if cky_decisions[st][en] == -1:
                # leaf of parse
                return "".join(word_list[st:en+1])
            else:
                split = cky_decisions[st][en]
                p1 = decode_cky_parse(word_list, cky_decisions, st, split)
                p2 = decode_cky_parse(word_list, cky_decisions, split+1, en)
                return (p1, p2)

        parses = []
        scores = []
        charts = []
        for idx, chart in enumerate(score_chart):
            # print(chart[0])
            if self.depth_dep:
                end = max([key[1] for key in chart[2]])
            else:
                end = max([key[1] for key in chart])
            curr_str = input_str[idx]
            word_list = curr_str.split(" ")
            new_chart = {}
            for _ in chart:
                new_chart[_] = chart[_].item()
            # new_chart = chart
            cky_scores, cky_decisions = self.run_cky(new_chart, word_list, end)
            parses.append(decode_cky_parse(word_list, cky_decisions, 0, end))
            scores.append(cky_scores[0][end])
            charts.append(cky_scores)
=======
                # Run normal computation
                score = self.recurse(chart, curr_str, 0, end, tau, mean, use_gold, print_parse, device)
            if (score == 0):
                score = torch.tensor(0, requires_grad = True, dtype = torch.float).to(device)
            scores.append(score.float())
>>>>>>> b3ce4da34f2346bdda5d7496d402133864818eee

        return parses, scores, charts
