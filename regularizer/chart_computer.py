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
from data_utils.text_helpers import reformat_sentence
from transformer_helpers import create_lm
from random import randint, sample
import random
import time
import pdb
import numpy as np

random.seed(42)

class OldChartComputer():
    def __init__(self, sim_metric, tokenizer, spaces, phrase_sampler, args):
    
        self.diff = args.use_difference
        self.spaces = spaces
        self.sim_metric = sim_metric
        self.tokenizer = tokenizer
        self.layer_id = args.layer_id
        self.single = args.reg_single
        self.phrase_sampler = phrase_sampler
        self.neg_samples = args.neg_samples
        self.start_relax_layer = args.start_relax_layer
        self.end_relax_layer = args.end_relax_layer
        self.retain_positions = args.retain_positions
        self.causal_only = args.causal_only
        self.sample_num = args.reg_sample_num


    def get_fully_contextual_vectors(self, all_vector_idxs, st, en):
        # Get in-context span representations
        if (self.diff):
            if (st == 0):
                fully_contextual_vectors = all_vector_idxs[en]
            else:
                fully_contextual_vectors = all_vector_idxs[en] - all_vector_idxs[st-1]
            fully_contextual_vectors = fully_contextual_vectors.squeeze()
        else:
            fully_contextual_vectors = all_vector_idxs[st : en + 1].sum(
                axis=0
            )
        return fully_contextual_vectors
    
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
        if (type(model) != torch.nn.Module and not eval):
            curr_model = model.model
        else:
            curr_model = model

        outer_input, sampled_input, index_info, parses, depths, scores = self.phrase_sampler.phrase_split(input_str, parses, batch, use_gold, eval)
        # pdb.set_trace()

        batch_sent_tokens, batch_idxs = [], []
        for curr_str in outer_input:
            sent_tokens, idxs = get_pre_tokenized_info(curr_str, tokenizer_helper, spaces = self.spaces)
            batch_sent_tokens.append(sent_tokens)
            batch_idxs.append(idxs)

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

        if (not batch and self.sample_num == -1):
            all_vector_idxs = outer_context_vecs[0][-1][1:]
            for idx, key in slice_dict[0]:
                st, en = key

                fully_contextual_vectors = self.get_fully_contextual_vectors(all_vector_idxs, st, en)
                inner_context_vectors = inner_context_vecs[idx][-1]
                scores[key] = self.sim_metric(
                    inner_context_vectors, fully_contextual_vectors
                )
        else:
            for str_idx in range(str_num):
                # corresponding outer_input index and offset
                (outer_idx, offset) = index_info[str_idx]
                all_vector_idxs = outer_context_vecs[outer_idx][-1]
                for idx, key in slice_dict[str_idx]:
                    st, en = key
                    fully_contextual_vectors = self.get_fully_contextual_vectors(all_vector_idxs, st + offset, en + offset)
                    # get the correct index in the computed array
                    inner_context_vectors = inner_context_vecs[map_back[idx]][-1]
                    scores[str_idx][key] = self.sim_metric(
                        inner_context_vectors, fully_contextual_vectors
                    )

        return scores, sampled_input, parses, depths
    
class OrthogonalChartComputer():
    def __init__(self, sim_metric, tokenizer, spaces, phrase_sampler, args):
    
        self.diff = args.use_difference
        self.spaces = spaces
        self.sim_metric = sim_metric
        self.tokenizer = tokenizer
        self.layer_id = args.layer_id
        self.single = args.reg_single
        self.sci_heads = args.sci_heads
        self.phrase_sampler = phrase_sampler
        self.neg_samples = args.neg_samples
        self.dataset = args.dataset

        self.orth_single = args.orth_single
        self.orth_comp = args.orth_comp
        self.orth_bidir = args.orth_bidir
        self.sci_heads = args.sci_heads
        self.proj = args.proj
        if self.proj:
            self.proj_M = torch.nn.Linear(args.vec_dim, args.vec_dim)

        # use packing 
        self.use_packing = args.pack
        self.max_sequence_len = args.max_seq_len
        self.hf = args.hf

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

            context = F.normalize(all_vector_idxs[st - 1, :proportion], dim=-1).unsqueeze(0) 

            # Alternative: trying mean of everything that came before
            # previous_info = F.normalize(torch.mean(all_vector_idxs[:st - 1], dim=0), dim=-1)

            # v_perp = v - (v.v1)v1
            # E-S, D   E-S, S  S, D
            # Original
            span_vectors = all_vector_idxs[st:en+1, :proportion] # (E-S), D

            components_along = span_vectors @ context.T

            if self.orth_comp:
                orthogonals = -torch.abs(components_along.squeeze(-1))
            else:
                orthogonals = span_vectors - (components_along @ context)

            # aggregate magnitudes of orthogonals
            return orthogonals
        
    def get_batched_orthogonal_scores(self, outer_vector, slice_dict):
        # Vectorize this out
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
    
    def build_scores(self, input_str, model, parses=None, batch=True, use_gold=False, eval=False):
        # Build SCI chart
        # use_gold: Tree regularization guided by gold parses

        def tokenizer_helper(s, add_special_tokens=True):
            # Wrapper to handle LM SOS and tokenize input
            if self.hf:
                if add_special_tokens:
                    return self.tokenizer(s)["input_ids"]
                else:
                    return self.tokenizer(s)["input_ids"][1:]
            else:
                if add_special_tokens:
                    return [curr_model.encoder_sos] + self.tokenizer(s)
                else:
                    return self.tokenizer(s)
            
        if batch:
            scores = [{} for _ in input_str]
        else:
            scores = {}

        device = torch.device("cuda")
        if (type(model) != torch.nn.Module and not eval):
            curr_model = model.model
        else:
            curr_model = model

        # sample phrases (identity transform if not required)
        outer_input, sampled_input, index_info, parses, depths, scores = self.phrase_sampler.phrase_split(input_str, parses, batch, use_gold, eval)

        batch_sent_tokens, batch_idxs = [], []

        if self.hf:
            new_outer_input = []
            for idx in range(len(outer_input)):
                # remove GPT-2 sow prefix
                words = outer_input[idx].split(" ")
                cleaned_words = []
                curr_word = ""
                for iidx in range(len(words)):
                    if words[iidx][0] == 'Ġ':
                        if iidx != 0:
                            cleaned_words.append(curr_word)
                        curr_word = words[iidx][1:]
                    else:
                        curr_word += words[iidx]
                cleaned_words.append(curr_word)
                new_outer_input.append(" ".join(cleaned_words))
        else:
            new_outer_input = outer_input
                

        for curr_str in new_outer_input:
            # idxs maps start and end of tokens
            sent_tokens, idxs = get_pre_tokenized_info(curr_str, tokenizer_helper, spaces = self.spaces)
            if self.hf:
                # manually create the idxs
                tokenized_length = len(sent_tokens)
                idxs = [(s,s+1) for s in range(1,tokenized_length)]
            batch_sent_tokens.append(sent_tokens)
            batch_idxs.append(idxs)

        outer_context_vecs = get_all_hidden_states_scratch(curr_model,
                                                        tokenizer_helper,
                                                        new_outer_input,
                                                        tqdm_disable=True,
                                                        pre_tokenized=(batch_sent_tokens, batch_idxs),
                                                        layer_id=self.layer_id,
                                                        start_relax_layer=0,
                                                        end_relax_layer=0,
                                                        is_lm=True,
                                                        compute_grad=True,
                                                        spaces = self.spaces,
                                                        use_packing = self.use_packing,
                                                        max_sequence_len = self.max_sequence_len,
                                                        hf = self.hf
                                                    )

        if eval or self.neg_samples != -1 or self.orth_bidir:
            slice_dict, masked_strs, input_masks = get_masking_info(tokenizer_helper, sampled_input, spaces=self.spaces, single=False)
        else:
            slice_dict, masked_strs, input_masks = get_masking_info(tokenizer_helper, sampled_input, spaces=self.spaces, single=self.single)

        if (not batch and self.sample_num == -1):
            # single element in input
            all_vector_idxs = outer_context_vecs[0][-1]
            for idx, key in slice_dict[0]:
                st, en = key

                scores[key] = torch.mean(self.get_orthogonal_score(all_vector_idxs, st, en))
        else:
            str_num = len(slice_dict)
            new_slice_dicts = {}
            curr_outer_idx = 0
            for str_idx in range(str_num):
                (outer_idx, offset) = index_info[str_idx]
                if outer_idx != curr_outer_idx:
                    # Moved on to phrases from a new string
                    # Do the computation now
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
                    # book-keeping required to recover the final scores
                    # This is done because with phrase sampling, there are a lot of coinciding spans
                    new_slice_dicts[(str_idx, key)] = (st + offset, en + offset)

                if self.orth_bidir:
                    # with phrase sampling, might need right term for rightmost constituents (j,j+1)
                    if self.dataset in ["ds-addmult-mod10"]:
                        if offset + len(sampled_input[str_idx]) != len(outer_input[outer_idx]):
                            max_en = max([_[1] for _ in slice_dict[str_idx]])
                            new_slice_dicts[(str_idx, (max_en+1, max_en+1))] = (max_en + 1 + offset, max_en + 1 + offset)
                    else:
                        if offset + len(sampled_input[str_idx].split(" ")) != len(outer_input[outer_idx].split(" ")):
                            max_en = max([_[1] for _ in slice_dict[str_idx]])
                            new_slice_dicts[(str_idx, (max_en+1, max_en+1))] = (max_en + 1 + offset, max_en + 1 + offset)

            all_vector_idxs = outer_context_vecs[curr_outer_idx][-1]
            if self.proj:
                all_vector_idxs = self.proj_M(all_vector_idxs)
            score_dict = self.get_batched_orthogonal_scores(all_vector_idxs, new_slice_dicts)
            for (str_idx, key) in score_dict.keys():
                scores[str_idx][key] = score_dict[(str_idx, key)]

        return scores, sampled_input, parses, depths
    
class BalancedOrthogonalChartComputer():
    def __init__(self, sim_metric, tokenizer, spaces, phrase_sampler, args):
    
        self.diff = args.use_difference
        self.spaces = spaces
        self.sim_metric = sim_metric
        self.tokenizer = tokenizer
        self.layer_id = args.layer_id
        self.single = args.reg_single
        self.sci_heads = args.sci_heads
        self.phrase_sampler = phrase_sampler
        self.neg_samples = args.neg_samples

        self.sci_heads = args.sci_heads
        self.proj = args.proj
        if self.proj:
            self.proj_M = torch.nn.Linear(args.vec_dim, args.vec_dim)

    def get_l2_norm(self, vec):
        return torch.sqrt(torch.sum(vec*vec, dim=-1))

    def cache_orthogonal_components(self, all_vector_idxs):
        # return orthogonals of all hidden states wrt all hidden states before it
        if self.sci_heads == -1:
            proportion = all_vector_idxs.shape[1]
        else:
            proportion = int(self.sci_heads * all_vector_idxs.shape[1])

        normalized_vectors = F.normalize(all_vector_idxs[:, :proportion], dim=-1)
        orthogonals = []
        for st in range(1,all_vector_idxs.shape[0]):
            right_vectors = normalized_vectors[st:, :]
            context_vector = normalized_vectors[st - 1, :].unsqueeze(0) 

            components_along = right_vectors @ context_vector.T
            orthogonals.append(F.normalize(right_vectors - (components_along @ context_vector), dim=-1))
        
        # prepend original vectors to list
        orthogonals = [normalized_vectors] + orthogonals

        # st, en -> orth[st][en-st]
        return orthogonals
    
    def get_orthogonal_score(self, normalized_vectors, st, k, en):
        # rep(k) -> norm_vec[st][k-st]
        # rep(en) -> norm_vec[st][en-st]
        # score -> norm(orth(rep(k), rep(en)) = norm(rep(en) - rep(en).rep(k)rep(k))
        # everything D dimensional

        # v_perp = v - (v.v1)v1
        
        int_repn = normalized_vectors[st][k-st].squeeze()
        end_repn = normalized_vectors[st][en-st].squeeze()

        components_along = torch.dot(int_repn,end_repn)
        
        return end_repn - components_along*int_repn
        
    def get_batched_orthogonal_scores(self, outer_vector, slice_dict):
        # Batch computation by starting indices st
        # Compute max endpoint required for all st
        normalized_vectors = self.cache_orthogonal_components(outer_vector)

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
            for key in ends_req_dict[st]:
                st, en = slice_dict[key]
                score_dict[key] = normalized_vectors[st][en-st].squeeze()

        return score_dict
    
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
        if (type(model) != torch.nn.Module and not eval):
            curr_model = model.model
        else:
            curr_model = model

        outer_input, sampled_input, index_info, parses, depths, scores = self.phrase_sampler.phrase_split(input_str, parses, batch, use_gold, eval)
        # pdb.set_trace()

        batch_sent_tokens, batch_idxs = [], []
        for curr_str in outer_input:
            sent_tokens, idxs = get_pre_tokenized_info(curr_str, tokenizer_helper, spaces = self.spaces)
            batch_sent_tokens.append(sent_tokens)
            batch_idxs.append(idxs)

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
        # pdb.set_trace()

        if eval or self.neg_samples != -1:
            slice_dict, masked_strs, input_masks = get_masking_info(tokenizer_helper, sampled_input, spaces=self.spaces, single=False)
        else:
            slice_dict, masked_strs, input_masks = get_masking_info(tokenizer_helper, sampled_input, spaces=self.spaces, single=self.single)


        str_num = len(slice_dict)
        new_slice_dicts = {}
        curr_outer_idx = 0
        for str_idx in range(str_num):
            (outer_idx, offset) = index_info[str_idx]
            if outer_idx != curr_outer_idx:
                # Do the computation now
                # print(new_slice_dicts)
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
        all_vector_idxs = outer_context_vecs[curr_outer_idx][-1]
        if self.proj:
            all_vector_idxs = self.proj_M(all_vector_idxs)
        score_dict = self.get_batched_orthogonal_scores(all_vector_idxs, new_slice_dicts)
        for (str_idx, key) in score_dict.keys():
            scores[str_idx][key] = score_dict[(str_idx, key)]

        return scores, sampled_input, parses, depths