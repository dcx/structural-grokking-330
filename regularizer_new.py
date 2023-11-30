### regularizes a model to encourage structural generalization
from asyncio import as_completed
from xml.etree.ElementTree import Element
from matplotlib.pyplot import streamplot
import torch
from tree_projections.tree_projections_utils import get_all_hidden_states_scratch, get_masking_info, get_pre_tokenized_info
from tqdm import tqdm
from torch.optim import Adam
import torch.nn.functional as F
import collate
from data_utils.dyck_helpers import build_datasets_dyck, eval_callback_dyck
from data_utils.lm_dataset_helpers import build_datasets_lm
from transformer_helpers import create_lm
from random import randint
from time import time

class Chart():
    def __init__(self, sim_metric, tokenizer, spaces, hinge_const, dataset, 
                 sample_num = -1, sample_len = 8, depth_limit=-1, single = False, diff = False, gumbel = False, linear=False):
        # Initialize the SCI chart
        self.sim_metric = sim_metric

        self.tokenizer = tokenizer
        self.train_data_collator = collate.VarLengthCollate(None)
        self.spaces = spaces # False for AddMult!
        self._cache = {}
        self.hinge_const = hinge_const
        self.dataset = dataset
        self.diff = diff
        self.gumbel = gumbel

        # phrase sampler version of regularizer: sample small phrases from the input and calculate
        self.sample_num = sample_num
        self.sample_len = sample_len

        # depth limited version of regularizer: computation only for last d layers of Chart
        self.depth_limit = depth_limit

        # linear pass idea
        self.linear = linear

        # try out top-layer decision idea
        self.single = single

        # R2D2 optimizations for CKY parsing

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
        phrases = set()

        # Guaranteed that str is larger than phrase len
        for iter in range(self.sample_num):
            if (self.dataset in ["ds-addmult-mod10", "dyck"]):
                # ensure phrases are the largest nested subtree
                if (self.dataset == "dyck"):
                    str = str.split(" ")
                start = randint(0, len(str) - self.sample_len - 1)
                start_brack = -1
                for pos in range(start, len(str)):
                    if (self.dataset == "dyck"):
                        if (str[pos][0] == '('):
                            start_brack = pos
                            break
                    else:
                        if (str[pos] == '('):
                            start_brack = pos
                            break
                if (start_brack == -1):
                    if (self.dataset == "dyck"):
                        str = " ".join(str)
                    continue
                depth = 0
                end_brack = -1
                for pos in range(start_brack, len(str)):
                    if (self.dataset == "dyck"):
                        if (str[pos][0] == '('):
                            depth += 1
                        elif (str[pos][-1] == ')'):
                            depth -= 1
                    else:
                        if (str[pos] == '('):
                            depth += 1
                        elif (str[pos] == ')'):
                            depth -= 1
                    if (depth == 0):
                        end_brack = pos
                        break
                if (end_brack == -1):
                    if (self.dataset == "dyck"):
                        str = " ".join(str)
                    continue
                # allow some relaxation in subtree length, as most samplings will be small
                if (end_brack - start_brack > 2*self.sample_len):
                    if (self.dataset == "dyck"):
                        str = " ".join(str)
                    continue
                if (self.dataset == "dyck"):
                    phrase = " ".join(str[start : start + self.sample_len + 1])
                    str = " ".join(str)
                else:
                    phrase = str[start_brack : end_brack + 1]
                phrases.add(phrase)
            else:
                # simple sampling
                split_str = str.split(" ")
                start = randint(0, len(split_str) - self.sample_len - 1)
                phrase = " ".join(split_str[start : start + self.sample_len + 1])
                phrases.add(phrase)
        
        return list(phrases)
    

    def phrase_split(self, input_str, batch, use_gold):
        # Split input into smaller phrases to get past O(n^2) computation

        if batch:
            outer_input = input_str
        else:
            outer_input = [input_str]

        if (self.sample_num != -1):
            # perform sampling
            sampled_input = []
            for str in outer_input:
                if (self.dataset in ["ds-addmult-mod10"]):
                    if (len(str) <= self.sample_len):
                        # Don't sample
                        curr_str =  [str]
                    else:
                        curr_str = self.get_phrases(str)
                    if (use_gold and self.single):
                        # Trim the useless stuff
                        for sampled in curr_str:
                            sampled_input.append(sampled[2:-1])
                    else:
                        sampled_input += curr_str
                else:
                    split_str = str.split(" ")
                    if (len(split_str) <= self.sample_len):
                        # Don't sample
                        sampled_input += [str]
                    else:
                        sampled_input += self.get_phrases(str)
            outer_input = sampled_input
        else:
            if (self.dataset in ["ds-addmult-mod10"]):
                sampled_input = []
                for str in outer_input:
                    sampled_input.append(str)
                outer_input = sampled_input

        # print(outer_input)

        # Init SCI chart
        if (not batch and self.sample_num == -1):
            scores = {}
        else:
            scores = [{} for _ in outer_input]

        return outer_input, scores

    def build_scores(self, input_str, model, start_relax_layer, tqdm_disable=True, parse_splits=None, batch=True, use_gold=False):
        # Build SCI chart
        if batch:
            scores = [{} for _ in input_str]
        else:
            scores = {}
        device = torch.device("cuda")
        # Required for testing
        # model.to(device)
        if (type(model) != torch.nn.Module):
            curr_model = model.model
        else:
            curr_model = model

        def tokenizer_helper(s, add_special_tokens=True):
            # Wrapper to handle LM SOS
            if add_special_tokens:
                return [curr_model.encoder_sos] + self.tokenizer(s)
            else:
                return self.tokenizer(s)

        outer_input, scores = self.phrase_split(input_str, batch, use_gold)

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
                                                            layer_id=-1,
                                                            is_lm=True,
                                                            compute_grad=True,
                                                            spaces = self.spaces
                                                        )


        slice_dict, masked_strs, input_masks = get_masking_info(tokenizer_helper, outer_input, spaces=self.spaces, single=self.single)

        if (not batch and self.sample_num == -1):
            pre_tokenized_construct = (batch_sent_tokens * len(masked_strs), batch_idxs * len(masked_strs))
        else:
            str_to_slice = {}

            str_num = 0
            for slice in slice_dict:
                for idx in slice:
                    str_to_slice[idx[0]] = str_num
                str_num += 1

            pre_tokenized_construct = [[], []]
            for _ in range(str_num):
                num_req = len(slice_dict[_])
                pre_tokenized_construct[0] += [batch_sent_tokens[_]] * num_req
                pre_tokenized_construct[1] += [batch_idxs[_]] * num_req
            pre_tokenized_construct = tuple(pre_tokenized_construct)

            if (not self.linear):
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

        if (not batch and self.sample_num == -1):
            all_vector_idxs = outer_context_vecs[0][-1]
            for idx, key in slice_dict[0]:
                st, en = key

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
                    if (self.linear):
                        if (st == 0):
                            scores[str_idx][key] = 0
                            continue
                        else:
                            inner_context_vectors = all_vector_idxs[st]
                    else:
                        inner_context_vectors = inner_context_vecs[idx][-1]
                    scores[str_idx][key] = self.sim_metric(
                        inner_context_vectors, fully_contextual_vectors
                    )

        return scores, outer_input

    def get_score(self, score_chart, tau=1, mean=True, input_str=None, use_gold=False, print_parse=False):

        def recurse(score_chart, curr_str, st, en):
            if (st == en):
                return 0
            else:
                if (use_gold):
                    # enforce LR parsing (addmult only)
                    if (len(curr_str) == 3):
                        # single digit
                        return 0
                    curr_phrase = curr_str[st : en+1]
                    if (print_parse):
                        print(curr_phrase)
                    if (len(curr_phrase) <= 5):
                        # depth 1 subexpression, don't really need to enforce
                        return 0
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

                    tot_score = 0
                    for k in range(st+2, en-1):
                        cand_score = score_chart[(st+2,k)] + score_chart[(k+1,en-1)]
                        tot_score += cand_score

                    s1 = recurse(score_chart, curr_str, st+2, st+end_brack)
                    s2 = recurse(score_chart, curr_str, st + end_brack + 1, en-1)
                    
                    # try mean instead? closer to expectation
                    if mean:
                        norm_score = chart[(st+2, st+end_brack)] + chart[(st + end_brack + 1, en-1)] - (tot_score/(en-st-3))
                    else:
                        k_rand = randint(st+2, en-3)
                        norm_score = chart[(st+2, st+end_brack)] + chart[(st + end_brack + 1, en-1)] - score_chart[(st+2,k_rand)] - score_chart[(k_rand+1,en-2)]
                else:
                    # Normal computation
                    scores = []
                    for k in range(st, en):
                        cand_score = score_chart[(st,k)] + score_chart[(k+1,en)]
                        scores.append(cand_score)
                    
                    scores = torch.tensor(scores, requires_grad = True, dtype = torch.float).to(device)
                    if (self.gumbel):
                        softmaxed = F.gumbel_softmax(scores, tau=tau, hard=True) # can schedule tau
                        best = (torch.argmax(softmaxed) + st).item()
                        best_score = torch.sum(softmaxed*scores)
                        tot_score = torch.sum(scores)
                    else:
                        best = (torch.argmax(scores) + st).item()
                        best_score = torch.max(scores)
                        tot_score = torch.sum(scores)

                    if (print_parse):
                        if (self.spaces):
                            word_list = curr_str.split(" ")
                            print((" ").join(word_list[st:best+1]), (" ").join(word_list[best+1:en]))
                        else:
                            print(curr_str[st:best+1], curr_str[best+1:en])

                    s1 = recurse(score_chart, curr_str, st, best)
                    s2 = recurse(score_chart, curr_str, best+1, en)
                    
                    # try mean instead? closer to expectation
                    if mean:
                        norm_score = best_score - (tot_score/(en-st))
                    else:
                        k_rand = randint(st, en-1)
                        norm_score = best_score - score_chart[(st,k_rand)] - score_chart[(k_rand+1,en)]

                norm_score = min(norm_score, self.hinge_const)
                return s1 + s2 + norm_score

        scores = []
        device = torch.device("cuda")
        for idx, chart in enumerate(score_chart):
            end = max([key[1] for key in chart])
            # print(chart)
            curr_str = input_str[idx]
            if (print_parse):
                print(curr_str)
            if (self.single):
                if use_gold:
                    # enforce LR parsing (addmult only)
                    curr_phrase = curr_str
                    # print(curr_phrase)
                    if (len(curr_phrase) <= 2):
                        # depth 1 subexpression, don't really need to enforce
                        score = torch.tensor(0, requires_grad = True, dtype = torch.float).to(device)
                        scores.append(score.float())
                        continue
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
                        tot_score += cand_score
                    
                    # try mean instead? closer to expectation
                    if mean:
                        norm_score = chart[(0, end_brack)] + chart[(end_brack + 1, end)] - (tot_score/end)
                    else:
                        k_rand = randint(0, end-1)
                        norm_score = chart[(0, end_brack)] + chart[(end_brack + 1, end)] - chart[(0,k_rand)] - chart[(k_rand+1,end)]
                else:
                    if (len(curr_str) <= 1):
                        # depth 1 subexpression, don't really need to enforce
                        score = torch.tensor(0, requires_grad = True, dtype = torch.float).to(device)
                        scores.append(score.float())
                        continue
                    curr_scores = []
                    for k in range(0, end):
                        cand_score = chart[(0,k)] + chart[(k+1,end)]
                        curr_scores.append(cand_score)
                    
                    curr_scores = torch.tensor(curr_scores, requires_grad = True, dtype = torch.float).to(device)
                    if (self.gumbel):
                        softmaxed = F.gumbel_softmax(curr_scores, tau=tau, hard=True) # can schedule tau
                        best_score = torch.sum(softmaxed*scores)
                        tot_score = torch.sum(scores)
                    else:
                        best_score = torch.max(curr_scores)
                    if mean:
                        norm_score = best_score - (tot_score/(end))
                    else:
                        k_rand = randint(0, end-1)
                        norm_score = best_score - chart[(0,k_rand)] - chart[(k_rand+1,end)]
                score = norm_score
            else:
                score = recurse(chart, curr_str, 0, end)
            if (score == 0):
                score = torch.tensor(0, requires_grad = True, dtype = torch.float).to(device)
            scores.append(score.float())

        return scores
