### regularizes a model to encourage structural generalization

from joblib import parallel_backend
from sklearn.linear_model import orthogonal_mp
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
from regularizer.phrase_sampler import PhraseSampler
from regularizer.chart_computer import OrthogonalChartComputer, OldChartComputer, BalancedOrthogonalChartComputer

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

        # Initialize the SCI chart
        self.sim_metric = sim_metric

        self.tokenizer = tokenizer
        self.train_data_collator = collate.VarLengthCollate(None)
        self.spaces = spaces # False for AddMult!
        self._cache = {}
        self.dataset = args.dataset
        self.diff = args.use_difference
        self.gumbel = args.use_gumbel
        self.layer_id = args.layer_id # which layer IRs to use for SCI computation
        if self.layer_id == -1:
            self.layer_id = args.encoder_n_layers
        self.start_relax_layer = args.start_relax_layer
        self.end_relax_layer = args.end_relax_layer
        self.retain_positions = args.retain_positions
        self.causal_only = args.causal_only

        # loss things
        self.margin = args.margin # Use margin loss for SCI
        self.ce = args.ce # Use cross-entropy loss for SCI
        self.neg_samples = args.neg_samples # Drive scores down for some negatively sampled phrases along with SCI score computation
        self.neg_rel_wt = args.neg_rel_wt

        # phrase sampler version of regularizer: sample small phrases from the input and calculate
        self.sample_num = args.reg_sample_num
        self.sample_len = args.reg_sample_len
        self.sample_subtrees = args.sample_subtrees

        # other versions of tree reg
        self.orthogonal = args.use_orthogonal # This is the current formulation
        self.orth_single = args.orth_single # This always needs to be set in the current formalation. Previously, unsetting this would use all vectors in a span as its representation
        self.balance = args.balance # This can be ignored, SCI(i,k,j) = norm(orth(orth(k,i-1), orth(j,i-1)))
        self.orth_comp = args.orth_comp # Use cosine similarity instead of L2 norm in SCI computation
        self.orth_bidir = args.orth_bidir # SCI(i,j) = norm(orth(j,i-1)) + norm(orth(j+1,j))
        self.sci_heads = args.sci_heads # What proportion of attention heads should be used in SCI computation
        self.proj = args.proj # Project the vectors before computing SCI
        if self.proj:
            self.proj_M = torch.nn.Linear(args.vec_dim, args.vec_dim)

        self.cky = args.cky # SCI loss according to CKY score (at tree level)
        self.cky_dec = args.cky_dec # Subtree-level CKY SCI loss 
        self.neg_only = args.neg_only # SCI loss according to just negative sampling
        if self.cky:
            self.orth_single = True

        # top-layer decision idea
        self.single = args.reg_single # can be ignored

        # class initializations
        self.phrase_sampler = PhraseSampler(spaces, args)

        # initialize scorers
        if self.orthogonal:
            if self.balance:
                self.scorer = BalancedOrthogonalChartComputer(sim_metric, tokenizer, spaces, self.phrase_sampler, args)
            else:
                self.scorer = OrthogonalChartComputer(sim_metric, tokenizer, spaces, self.phrase_sampler, args)
        else:
            self.scorer = OldChartComputer(sim_metric, tokenizer, spaces, self.phrase_sampler, args)

    def build_scores(self, input_str, model, parses=None, batch=True, use_gold=False, eval=False):
        return self.scorer.build_scores(input_str, model, parses, batch, use_gold, eval)
    
    def get_best_and_mean(self, scores, tau, st):
        # Get the best decision by score from a list of decisions
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

        return best, best_score, tot_score
    
    def get_orth_norm(self, v1, v2):
        # return norm of orthogonal component for two vectors
        int_repn = v1.squeeze()
        end_repn = v2.squeeze()

        components_along = torch.dot(int_repn,end_repn)
        
        return torch.norm(end_repn - components_along*int_repn)

    def get_penalty(self, chart, st, k, en, cky=False):
        if cky:
            if self.balance:
                return self.get_orth_norm(chart[(st, k)], chart[(k+1, en)])
            else:
                base_score = chart[(k+1, en)]
                if self.orth_bidir:
                    # SCI(st,k) + SCI(k,en)
                    base_score = base_score + chart[(st, k)] + chart[(k+1, k+1)]
                    if (en, en+1) in chart:
                        base_score = base_score + chart[(en+1, en+1)]
                return base_score
        else:
            # SCI(st,k) + SCI(k,en)
            base_score = chart[(st, k)] + chart[(k+1, en)]
            if self.orth_bidir:
                base_score = base_score + chart[(k+1, k+1)]
                if (en+1, en+1) in chart:
                    base_score = base_score + chart[(en+1, en+1)]
            return base_score

    def recurse(self, dataset, chart, curr_str, st, en, tau, mean, use_gold, print_parse, device, depth):
        # Helper for tree regularization score computation
        # Contains logic for normal computation as well as gold parse computation for SimPL
        if (en - st <= 2):
            return 0, 0
        else:
            if (use_gold):
                # enforce LR parsing (addmult only)
                if (len(curr_str) == 3):
                    # single digit
                    return 0, 0
                curr_phrase = curr_str[st : en+1]
                if (print_parse):
                    print(curr_phrase)
                if (len(curr_phrase) <= 5):
                    # depth 1 subexpression, don't need to enforce
                    return 0, 0
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

                curr_chart = chart

                tot_score = 0
                for k in range(st+2, en-1):
                    cand_score = self.get_penalty(curr_chart, st+2, k, en-1)
                    tot_score += cand_score

                s1, n1 = self.recurse(dataset, chart, curr_str, st+2, st+end_brack, tau, mean, use_gold, print_parse, device, depth + 1)
                s2, n2 = self.recurse(dataset, chart, curr_str, st + end_brack + 1, en-1, tau, mean, use_gold, print_parse, device, depth + 1)
                
                des_score = self.get_penalty(curr_chart, st+2, st+end_brack, en-1)

                tot_score = tot_score - des_score
                
                if mean:
                    norm_score = des_score - (tot_score/(en-st-4))
                else:
                    k_rand = randint(st+2, en-3)
                    norm_score = des_score - self.get_penalty(curr_chart, st+2, k_rand, en-1)
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
                
                curr_chart = chart

                offset = 0
                maint_offsets = []
                for k in range(st, en):
                    cand_score = self.get_penalty(curr_chart, st, k, en)

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
                    norm_score = best_score - self.get_penalty(curr_chart, st, k_rand, en)

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
                        phrase_score = curr_chart[(st, en)]
                        if self.orth_bidir and (en+1, en+1) in curr_chart:
                            phrase_score += curr_chart[(en+1, en+1)]
                        sub_score.append(phrase_score)

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
            
            curr_chart = chart

            offset = 0
            maint_offsets = []
            for k in range(st, en):
                cand_score = self.get_penalty(curr_chart, st, k, en)

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
                    # Margin with mean score
                    tot_score = tot_score - best_score
                    norm_score = best_score - (tot_score/(en-st-1))
                else:
                # max margin
                    if is_best:
                        second_best = scores[torch.topk(scores, 2).indices[1]]
                        norm_score = torch.min(best_score - second_best - torch.tensor(self.margin, requires_grad=True, dtype = torch.float).to(scores.device), 0)[0]
                    else:
                        norm_score = best_score - torch.max(scores) - torch.tensor(self.margin, requires_grad=True, dtype = torch.float).to(scores.device)
            elif self.ce:
                # Cross entropy
                norm_score = -F.cross_entropy(scores, torch.tensor(best - st + maint_offsets[best - st], dtype = torch.int64).to(scores.device))
            else:
                # Margin with random score
                k_rand = randint(st, en-1)
                norm_score = best_score - self.get_penalty(curr_chart, st, k_rand, en)

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
                    phrase_score = curr_chart[(st, en)]
                    if self.orth_bidir and (en+1, en+1) in curr_chart:
                        phrase_score += curr_chart[(en+1, en+1)]
                    sub_score.append(phrase_score)

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
                cand_score = self.get_penalty(chart, 0, k, end)
                tot_score += cand_score

            des_score = self.get_penalty(chart, 0, end_brack, end)

            tot_score = tot_score - des_score
            
            if mean:
                norm_score = des_score - (tot_score/(end - 1))
            else:
                k_rand = randint(0, end-1)
                norm_score = des_score - self.get_penalty(chart, 0, k_rand, end)
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
                cand_score = self.get_penalty(curr_chart, 0, k, end)

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
                norm_score = best_score - self.get_penalty(curr_chart, 0, k_rand, end)

            # Not implemented right now, needs some additional tooling
            # if self.neg_samples != -1:

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
        
        curr_chart = chart

        curr_scores = []
        offsets = []
        offset = 0
        for k in range(0, end):
            cand_score = self.get_penalty(curr_chart, 0, k, end)

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
            norm_score = best_score - self.get_penalty(curr_chart, 0, k_rand, end)

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
                    phrase_score = curr_chart[(st, en)]
                    if self.orth_bidir and (en+1, en+1) in curr_chart:
                        phrase_score += curr_chart[(en+1, en+1)]
                    sub_score.append(phrase_score)
                norm_score = norm_score - self.neg_rel_wt*torch.mean(torch.stack(sub_score))
    
        return norm_score, int(is_best)

    def get_gold_parse_score_right(self, chart, word_list, parse, st, en):
        # Used for tree-level CKY computation
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
            return self.get_penalty(chart, st, st, en, cky=True)

        curr_span = str(st) + " " + str(en + 1)
        best = parse[curr_span] - 1

        s1 = self.get_gold_parse_score_right(chart, word_list, parse, st, best)
        s2 = self.get_gold_parse_score_right(chart, word_list, parse, best+1, en)

        # scored by right side split
        # print((st, best + 1, en))
        return s1 + s2 + self.get_penalty(chart, st, best, en, cky=True)

    def cky_decision_level_loss(self, cky_scores, chart, curr_str, parse, device):
        # instead of a global margin, influence cky scores at each decision

        word_list = curr_str.split(" ")
        score = 0

        for key in parse:
            [s, e] = [int(_) for _ in key.split(" ")]
            g = parse[key]

            best_score = self.get_penalty(chart, s, g-1, e-1, True) + cky_scores[s][g-1] + cky_scores[g][e-1]

            curr_scores = []
            for k in range(s+1,e):
                if word_list[k][0] != 'Ġ' or k == g:
                    # middle of a word
                    continue
                curr_scores.append(self.get_penalty(chart, s, k-1, e-1, True) + cky_scores[s][k-1] + cky_scores[k][e-1])

            if len(curr_scores) == 0:
                # score += best_score
                continue
            
            if self.ce:
                curr_scores = torch.stack([best_score] + curr_scores)
                norm_score = -F.cross_entropy(curr_scores, torch.tensor(0, dtype = torch.int64).to(curr_scores.device))
            elif self.margin == -1:
                curr_scores = torch.stack(curr_scores)
                norm_score = best_score - (torch.sum(curr_scores)/len(curr_scores))
            else:
                curr_scores = torch.stack(curr_scores)
                if torch.max(curr_scores).item() < best_score:
                    norm_score = torch.min(best_score - torch.max(curr_scores), torch.tensor(self.margin, requires_grad=True, dtype = torch.float).to(curr_scores.device))
                else:
                    norm_score = best_score - torch.max(curr_scores)

            score = score + norm_score

        return score


    def cky_reg(self, chart, curr_str, parse, end, device):
        # Version of tree reg that uses margin loss on CKY-style parses

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
                return s1 + s2 + self.get_penalty(chart, st, split, en, True)

        word_list = curr_str.split(" ")
        positive_score = self.get_gold_parse_score_right(chart, word_list, parse, 0, end)
        new_chart = {}
        for _ in chart:
            if self.balance:
                new_chart[_] = chart[_]
            else:
                new_chart[_] = chart[_].item()
        # new_chart = chart
        if self.balance:
            cky_scores, cky_decisions = self.run_cky(new_chart, word_list, end, gpu=True)
        else:
            cky_scores, cky_decisions = self.run_cky(new_chart, word_list, end, gpu=False)
        
        if self.cky_dec and not self.balance:
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
                    gpu_cky_scores[st][en] = self.get_penalty(chart, st, split_point, en, True) + gpu_cky_scores[st][split_point] + gpu_cky_scores[split_point+1][en]

        tot_terms = len(parse.keys())
        cky_parse = []
        negative_score = decode_cky_parse_list(chart, word_list, cky_decisions, 0, end, cky_parse)
        tot_correct = 0
        for key in parse.keys():
            st, en = key.split(" ")
            if (int(st), int(en)) in cky_parse:
                tot_correct += 1
        
        if self.cky_dec:
            if not self.balance:
                score = self.cky_decision_level_loss(gpu_cky_scores, chart, curr_str, parse, device)
            else:
                score = self.cky_decision_level_loss(cky_scores, chart, curr_str, parse, device)
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
        if self.margin == -1:
            # this should have another argument
            # get topk samples and treat them as negatives
            filtered_scores_dict = {}
            tot_correct = 0
            filtered_scores = []
            for key in chart:
                (s,e) = key
                # if f"{s} {e+1}" in parse:
                #     tot_correct += 1
                #     # continue
                filtered_scores_dict[chart[key].item()] = f"{s} {e+1}"
                if self.orth_bidir and (e+1,e+1) in chart:
                    filtered_scores.append(chart[key] + chart[(e+1,e+1)])
                else:
                    filtered_scores.append(chart[key])
            filtered_scores = torch.stack(filtered_scores)
            # filtered_scores = torch.stack(list(chart.values()))
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
            # sample randomly
            for _ in range(self.neg_samples):
                cand_s = randint(0, end - 1)
                cand_e = randint(cand_s+1, end)
                if f"{cand_s} {cand_e+1}" in parse:
                    continue
                if self.orth_bidir and (cand_e+1,cand_e+1) in chart: 
                    negative_score += chart[(cand_s, cand_e)] + chart[(cand_e+1, cand_e+1)]
                else:
                    negative_score += chart[(cand_s, cand_e)]
                tot_correct = -1
        tot_terms = len(parse.keys())

        return positive_score - self.neg_rel_wt*negative_score, tot_terms, tot_correct

    def get_score(self, score_chart, dataset, parses=None, depths=None, tau=1, mean=True, input_str=None, use_gold=False, print_parse=False, override=False):
        # Compute tree regularization scores
        # print_parse: when enabled, print subtrees after every split decision.
        # tau: Temparature for gumbel softmax.
        # mean: Expectation approximation using mean of scores.

        scores = []
        device = torch.device("cuda")
        tot = 0
        tot_correct = 0
        for idx, chart in enumerate(score_chart):
            end = max([key[1] for key in chart])
            curr_str = input_str[idx]
            if (print_parse):
                print(curr_str)
            if (self.single):
                if (parses is not None):
                    parse = parses[idx]
                    depth = depths[idx]
                    score, right = self.get_score_single_gold(chart, parse, curr_str, end, depth, device, mean)
                    if right == -1:
                        continue
                    else:
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
                    elif self.cky or override:
                        score, tot_terms, tot_right = self.cky_reg(chart, curr_str, parse, end, device)
                    else:
                        # print(curr_str)
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
        if self.cky or override:
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
                curr_chart = chart

                scores = []
                offset = 0
                maint_offsets = []
                for k in range(st, en):
                    cand_score = self.get_penalty(curr_chart, st, k, en)
                    if word_list[k+1][0] != 'Ġ':
                        offset += 1
                        continue
                    maint_offsets.append(offset)
                    scores.append(cand_score)
                
                scores = torch.stack(scores)
                best_idx = torch.argmax(scores)
                best = best_idx.item() + st

                p1, s1 = recurse(chart, word_list, st, best+maint_offsets[best - st], depth+1)
                p2, s2 = recurse(chart, word_list, best+1+maint_offsets[best - st], en, depth+1)

                return (p1, p2), scores[best_idx] + s1 + s2

        parses = []
        scores = []
        for idx, chart in enumerate(score_chart):
            # print(chart[0])
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

                curr_chart = chart

                scores = []
                offset = 0
                maint_offsets = []
                for k in range(st, en):
                    cand_score = self.get_penalty(chart, st, k, en)
                    if word_list[k+1][0] != 'Ġ':
                        offset += 1
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
                    cand_score = self.get_penalty(curr_chart, st, k, en, True) + cky_scores[st][k] + cky_scores[k+1][en]
                    if word_list[k+1][0] != 'Ġ':
                        offset += 1
                        continue
                    maint_offsets.append(offset)
                    scores.append(cand_score)
                
                if gpu:
                    scores = torch.stack(scores)
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

        return parses, scores, charts
