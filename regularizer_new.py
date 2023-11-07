### regularizes a model to encourage structural generalization
from asyncio import as_completed
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
    # Populate SCI chart
    def __init__(self, sim_metric, tokenizer, spaces):
        self.sim_metric = sim_metric

        self.tokenizer = tokenizer
        self.train_data_collator = collate.VarLengthCollate(None)
        self.spaces = spaces # False for DSMULT!
        self._cache = {}

    def relax_cond(self, mask, relax_mask, start_relax_layer, num_layers):
        ### relax mask only masks padded stuff
        ### mas masked everything
        #### relax mask from 0 ... start_relax_layer-1, 
        #### relax_mask from num_layers - end_relax_layer to num_layers - 1
        #### mask from start_relax_layer to num_layers - end_layers - 1  
        return [relax_mask]*start_relax_layer + [mask]*(num_layers - start_relax_layer)

    def tokenizer_helper(self, inp_slice):
        inp_list = [self.tokenizer(s) for s in inp_slice]
        in_lens = [len(s) for s in inp_list]
        
        inp_to_collate = [{'in': x, 'in_len': y} for x, y in zip(inp_list, in_lens)]
        inp = self.train_data_collator(inp_to_collate)
        in_len = inp["in_len"].long()
        return inp['in'].transpose(0, 1), in_len

    def cache(self, input_str, parse_split=None):
        if input_str not in self._cache:
            sentence2idx_tuple, masked_strs, input_masks = get_masking_info(self.tokenizer, [input_str])
            slice_dict = {idx: key for idx, key in sentence2idx_tuple[0]}
            self._cache[input_str] = (slice_dict, masked_strs, input_masks)
        
        return self._cache[input_str]

    def build_scores_single(self, input_str, model, start_relax_layer, tqdm_disable=True, parse_splits=None):
        scores = {}
        device = torch.device("cuda")
        model.to(device)

        def tokenizer_helper(s, add_special_tokens=True):
            if add_special_tokens:
                return [model.encoder_sos] + self.tokenizer(s)
            else:
                return self.tokenizer(s)

        sent_tokens, idxs = get_pre_tokenized_info(input_str, tokenizer_helper)
        outer_context_vecs = get_all_hidden_states_scratch(model,
                                                            tokenizer_helper,
                                                            [input_str],
                                                            tqdm_disable=True,
                                                            pre_tokenized=([sent_tokens], [idxs]),
                                                            layer_id=-1,
                                                            is_lm=True,
                                                            compute_grad=True
                                                        )


        slice_dict, masked_strs, input_masks = get_masking_info(tokenizer_helper, [input_str], spaces=self.spaces)

        inner_context_vecs = get_all_hidden_states_scratch(
                        model,
                        tokenizer_helper,
                        masked_strs,
                        input_masks,
                        sum_all=True,
                        start_relax_layer=start_relax_layer,
                        tqdm_disable=True,
                        pre_tokenized=([sent_tokens] * len(masked_strs), [idxs] * len(masked_strs)),
                        layer_id=-1,
                        is_lm=True,
                        compute_grad=True
                    )

        all_vector_idxs = outer_context_vecs[0][-1]
        for idx, key in slice_dict[0]:
            st, en = key

            fully_contextual_vectors = all_vector_idxs[st : en + 1].sum(
                axis=0
            )
            inner_context_vectors = inner_context_vecs[idx][-1]
            scores[key] = self.sim_metric(
                inner_context_vectors, fully_contextual_vectors
            )

        # print(scores)
        return scores

    def build_scores_batch(self, input_strs, model, start_relax_layer, tqdm_disable=True, parse_splits=None):
        scores = [{} for _ in input_strs]
        device = torch.device("cuda")
        # model.to(device)

        if (type(model) != torch.nn.Module):
            curr_model = model.model
        else:
            curr_model = model

        def tokenizer_helper(s, add_special_tokens=True):
            if add_special_tokens:
                return [curr_model.encoder_sos] + self.tokenizer(s)
            else:
                return self.tokenizer(s)

        batch_sent_tokens, batch_idxs = [], []
        for input_str in input_strs:
            sent_tokens, idxs = get_pre_tokenized_info(input_str, tokenizer_helper)
            batch_sent_tokens.append(sent_tokens)
            batch_idxs.append(idxs)

        outer_context_vecs = get_all_hidden_states_scratch(curr_model,
                                                            tokenizer_helper,
                                                            input_strs,
                                                            tqdm_disable=True,
                                                            pre_tokenized=(batch_sent_tokens, batch_idxs),
                                                            layer_id=-1,
                                                            is_lm=True,
                                                            compute_grad=True
                                                        )


        slice_dict, masked_strs, input_masks = get_masking_info(tokenizer_helper, input_strs, spaces=self.spaces)
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
                        compute_grad=True
                    )

        for str_idx in range(str_num):
            all_vector_idxs = outer_context_vecs[str_idx][-1]
            for idx, key in slice_dict[str_idx]:
                st, en = key

                fully_contextual_vectors = all_vector_idxs[st : en + 1].sum(
                    axis=0
                )
                inner_context_vectors = inner_context_vecs[idx][-1]
                scores[str_idx][key] = self.sim_metric(
                    inner_context_vectors, fully_contextual_vectors
                )

        return scores
    
    def get_score(self, score_chart):

        def recurse(score_chart, st, en):
            if (st == en):
                return 0
            else:
                best = -1
                best_score = -4
                for k in range(st, en):
                    cand_score = score_chart[(st,k)] + score_chart[(k+1,en)]
                    if (cand_score > best_score):
                        best = k
                        best_score = cand_score
                s1 = recurse(score_chart, st, best)
                s2 = recurse(score_chart, best+1, en)

                # try mean instead? closer to expectation
                k_rand = randint(st, en-1)
                rand_score = best_score - score_chart[(st,k_rand)] - score_chart[(k_rand+1,en)]
        
                return s1 + s2 + rand_score

        scores = []
        device = torch.device("cuda")
        for chart in score_chart:
            end = max([key[1] for key in chart])
            score = recurse(chart, 0, end)
            if (score == 0):
                score = torch.tensor(0).to(device)
            scores.append(score.float())

        return scores
    
    def get_score_mean(self, score_chart):

        def recurse(score_chart, st, en):
            if (st == en):
                return 0
            else:
                best = -1
                best_score = -4
                tot_score = 0
                for k in range(st, en):
                    cand_score = score_chart[(st,k)] + score_chart[(k+1,en)]
                    tot_score += cand_score
                    if (cand_score > best_score):
                        best = k
                        best_score = cand_score
                s1 = recurse(score_chart, st, best)
                s2 = recurse(score_chart, best+1, en)
                # try mean instead? closer to expectation
                mean_score = best_score - (tot_score/(en-st))
        
                return s1 + s2 + mean_score

        scores = []
        device = torch.device("cuda")
        for chart in score_chart:
            end = max([key[1] for key in chart])
            score = recurse(chart, 0, end)
            if (score == 0):
                score = torch.tensor(0, requires_grad = True, dtype = torch.float).to(device)
            print(score)
            scores.append(score.float())

        return scores

if __name__ == '__main__':
    dataset, in_vocab, _ = build_datasets_lm()
    # print(dataset['test'][0])
    chart = Chart(lambda x1, x2: F.cosine_similarity(x1, x2, dim=0), in_vocab)
    # model_name = ''
    model = create_lm(len(in_vocab), 512, 4, 4)
    model.load_state_dict(torch.load('/nlp/scr/ananjan/layer4/checkpoint_990000.pickle', map_location=torch.device("cuda")))
    model.train()
    opt = Adam(model.parameters())
    # for _ in tqdm(range(1000)):
    #     sci = chart.build_scores_single("peacocks who do comfort her xylophone don't remember our raven", model, -1, tqdm_disable=True, parse_splits=None)
    #     scores = chart.get_score([sci])
    #     print(scores)
    sci = chart.build_scores_batch(["peacocks who do comfort her xylophone don't remember our raven" for _ in range(8)], model, -1, tqdm_disable=False, parse_splits=None)
    scores = chart.get_score(sci)
    print(scores)
