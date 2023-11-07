### regularizes a model to encourage structural generalization
from asyncio import as_completed
import torch
from tree_projections_utils import get_all_hidden_states_scratch, get_masking_info 
from tqdm import tqdm
from torch.optim import Adam
import collate
import torch.nn.functional as F
from data_utils.dyck_helpers import build_datasets_dyck, eval_callback_dyck
from transformer_helpers import create_lm

class Chart():
    # Populate SCI chart
    def __init__(self, sim_metric, tokenizer):
        self.sim_metric = sim_metric
        self.tokenizer = tokenizer
        self.train_data_collator = collate.VarLengthCollate(None)
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

    def build_scores(self, input_strs, model, start_relax_layer, tqdm_disable=True, parse_splits=None):
        scores = [{} for _ in input_strs]
        outer_context_vecs = get_all_hidden_states_scratch(model, 
                                                           self.tokenizer, 
                                                           input_strs, 
                                                           tqdm_disable=True, 
                                                           is_lm = True,
                                                           layer_id = -1)

        all_masked_strs = []
        all_input_masks = []
        all_slice_dicts = {}
        str_idx = {}

        for idx, input_str in enumerate(input_strs):
            if parse_splits is not None:
                slice_dict, masked_strs, input_masks = self.cache(input_str, parse_splits[idx])
            else:
                slice_dict, masked_strs, input_masks = self.cache(input_str)
            clen = len(all_input_masks)
            for offset in range(len(masked_strs)):
                str_idx[clen + offset] = idx
                all_slice_dicts[clen + offset] = slice_dict[offset] 
                all_masked_strs.append(masked_strs[offset])
                all_input_masks.append(input_masks[offset])

        batch_size = 1024 
        st = 0

        device = torch.device('cuda')
    
        num_layers = model.get_encoder_layers()
        with tqdm(total=len(all_masked_strs), disable=tqdm_disable) as progress_bar:
            while st < len(all_masked_strs):
                en = min(len(all_masked_strs),st+batch_size)
                cslice = all_masked_strs[st: en]
                inputs, input_lens = self.tokenizer_helper(cslice)
                inputs = inputs.to(device)
                input_lens = input_lens.to(device)
                inp_len = inputs.shape[1]
                # input masks specify the inner context
                if all_input_masks is not None:
                    masks_curr = all_input_masks[st: en]
                    masks_padded = []
                    for mask in masks_curr:
                        mask_padded = mask + [1]*(inp_len - len(mask))
                        masks_padded.append(mask_padded)
                    tree_mask = torch.tensor(masks_padded).bool().to(device)
                    relax_mask = model.generate_len_mask(inp_len, input_lens).to(device)
                    mask = self.relax_cond(tree_mask, relax_mask, start_relax_layer, num_layers)
                    mask_mult = tree_mask.unsqueeze(-1)
                else:
                    mask = model.generate_len_mask(inp_len, input_lens).to(device)
                    mask_mult = mask.unsqueeze(-1)
                

                outputs = model.encoder_only(inputs, mask) * (~mask_mult)
                for idx, _ in enumerate(cslice):
                    inner_vec = outputs[idx][1:-1].sum(axis=0)
                    oidx = str_idx[idx + st]
                    i, j = all_slice_dicts[idx + st]
                    # i+1 because the first vector is [SOS] and j+2 because we want everything from ith token to jth token.
                    outer_vec = outer_context_vecs[oidx][0][i+1: j+2].sum(axis=0)
                    scores[oidx][(i, j)] = self.sim_metric(outer_vec, inner_vec)
                progress_bar.update(en - st)
                st = en

        print(scores)
        return scores 
    

class Regularizer():
    def __init__(self, sim_metric, input_strs=None, parse_splits=None, as_hinge=False, tokenizer=None, start_relax_layer=0):
        self.sim_metric = sim_metric
        self.as_hinge = as_hinge
        self.input_strs = input_strs
        self.parse_splits = parse_splits
        self.tokenizer = tokenizer
        self.start_relax_layer = start_relax_layer
        self.chart = Chart(sim_metric, tokenizer)

    def preprocess(self, input_strs, tokenizer):
        self.cache = []
        for inp_str in tqdm(input_strs):
            sentence2idx_tuple, masked_strs, input_masks = get_masking_info(tokenizer, [inp_str], pretrained=False)
            self.cache.append((sentence2idx_tuple, masked_strs, input_masks))
        print("Done building cache")        

    def run_on_indices(self, idxs, model):
        input_strs = [self.input_strs[idx] for idx in idxs]
        parse_splits = [self.parse_splits[idx] for idx in idxs]

        if self.as_hinge:
            chart_scores_all = self.chart.build_scores(input_strs, model, start_relax_layer=self.start_relax_layer, tqdm_disable=True)
        else:
            ### only have to get inner vecs for (i, k), (k+1, j) for all (i, val, j) in parse_splits...
            chart_scores_all = self.chart.build_scores(input_strs, model, start_relax_layer=self.start_relax_layer, tqdm_disable=True, parse_splits=parse_splits)

        loss_curr = 0.0
        for idx, parse_split in enumerate(parse_splits):
            #if len(input_str.split(' ')) > 40:
            #    continue
            chart_scores = chart_scores_all[idx] 
            try:
                loss_curr += self.get_losses(chart_scores, parse_split)
            except:
                import pdb; pdb.set_trace();
        return loss_curr / len(idxs)

        #for input_str, parse_split in zip(input_strs, parse_splits):
            #if len(input_str.split(' ')) > 40:
            #    continue
        #    chart_scores = self.chart.build_scores(input_str, model, start_relax_layer=0, tqdm_disable=True)
        #    loss_curr += self.get_losses(chart_scores, parse_split)
        #return loss_curr / len(idxs)

        # for input_str, parse_split, orig_idx in zip(input_strs, parse_splits, idxs):
        #     sentence1idx_tuple, masked_strs, input_masks = self.cache[orig_idx]
        #     print(len(masked_strs))
        #     outer_context_vecs = get_all_hidden_states_scratch(model, self.tokenizer, [input_str], tqdm_disable=True, regularize=True)
        #     inner_context_vecs = get_all_hidden_states_scratch(model, self.tokenizer, masked_strs, input_masks, sum_all=True,  
        #                                             tqdm_disable=True, regularize=True)
            

        #     key1idx = {key: idx for idx, key in sentence2idx_tuple[0]}    
        #     loss_curr = reg_weight * (self.__call__(key1idx, [v[0] for v in inner_context_vecs], outer_context_vecs[0][0], parse_split))
        #     loss_curr /= len(idxs)
        #     loss_curr.backward()
        # return -1

    def get_hinge(self, i, j, gold_split, inner_context_vecs, contextual_vec, keys):
        best_score = 0.0
        gold_score = self.get_score(i, j, gold_split, inner_context_vecs, contextual_vec, keys)
        for k in range(i, j):
            if k == gold_split:
                continue
            else:
                curr_score = self.get_score(i, j, k, inner_context_vecs, contextual_vec, keys)
                if curr_score > best_score:
                    best_score = curr_score
        return self._hinge_loss(best_score, gold_score)

    def _hinge_loss(self, ours, gold):
        loss = 0.1 + ours - gold
        return (loss > 0.1) * loss

    def get_score(self, i, j, k, inner_context_vecs, contextual_vec, keys):        
        cont_vec_1 = contextual_vec[i: k+1].sum(dim=0)
        inner_vec_1 = inner_context_vecs[keys[(i, k)]]

        cont_vec_2 = contextual_vec[k+1:j+1].sum(dim=0)
        inner_vec_2 = inner_context_vecs[keys[(k+1, j)]]
        return (self.sim_metric(cont_vec_1, inner_vec_1) + self.sim_metric(cont_vec_2, inner_vec_2))


    def get_hinge_2(self, i, j, k, chart_scores):
        gold_score = chart_scores[(i,k)] + chart_scores[(k+1,j)]
        return self._hinge_loss(max(chart_scores[(i,k1)] + chart_scores[(k1+1, j)] for k1 in range(i,j)), gold_score)

    def get_losses(self, chart_scores, parse_splits):
        total_loss = 0.0
        for key in parse_splits:
            i, j = key
            k = parse_splits[key]
            if self.as_hinge:
                total_loss += self.get_hinge_2(i, j, k, chart_scores)
            else:
                total_loss += -1.0*(chart_scores[(i, k)] + chart_scores[(k+1, j)])
        if len(parse_splits) > 0:
            return total_loss / len(parse_splits) # this is a loss
        else:
            return 0

    def __call__(self, keys, inner_context_vecs, contextual_vec, parse_splits):
        '''
            takes as input a set of inner context vectors, and a contextual vector
            two versions
            1. maximise sci score for gold tree
            2. hinge loss between sci score for gold tree and sci score for induced tree
        '''

        total_loss = 0.0
        for key in parse_splits:
            i, j = key
            k = parse_splits[key]
            if self.as_hinge:
                try:
                    total_loss += self.get_hinge(i, j, k, inner_context_vecs, contextual_vec, keys)
                except:
                    import pdb; pdb.set_trace()
            else:
                total_loss += (1.0 - self.get_score(i, j, k, inner_context_vecs, contextual_vec, keys))
        return total_loss # this is a loss
            ### make it so that (i, j) splits at k i.e contextual_vec[i, k] ~ inner_vec[i, k]
            ### and contextual_vec[k+1, j] ~ 

if __name__ == '__main__':
    datasets, in_vocab, _ = build_datasets_dyck(vocab=20)
    chart = Chart(lambda x1, x2: F.cosine_similarity(x1, x2, dim=0), in_vocab)
    model = create_lm(len(in_vocab), 512, 4, 4)
    sci = chart.build_scores(["(f (g g) (b b) f) (b b)"], model, 1, tqdm_disable=False, parse_splits=None)
