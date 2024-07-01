# utils for running pushdown layers in various settings
import torch
import collate
import torch.nn.functional as F
from tqdm import tqdm
import pdb

def add_eos(input: torch.Tensor, lengths: torch.Tensor, eos_id: int):
    input = torch.cat((input, torch.zeros_like(input[0:1])), dim=0)
    input.scatter_(0, lengths.unsqueeze(0).long(), value=eos_id)
    return input


def compute_per_token_logprob(eos_token, str_logits, inputs, input_lens):
    str_logprobs = []
    # (bs x len x vocab)
    str_logits = str_logits.transpose(0, 1)
    eos_token = torch.tensor([eos_token]).to(inputs.device)
    for idx, (c_input, str_logprob) in enumerate(zip(inputs, str_logits)):
        curr_len = input_lens[idx]
        ## len x vocab
        ### shift input by 1 to evaluate LM
        target = torch.cat([c_input[1:curr_len], eos_token])
        eos_removed_logits = str_logprob[:curr_len]
        eos_logprobs = F.log_softmax(eos_removed_logits, dim=1)
        logprobs_curr = torch.gather(eos_logprobs, 1, target.unsqueeze(1)).squeeze(1)
        str_logprobs.append(logprobs_curr.cpu().numpy())
    return str_logprobs


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


@torch.no_grad()
def make_preds_base_model(
    lm, tokenizer, sents, gpu_id=0, get_final_answer=False, get_attn_matrices=False, no_tqdm=False, hf=False
):
    """
    Use language model to make predictions on the given sentences.
    But cannot parse.
    Output:
        - per sentence logprobs
    """

    data_collator = collate.VarLengthCollate(None)
    batch_size = 64
    st = 0
    device = torch.device("cuda:{}".format(gpu_id))
    all_sent_logprobs = []
    all_answers = []
    all_attn_matrices = []

    def tokenizer_add(s):
        if hf:
            return tokenizer.encode(s)
        else:
            return [lm.encoder_sos] + tokenizer(s)
    
    def generate_mask(max_len, in_lens):
        return torch.arange(max_len).expand(len(in_lens), max_len).to(in_lens.device) < in_lens.unsqueeze(1)

    with tqdm(total=len(sents), disable = no_tqdm) as progress_bar:
        while st < len(sents):
            en = min(len(sents), st + batch_size)
            sent_slice = sents[st:en]
            inputs, input_lens = tokenizer_helper(
                tokenizer_add, data_collator, sent_slice
            )
            inputs = inputs.to(device)
            input_lens = input_lens.to(device)
            if get_attn_matrices:
                outputs = lm.get_attention_matrices(inputs, input_lens)
                all_attn_matrices.append(outputs)
            else:
                if hf:
                    attn_mask = generate_mask(inputs.shape[1], input_lens).to(inputs.device)
                    outputs = lm(inputs, attention_mask=attn_mask)
                    all_str_logits_curr = outputs.logits
                else:
                    outputs = lm(inputs, input_lens)
                    all_str_logits_curr = outputs.data
                if get_final_answer:
                    ## bs x max_len x vocab
                    preds_curr = [
                        logit[l - 1].argmax().item()
                        for logit, l in zip(all_str_logits_curr, input_lens)
                    ]
                    all_answers += preds_curr
                else:
                    logprobs_curr = compute_per_token_logprob(
                        tokenizer(tokenizer.eos_token)["input_ids"][1] if hf else lm.encoder_eos, 
                        all_str_logits_curr.transpose(0, 1), inputs, input_lens
                    )
                    all_sent_logprobs += logprobs_curr
            progress_bar.update(en - st)
            st = en
    if get_final_answer:
        return all_answers
    elif get_attn_matrices:
        return all_attn_matrices
    else:
        return all_sent_logprobs
