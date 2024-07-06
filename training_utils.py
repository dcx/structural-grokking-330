from sympy import use
import torch
import sequence
from tqdm import tqdm
import os
import wandb
import time
import math

from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import AdamW
import torch

from transformers.data.data_collator import DataCollatorWithPadding
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler,
)

import collate
import wandb
from plot import CustomPlot
import json
import pdb
import util
from itertools import chain


def get_grad_norm(model):
    total_norm = 0
    parameters = [
        p for p in model.parameters() if p.grad is not None and p.requires_grad
    ]
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm**0.5
    return total_norm


def get_opt(lr, weight_decay, model, regularizer=None):
    if type(model) != torch.nn.Module:
        model = model.model
    no_decay = ["bias", "LayerNorm.weight"]
    adam_epsilon = 1e-7
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    if regularizer is not None:
        optimizer_grouped_parameters.append({
            "params": [
                p
                for n, p in regularizer.named_parameters()
            ],
            "weight_decay": 0.0,
        })
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=lr,
        eps=adam_epsilon,
    )
    return optimizer


def get_scheduler(opt, start_lr, min_lr, t_total):
    # cosine scheduler from 6e-4 to 6e-5
    num_warmup_steps = (5 * t_total) // 1000 # 0.5% of total

    def get_lr(it):
        if it < num_warmup_steps:
            return it / num_warmup_steps
        
        decay_ratio = (it - num_warmup_steps) / (t_total - num_warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) 
        return min_lr/start_lr + coeff * (1 - min_lr/start_lr)

    return LambdaLR(opt, get_lr, -1)

    # scheduler = get_linear_schedule_with_warmup(
    #     opt, num_warmup_steps=num_warmup_steps, num_training_steps=t_total
    # )
    # return scheduler

def get_scaler(use_amp=False):
    return torch.cuda.amp.GradScaler(enabled=use_amp)


def eval_lm(model_interface, val_datasets, best_accs, device, num_steps, collator, eval_batch_size=32):
    def helper(validation):
        model_interface.model.eval()
        loss_curr = 0
        total = 0
        with torch.no_grad():
            for batch in tqdm(validation):
                batch_gpu = {}
                for key in batch:
                    if (key in ['string', 'parses']):
                        batch_gpu[key] = batch[key]
                    else:
                        batch_gpu[key] = batch[key].to(device)
                res = model_interface(batch_gpu, normalize=True)
                loss_curr += res.loss.cpu().numpy()
                total += 1
        return loss_curr / total

    plots = {}
    curr_accs = {}
    for key, val_dataset in val_datasets.items():
        validation = DataLoader(
            val_dataset,
            sampler=SequentialSampler(val_dataset),
            batch_size=eval_batch_size,
            collate_fn=collator,
        )
        curr_accs[key] = helper(validation)
        plots["curr-{}-loss".format(key)] = curr_accs[key]
    best_accs = {key: min(curr_accs[key], best_accs[key]) for key in curr_accs}
    plots.update({"best/{}": v for k, v in best_accs.items()})
    plotting_util(plots, num_steps)
    return best_accs, curr_accs


def plotting_util(dict_of_elems, step):
    wandbdict = {}
    for k, v in dict_of_elems.items():
        if isinstance(v, CustomPlot):
            v = v.to_wandb()
            if v is None:
                continue

            if isinstance(v, dict):
                for k2, v2 in v.items():
                    wandbdict[k + "/" + k2] = v2
            else:
                wandbdict[k] = v
        elif isinstance(v, (int, float)):
            wandbdict[k] = v
        else:
            assert False, f"Invalid data type {type(v)}"
    wandbdict["iteration"] = step
    wandb.log(wandbdict)


def eval_func(model, validation, tokenizer, best_acc, device):
    def get_decoding_acc(outputs, labels):
        acc = 0
        for out, label in zip(outputs, labels):
            dec_str = tokenizer.decode(out, skip_special_tokens=True)
            label = [(l if l != -100 else tokenizer.pad_token_id) for l in label]
            orig_str = tokenizer.decode(label, skip_special_tokens=True)
            acc += dec_str == orig_str
        return acc

    curr_acc = 0
    total = 0
    if type(model) != torch.nn.Module:
        model.model.eval()
    else:
        model.eval()
    with torch.no_grad():
        for batch in tqdm(validation):
            batch_gpu = {}
            for key in batch:
                batch_gpu[key] = batch[key].to(device)
            curr_acc += get_decoding_acc(
                model.generate(batch_gpu["input_ids"]).cpu().tolist(),
                batch["labels"].cpu().tolist(),
            )
            total += len(batch["labels"])

    curr_acc /= 1.0 * total
    print("Current Accuracy: {:.4f}".format(curr_acc))
    if curr_acc > best_acc:
        return curr_acc
    else:
        return best_acc


def eval_callback(
    args,
    model,
    val_datasets,
    tokenizer,
    best_accs,
    device,
    num_steps,
    train_data_collator,
):
    # assert model.model.mode == "lm"
    best_accs, curr_accs = eval_lm(
        model,
        val_datasets,
        best_accs,
        device,
        num_steps,
        train_data_collator,
        args.batch_size_eval,
    )
    return best_accs, curr_accs

def get_grads(model):
    parameters = [
        p for p in model.model.parameters() if p.requires_grad
    ]
    curr_grads = []
    for p in parameters:
        if (p.grad == None):
            curr_grads.append(None)
        else:
            param = p.grad.cpu().data
            curr_grads.append(param)
    return curr_grads

def train_loop(
    args,
    model,
    train_dataset,
    val_datasets,
    device,
    save_dir,
    save_model: bool,
    save_interval=1000,
    tokenizer=None,
    metric="acc",
    in_vocab=None,
    callback_fn=None,
    regularizer=None,
    batch_limit=-1
):
    num_steps = 0
    max_grad_norm = 2.0
    train_batch_size = args.batch_size
    accum_steps = args.accum_steps
    eval_every = args.eval_every
    max_steps = args.max_train_steps
    regularizer_steps = args.regularizer_steps
    change_steps = args.change_steps
    use_gold = args.use_gold
    tau_init = args.tau_init
    tau_final = args.tau_final
    print_parse = args.print_parse
    complete_tree_reg = (args.regularize and not args.reg_single and not args.use_orthogonal)
    mixture = args.mixture
    use_amp = args.use_amp
    use_packing = args.pack
    max_seq_len = args.max_seq_len
    
    if args.proj:
        opt = get_opt(args.start_lr, args.weight_decay, model, regularizer)
    else:
        opt = get_opt(args.start_lr, args.weight_decay, model, None)
    scheduler = get_scheduler(opt, args.start_lr, args.end_lr, max_steps)
    scaler = get_scaler(use_amp)

    if tokenizer is not None:
        train_data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    else:
        train_data_collator = collate.VarLengthCollate(tokenizer)

    tau_step = (tau_final - tau_init)/max_steps
    if args.regularizer_rel_wt_end != -1.0:
        wt_step = (args.regularizer_rel_wt_end - args.regularizer_rel_wt_init)/max_steps
    else:
        wt_step = 0
    
    if args.pack:
        # this dataset handles all packing
        train_dataset = util.PackedDataset(train_dataset, max_seq_len, model.encoder_sos)

    best_ppl = {key: 10000.0 for key in val_datasets}
    sci_grads = None
    loss_grads = None
    best_blimp = -1 
    best_val_ppl = 1000000
    while True:
        # print(train_dataset)

        train_dataloader = DataLoader(
            train_dataset,
            sampler=RandomSampler(train_dataset),
            batch_size=train_batch_size,
            collate_fn=train_data_collator,
        )
        if args.parse_portion != 1.0:
            parse_dataset = torch.utils.data.Subset(train_dataset, range(int(args.parse_portion * len(train_dataset))))
            if use_packing:
                # this dataset handles all packing
                parse_dataset = util.PackedDataset(parse_dataset, max_seq_len, model.encoder_sos)
            parse_dataloader = iter(DataLoader(
                parse_dataset,
                sampler=RandomSampler(parse_dataset, replacement=True, num_samples=100000000),
                batch_size=train_batch_size,
                collate_fn=train_data_collator,
            ))
        
        total_train_sz = len(train_dataset)
        if num_steps > max_steps:
            break

        with torch.enable_grad(), tqdm(total=total_train_sz) as progress_bar:
            losses = []
            accum_strings = []
            if mixture:
                sci_scores_cky_agg = []
                sci_scores_g_agg = []
            else:
                sci_scores_agg = []
            
            for curr_batch_dict in train_dataloader:
                if type(model) != torch.nn.Module:
                    model.model.train()
                else:
                    model.train()
                
                curr_batch_dict_gpu = {}
                for key in curr_batch_dict:
                    if (key in ['string', 'parses', 'idxs', 'contained_exs']):
                        curr_batch_dict_gpu[key] = curr_batch_dict[key]
                    else:
                        curr_batch_dict_gpu[key] = curr_batch_dict[key].to(device)
                t = time.time()

                # LM Loss
                with torch.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.bfloat16, enabled=use_amp):
                    loss_curr = model(curr_batch_dict_gpu, pack=use_packing).loss

                progress_bar.update(curr_batch_dict["in"].shape[1])
                if (regularizer is not None):
                    accum_strings += curr_batch_dict['string']

                # Tree regularizer!
                broken_acc = None
                if num_steps % regularizer_steps == 0 and args.regularize:
                    regularizer_rel_wt = args.regularizer_rel_wt_init + num_steps * wt_step

                    # Sample a batch from the parse dataloader
                    if args.parse_portion != 1.0:
                        parse_batch_dict = next(parse_dataloader)
                    else:
                        parse_batch_dict = curr_batch_dict

                    # This is for Gumbel softmax, rarely used
                    tau = tau_init + tau_step * num_steps

                    # Get strings to pass to the tree regularizer
                    if (regularizer is not None):
                        curr_parses = parse_batch_dict['parses']
                        curr_strings = parse_batch_dict['string']
                        if use_packing:
                            curr_parses = list(chain(*curr_parses))
                            curr_strings = list(chain(*curr_strings))
                        
                        if (use_gold and args.dataset!='ds-addmult-mod10'):
                            parses = [json.loads(_) for _ in curr_parses]
                        else:
                            parses = None
                        
                        # Get SCI chart
                        skip = False
                        if (batch_limit != -1):
                            sampled_strings = []
                            sampled_parses = []
                            for idx in range(len(curr_strings)):
                                if (complete_tree_reg and len(curr_strings[idx].split(" ")) > 30):
                                    continue
                                sampled_strings.append(curr_strings[idx])
                                if (parses is not None):
                                    sampled_parses.append(parses[idx])
                                if (len(sampled_strings) == batch_limit):
                                    break
                            if (len(sampled_parses) == 0):
                                sampled_parses = None
                            if len(sampled_strings) == 0:
                                # The current batch has no appropriate strings
                                skip = True
                            else:
                                sci_charts, samples, sampled_parses, depths = regularizer.build_scores(sampled_strings, model, sampled_parses, batch=True, use_gold=use_gold)
                        else:
                            sci_charts, samples, sampled_parses, depths = regularizer.build_scores(curr_strings, model, parses, batch=True, use_gold=use_gold)
                        
                        # get SCI scores
                        t = time.time()
                        if not skip:
                            with torch.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.bfloat16, enabled=use_amp):
                                if (use_gold and args.enforce_gold):
                                    if mixture:
                                        sci_scores_cky, broken_acc_cky = regularizer.get_score(sci_charts, args.dataset, sampled_parses, depths, mean=args.mean_regularize, input_str=samples, use_gold=use_gold, tau=tau, print_parse=print_parse, override=True)
                                        sci_scores_g, broken_acc_g = regularizer.get_score(sci_charts, args.dataset, sampled_parses, depths, mean=args.mean_regularize, input_str=samples, use_gold=use_gold, tau=tau, print_parse=print_parse)
                                    else:
                                        sci_scores, broken_acc = regularizer.get_score(sci_charts, args.dataset, sampled_parses, depths, mean=args.mean_regularize, input_str=samples, use_gold=use_gold, tau=tau, print_parse=print_parse)
                                else:
                                    sci_scores, broken_acc = regularizer.get_score(sci_charts, args.dataset, parses=None, depths=None, mean=args.mean_regularize, input_str=samples, use_gold=use_gold, tau=tau, print_parse=print_parse)
                            
                            # turn the scores into a loss
                            if mixture:
                                fin_sci_score_cky = torch.mean(torch.stack(sci_scores_cky))
                                sci_loss_cky = -regularizer_rel_wt * fin_sci_score_cky
                                sci_loss_cky /= accum_steps
                                sci_scores_cky_agg.append(fin_sci_score_cky.item() / accum_steps)
                                fin_sci_score_g = torch.mean(torch.stack(sci_scores_g))
                                sci_loss_g = -regularizer_rel_wt * fin_sci_score_g
                                sci_loss_g /= accum_steps
                                sci_scores_g_agg.append(fin_sci_score_g.item() / accum_steps)
                                # t = time.time()
                                sci_loss = sci_loss_cky + sci_loss_g
                                sci_loss.backward()
                                # print(time.time() - t)
                            else: 
                                fin_sci_score = torch.mean(torch.stack(sci_scores))
                                sci_loss = -regularizer_rel_wt * fin_sci_score
                                # t = time.time()
                                sci_loss /= accum_steps
                                sci_scores_agg.append(fin_sci_score.item() / accum_steps)
                                sci_loss.backward()
                                # print(time.time() - t)
                            # print(sci_loss)
                
                # LM loss backward
                loss_curr /= accum_steps
                losses.append(loss_curr.item())
                loss_curr.backward()

                # Update optim
                if len(losses) == accum_steps:
                    torch.nn.utils.clip_grad_norm_(
                        model.model.parameters(), max_grad_norm
                    )
                    progress_bar.set_postfix(
                        {"loss": sum(losses), "num_steps": num_steps}
                    )
                    # print(sum(losses))
                    grad_norm = get_grad_norm(model.model)
                    if (regularizer is not None and num_steps % regularizer_steps == 0):
                        if mixture:
                            if broken_acc_cky is not None:
                                wandb.log(
                                    {
                                        "bracket_acc_train_cky": broken_acc_cky
                                    }
                                )
                            if broken_acc_g is not None:
                                wandb.log(
                                    {
                                        "bracket_acc_train_greedy": broken_acc_g
                                    }
                                )
                            wandb.log(
                                {
                                    "sci_score_cky": sum(sci_scores_cky_agg),
                                    "sci_score_greedy": sum(sci_scores_g_agg)
                                }
                            )
                        else:
                            if broken_acc is not None:
                                wandb.log(
                                    {
                                        "bracket_acc_train": broken_acc
                                    }
                                )
                            wandb.log(
                                {
                                    "sci_score": sum(sci_scores_agg)
                                }
                            )
                        if mixture:
                            wandb.log(
                                {
                                    "loss": sum(losses),
                                    "grad_norm": grad_norm,
                                    "iteration": num_steps
                                }
                            )
                        else:
                            wandb.log(
                                {
                                    "loss": sum(losses),
                                    "grad_norm": grad_norm,
                                    "iteration": num_steps
                                }
                            )
                    else:
                        wandb.log(
                            {
                                "loss": sum(losses),
                                "grad_norm": grad_norm,
                                "iteration": num_steps
                            }
                        )
                    # scaler.step(opt)
                    opt.step()
                    scheduler.step()
                    # scaler.update()
                    # model.model.zero_grad()
                    model.model.zero_grad(set_to_none=True)
                    losses = []
                    if mixture:
                        sci_scores_cky_agg = []
                        sci_scores_g_agg = []
                    else:
                        sci_scores_agg = []
                    num_steps += 1

                    # Save model if save_dir and save_interval has been hit
                    if (save_model and save_interval == 0):
                        # Always save to same file [AN, I find this more convenient for my runs]
                        if num_steps % 100000 == 0:
                            save_path = f"{os.path.join(save_dir, 'state')}.pt"
                            torch.save(model.model.state_dict(), save_path)
                            print(f"Saved model at step {num_steps} to {save_path}")
                    elif num_steps % save_interval == 0 and save_model:
                        save_path = f"{os.path.join(save_dir, 'state')}_{num_steps}.pt"
                        torch.save(model.model.state_dict(), save_path)
                        print(f"Saved model at step {num_steps} to {save_path}")

                    # print(num_steps)
                    if num_steps % eval_every == 0:
                        print("Evaluating at step {}".format(num_steps))
                        if args.lm:
                            best_ppl, curr_ppl = eval_callback(
                                args,
                                model,
                                val_datasets,
                                tokenizer,
                                best_ppl,
                                device,
                                num_steps,
                                train_data_collator,
                            )

                        if callback_fn is not None:
                            print(args.dataset)
                            if args.dataset in ["bllip-lg", "bllip-md", "bllip-int", "ptb"]:
                                if mixture:
                                    train_score_cky = callback_fn("train", regularizer, True, args)
                                    val_score_cky = callback_fn("val", regularizer, True, args)
                                    test_score_cky = callback_fn("test", regularizer, True, args)
                                    train_score_g = callback_fn("train", regularizer, False, args)
                                    val_score_g = callback_fn("val", regularizer, False, args)
                                    test_score_g = callback_fn("test", regularizer, False, args)
                                    if (val_score_cky['ppl'] < best_val_ppl):
                                        best_val_ppl = val_score_cky['ppl']
                                        best_blimp = val_score_cky['blimp_acc']
                                else:
                                    train_score = callback_fn("train", regularizer, False, args)
                                    val_score = callback_fn("val", regularizer, False, args)
                                    test_score = callback_fn("test", regularizer, False, args)
                                    if (val_score['ppl'] < best_val_ppl):
                                        best_val_ppl = val_score['ppl']
                                        best_blimp = val_score['blimp_acc']
                            else:
                                val_score = callback_fn("val")
                                test_score = callback_fn("test")
                                print(f"{val_score = }", f"{test_score = }")
                            if args.dataset in ["bllip-lg", "bllip-md", "bllip-int", "ptb"]:
                                if (regularizer is not None):
                                    if mixture:
                                        wandbdict = {
                                            "iteration": num_steps,
                                            "train_parseval_cky": train_score_cky['parsing_acc'],
                                            "train_parseval_greedy": train_score_g['parsing_acc'],
                                            "val_ppl": val_score_cky['ppl'],
                                            "test_ppl": test_score_cky['ppl'],
                                            "blimp_score": val_score_cky['blimp_acc'],
                                            "best_blimp_score": best_blimp,
                                            "val_parseval_cky": val_score_cky['parsing_acc'],
                                            "test_parseval_cky": test_score_cky['parsing_acc'],
                                            "val_parseval_greedy": val_score_g['parsing_acc'],
                                            "test_parseval_greedy": test_score_g['parsing_acc']
                                        }
                                    else:
                                        wandbdict = {
                                            "iteration": num_steps,
                                            "train_parseval": train_score['parsing_acc'],
                                            "val_ppl": val_score['ppl'],
                                            "test_ppl": test_score['ppl'],
                                            "blimp_score": val_score['blimp_acc'],
                                            "best_blimp_score": best_blimp,
                                            "val_parseval": val_score['parsing_acc'],
                                            "test_parseval": test_score['parsing_acc'],
                                        }
                                else:
                                    wandbdict = {
                                        "iteration": num_steps,
                                        "val_ppl": val_score['ppl'],
                                        "test_ppl": test_score['ppl'],
                                        "blimp_score": val_score['blimp_acc'],
                                        "best_blimp_score": best_blimp
                                    }
                            else:
                                wandbdict = {
                                    "iteration": num_steps,
                                    "val_aux": val_score,
                                    "test_aux": test_score,
                                }
                            wandb.log(wandbdict)

                    if num_steps > max_steps:
                        break
            if losses:
                # num_steps += 1
                progress_bar.set_postfix({"loss": sum(losses), "num_steps": num_steps})
                grad_norm = get_grad_norm(model.model)
                wandb.log(
                    {
                        "loss": sum(losses),
                        "grad_norm": grad_norm,
                        "iteration": num_steps,
                    }
                )
                # torch.nn.utils.clip_grad_norm_(model.model.parameters(), max_grad_norm)
                # opt.step()
                # scheduler.step()
                model.model.zero_grad()
                losses = []
                if num_steps > max_steps:
                    break

    if save_model:
        save_path = f"{os.path.join(save_dir, 'state')}_final_model.pt"
        torch.save(model.model.state_dict(), save_path)
        print(f"Saved final model to {save_path}")
    
    print("BLIMP Score,", best_blimp)
    print("Best Perplexities,", best_ppl)
    return

def reg_loop(
    args,
    model,
    train_dataset,
    device,
    tokenizer=None,
    regularizer=None
):
    '''
    Quick hack to get tree regularization scores for a saved model on any dataset. 
    Prints tree regularization scores for each batch, and the average score over all batches. 

    Args:
        args: Command line arguments.
        model: The trained model.
        train_dataset: Dataset for which tree regularization scores are required.
        device: Device on which score computation will take place.
        tokenizer: Input vocabulary.
        regularizer: Initialized tree regularizer object.

    Returns:
        None
    '''
    num_steps = 0
    train_batch_size = args.batch_size
    use_gold = args.use_gold
    max_steps = args.max_train_steps

    if tokenizer is not None:
        train_data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    else:
        train_data_collator = collate.VarLengthCollate(tokenizer)

    all_sci_scores = []

    while True:
        # print(train_dataset)
        train_dataloader = DataLoader(
            train_dataset,
            sampler=RandomSampler(train_dataset),
            batch_size=train_batch_size,
            collate_fn=train_data_collator,
        )
        total_train_sz = len(train_dataset)
        if num_steps > max_steps:
            break
        with torch.enable_grad(), tqdm(total=total_train_sz) as progress_bar:
            for curr_batch_dict in train_dataloader:
                num_steps += 1
                if num_steps > max_steps:
                    break
                if type(model) != torch.nn.Module:
                    model.model.train()
                else:
                    model.train()
                curr_batch_dict_gpu = {}
                for key in curr_batch_dict:
                    if (key == 'string'):
                        curr_batch_dict_gpu[key] = curr_batch_dict[key]
                    else:
                        curr_batch_dict_gpu[key] = curr_batch_dict[key].to(device)
                progress_bar.update(curr_batch_dict["in"].shape[1])

                # Tree regularizer!
                if (regularizer is not None):
                    sci_charts, samples = regularizer.build_scores(curr_batch_dict['string'], model, 0, tqdm_disable=True, parse_splits=None, batch=True, use_gold=use_gold)
                    if (args.mean_regularize):
                        sci_scores = regularizer.get_score(sci_charts, mean=True, input_str=samples, use_gold=use_gold, print_parse=False)
                    else:
                        sci_scores = regularizer.get_score(sci_charts, mean=False, input_str=samples, use_gold=use_gold, print_parse=False)
                    # print(sci_scores)
                    fin_sci_score = torch.mean(torch.stack(sci_scores))
                    all_sci_scores.append(fin_sci_score.item())
                    print(fin_sci_score)
    print(sum(all_sci_scores)/len(all_sci_scores))

        
