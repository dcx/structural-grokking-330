import os
import random
import numpy as np
import torch
import torch.nn.functional as F

import argparse
import wandb
import pickle
from vocabulary import CharVocabulary
from data_utils import (
    build_datasets_lm,
    build_datasets_tense_inflection,
    build_dataset_addmult_mod10
)
from data_utils.dyck_helpers import build_datasets_dyck, eval_callback_dyck
from data_utils.lm_dataset_helpers import eval_lm_callback
from data_utils.tense_inflection_helpers import eval_callback_tense_inflection
from data_utils.ds_addmult_mod10_helpers import (
    eval_callback_mod10,
    eval_callback_mod10_lm,
    build_dataset_addmult_hf
)
from data_utils.text_helpers import build_datasets_pushdown, build_datasets_ptb
from eval_utils.eval_pushdown_model import callback_pushdown, callback_ptb

from transformer_helpers import create_model, create_lm, create_model_interface
from transformers import AutoModelForCausalLM, AutoTokenizer
from training_utils import train_loop, reg_loop
from regularizer.regularizer_main import Chart

WANDB_USERS = {
    "kyle": {"project": "research-cs330", "entity": "mcgrathk"},
    "derek": {"project": "330", "entity": "dcx"},
    "ananjan": {"project": "structural-grokking", "entity": "tgk-ananjan"}
}

torch.set_printoptions(threshold=10000)

def working_dir() -> str:
    """
    Returns the working directory based on the USER environment variable.

    Returns:
        str: The path to the working directory.
    """
    USER = os.environ["USER"]
    dir_name = f"/scr/biggest"

    def helper(dir_name: str) -> str:
        if os.path.exists(dir_name):
            sub_dir = os.path.join(dir_name, USER, "compositionality")
            os.makedirs(sub_dir, exist_ok=True)
            return sub_dir
        else:
            return ""

    try:
        return helper(dir_name)
    except:
        return helper(f"/scr/smurty/biggest")


def get_base_transformer_model(args, in_vocab: CharVocabulary, out_vocab: CharVocabulary, model_load_path: str = None):
    """
    Returns a base transformer model and its interface.

    Args:
        args: Command line arguments.
        in_vocab (CharVocabulary): Input vocabulary.
        out_vocab (CharVocabulary): Output vocabulary.
        model_load_path (str, optional): Path to the pre-trained model. Defaults to None.

    Returns:
        tuple: A tuple containing the model and its interface.
    """
    model = create_model(
        len(in_vocab),
        len(out_vocab),
        args.vec_dim,
        args.n_heads,
        args.encoder_n_layers,
        args.decoder_n_layers,
        mode=args.mode,
        relative=args.relative,
        is_null_encoder=args.enc,
        activation = F.leaky_relu
    )

    # initial_parameters = {name: param.clone() for name, param in model.named_parameters()}

    if model_load_path:
        print(f"INFO: Loading pretrained model from {model_load_path}")
        model.load_state_dict(torch.load(
            model_load_path, map_location=torch.device("cpu")))

    # validate model loaded correctly
    # for name, param in model.named_parameters():
    #     assert torch.equal(initial_parameters[name], param) == False, f"Parameter {name} was not overwritten"

    interface = create_model_interface(model)
    return model, interface


def get_base_transformer_lm(args, in_vocab: CharVocabulary, model_load_path: str = None):
    """
    Returns a base transformer language model and its interface.

    Args:
        args: Command line arguments.
        in_vocab (CharVocabulary): Input vocabulary.
        model_load_path (str, optional): Path to the pre-trained model. Defaults to None.

    Returns:
        tuple: A tuple containing the model and its interface.
    """
    model = create_lm(len(in_vocab), args.vec_dim,
                      args.n_heads, args.encoder_n_layers, args.embedding_dropout, args.output_dropout, args.relative, args.rotary, 
                      args.causal_only, activation=F.leaky_relu)
    if model_load_path:
        print(f"INFO: Loading pretrained model from {model_load_path}")
        model.load_state_dict(torch.load(
            model_load_path, map_location=torch.device("cpu")))
    interface = create_model_interface(model, is_lm=True, has_token_labels=args.lm_with_token_labels)
    return model, interface

def get_base_transformer_hf(args, in_vocab: CharVocabulary, model_load_path: str = None):
    """
    Returns a base transformer language model and its interface.

    Args:
        args: Command line arguments.
        in_vocab (CharVocabulary): Input vocabulary.
        model_load_path (str, optional): Path to the pre-trained model. Defaults to None.

    Returns:
        tuple: A tuple containing the model and its interface.
    """
    model = AutoModelForCausalLM.from_pretrained(args.hf_model_name)

    if model_load_path:
        model.load_state_dict(torch.load(
            model_load_path, map_location=torch.device("cpu")))
   
    interface = create_model_interface(model, is_lm=True, hf=True, in_vocab=in_vocab, has_token_labels=args.lm_with_token_labels)
    return model, interface


def set_seed(args):
    """
    Sets random seeds for reproducibility.

    Args:
        args: Command line arguments.
    """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def init_wandb(args):
    """
    Initializes the wandb environment.

    Args:
        args: Command line arguments.
    """
    wandb.init(project=WANDB_USERS[args.wandb_user]["project"],
               entity=WANDB_USERS[args.wandb_user]["entity"])
    wandb.run.name = f"{args.save_dir}-{args.seed}"
    wandb.run.save()


def get_datasets_and_vocab(args, language_model: bool):
    """
    Retrieves datasets and vocab based on the dataset type specified in args.

    Args:
        args: Command line arguments.
        language_model (bool): Flag to determine if it's a language model.

    Returns:
        tuple: A tuple containing the datasets and input vocabulary.
    """
    if args.dataset == "dyck":
        datasets, in_vocab, _ =  build_datasets_dyck(vocab=args.dyck_vocab)
    elif args.dataset == "tense":
        datasets, in_vocab, _ = build_datasets_tense_inflection()
    elif args.dataset == "ds-addmult-mod10":
        if (args.dsam_hf_file != ""):
            datasets, in_vocab = build_dataset_addmult_hf(args.dsam_hf_file)
        else:
            datasets, in_vocab = build_dataset_addmult_mod10(
                data_file=args.dsam_data_file, min_tree_height=args.dsam_min_tree_height, 
                max_tree_height=args.dsam_max_tree_height, max_tree_width=args.dsam_max_tree_width, 
                hold_out_n_unique_examples=args.dsam_hold_out_n_unique_examples, 
                hold_out_regex=args.dsam_hold_out_regex,
                hold_out_p_subtrees=args.dsam_hold_out_p_subtrees,
                max_held_examples=args.dsam_max_held_examples,
                use_intermediates=args.lm_with_token_labels,
                balance_depths=args.dsam_balance_depths,
                lm_mode=language_model)
    elif args.dataset == "bllip-lg":
        if args.hf:
            in_vocab = AutoTokenizer.from_pretrained(args.hf_model_name)
        else:
            in_vocab = pickle.load(open('/afs/cs.stanford.edu/u/ananjan/grokking-330/structural-grokking-330/data_utils/blimp_vocab.pkl', 'rb'))
        datasets, _, _ = build_datasets_pushdown(
                data_regime="normal",
                only_vocab=False,
                data_file_given="/u/scr/smurty/pushdown-lm/data_utils/bllip-lg-depth",
                randomize = args.randomize,
                hf = args.hf,
                in_vocab = in_vocab
            )
    elif args.dataset == "bllip-md":
        if args.hf:
            in_vocab = AutoTokenizer.from_pretrained(args.hf_model_name)
        else:
            in_vocab = pickle.load(open('/afs/cs.stanford.edu/u/ananjan/grokking-330/structural-grokking-330/data_utils/blimp_vocab.pkl', 'rb'))
        datasets, in_vocab, _ = build_datasets_pushdown(
                data_regime="normal",
                only_vocab=False,
                data_file_given="/u/scr/smurty/pushdown-lm/data_utils/bllip-lg-depth",
                data_ratio=0.001,
                randomize = args.randomize,
                hf = args.hf,
                in_vocab = in_vocab
            )
    elif args.dataset == "bllip-int":
        if args.hf:
            in_vocab = AutoTokenizer.from_pretrained(args.hf_model_name)
        else:
            in_vocab = pickle.load(open('/afs/cs.stanford.edu/u/ananjan/grokking-330/structural-grokking-330/data_utils/blimp_vocab.pkl', 'rb'))
        datasets, in_vocab, _ = build_datasets_pushdown(
                data_regime="normal",
                only_vocab=False,
                data_file_given="/u/scr/smurty/pushdown-lm/data_utils/bllip-lg-depth",
                data_ratio=0.5,
                randomize = args.randomize,
                hf = args.hf,
                in_vocab = in_vocab
            )
    elif args.dataset == "ptb":
        in_vocab = pickle.load(open('/afs/cs.stanford.edu/u/ananjan/grokking-330/structural-grokking-330/data_utils/blimp_vocab.pkl', 'rb'))
        datasets, _ = build_datasets_ptb(
                            in_vocab = in_vocab,
                            data_file_given = "/nlp/scr/horatio/data/constituency",
                            data_ratio = 1.0,
                            randomize = args.randomize
                        )
    else:
        datasets, in_vocab, _ = build_datasets_lm()

    return datasets, in_vocab


def get_callback_fn(args, language_model: bool, model, in_vocab, datasets):
    """
    Returns the appropriate callback function based on the dataset type specified in args.

    Args:
        args: Command line arguments.
        language_model (bool): Flag to determine if it's a language model.
        model: The trained model.
        in_vocab (CharVocabulary): Input vocabulary.
        datasets: The datasets used for training and evaluation.

    Returns:
        function: The corresponding callback function.
    """

<<<<<<< HEAD
    # CHANGES REQUIRED HERE
    if not args.callback:
        return None

    dataset_callbacks = {
        "lm": lambda split: eval_lm_callback(model, in_vocab, split),
        "tense": lambda split: eval_callback_tense_inflection(model, in_vocab, split),
        "dyck": lambda split: eval_callback_dyck(model, in_vocab, split),
        "ds-addmult-mod10": lambda split: eval_callback_mod10_lm(model, in_vocab, split, datasets, eval_batch_size=args.batch_size_eval) \
            if language_model and not args.lm_with_token_labels else \
                eval_callback_mod10(model, in_vocab, split, datasets, eval_batch_size=args.batch_size_eval, has_token_labels=args.lm_with_token_labels),
        "bllip-lg": lambda split, regularizer, override, args: callback_pushdown(model, in_vocab, split, regularizer, args, 
            data_folder_given="/u/scr/smurty/pushdown-lm/data_utils/bllip-lg-depth", override=override, hf=args.hf),
        "bllip-md": lambda split, regularizer, override, args: callback_pushdown(model, in_vocab, split, regularizer, args, data_ratio = 0.001,
            data_folder_given="/u/scr/smurty/pushdown-lm/data_utils/bllip-lg-depth", override=override, hf=args.hf),
        "bllip-int": lambda split, regularizer, override, args: callback_pushdown(model, in_vocab, split, regularizer, args, data_ratio = 0.5,
            data_folder_given="/u/scr/smurty/pushdown-lm/data_utils/bllip-lg-depth", override=override, hf=args.hf),
        "ptb": lambda split, regularizer, override, args: callback_ptb(model, in_vocab, split, regularizer, args,
            data_folder_given="/nlp/scr/horatio/data/constituency", override=override)
    }

    return dataset_callbacks.get(args.dataset, lambda split: Exception("Invalid dataset"))

=======
>>>>>>> b3ce4da34f2346bdda5d7496d402133864818eee
def get_regularizer(args, in_vocab):
    """
    Get tree projection regularizer if required.

    Args:
        args: Command line arguments.
    """
    if (args.distance_fn == "cosine"):
        dist_fn = lambda x1, x2: F.cosine_similarity(x1, x2, dim=-1)
    else:
        dist_fn = lambda x1, x2: -torch.sqrt(torch.sum((x1 - x2)**2, dim = -1))
    if (args.regularize):
        if args.dataset == "ds-addmult-mod10":
            spaces = False
        else:
            spaces = True
        regularizer = Chart(dist_fn, in_vocab, spaces, args)
    else:
        regularizer = None
    if args.proj:
        device = torch.device(f"cuda:{args.gpu_id}")
        regularizer.to(device)
    return regularizer

def main_lm(args):
    """
    Main function for language modeling tasks.

    Args:
        args: Command line arguments.
    """
    language_model = args.lm
    out_vocab = CharVocabulary(chars=set('0123456789'))

    datasets, in_vocab = get_datasets_and_vocab(args, language_model)

    if language_model:
        if args.hf:
            model, interface = get_base_transformer_hf(
                args, in_vocab, model_load_path=args.model_load_path)
        else:
            model, interface = get_base_transformer_lm(
                args, in_vocab, model_load_path=args.model_load_path)
    else:
        model, interface = get_base_transformer_model(
            args, in_vocab, out_vocab, model_load_path=args.model_load_path)

    callback_fn = get_callback_fn(
        args, language_model, model, in_vocab, datasets)
    
    regularizer = get_regularizer(args, in_vocab)

    device = torch.device(f"cuda:{args.gpu_id}")
    model.to(device)
    if args.save_dir:
        dir_path = working_dir()
        args.save_dir = os.path.join(dir_path, args.save_dir)
        os.makedirs(args.save_dir, exist_ok=True)

    eval_keys = ["val", "test"]

    if args.eval_only:
        raise ValueError("Testing functionality not implemented yet!")
    elif args.reg_only:
        reg_loop(
            args,
            interface,
            datasets["train"],
            device,
            regularizer = regularizer
        )
    else:
        train_loop(
            args,
            interface,
            datasets["train"],
            {key: datasets[key] for key in eval_keys},
            device,
            args.save_dir,
            args.save_model,
            args.save_interval,
            in_vocab=in_vocab,
            callback_fn=callback_fn,
            regularizer = regularizer,
            batch_limit = args.batch_limit
        )


def validate_args(args):
    # Check model_load_path and eval_only conditions
    if args.model_load_path:
        if not args.eval_only:
            print("WARNING: Not evaluating, resuming training at loaded model checkpoint")
        else:
            print("INFO: Evaluating model, no training will occur")
    elif args.eval_only:
        raise ValueError("Must specify --model_load_path before evaluating")

    # Print model checkpoint folder if saving is enabled
    if args.save_model:
        print(
            f"INFO: Saving model checkpoints in folder: '{os.path.abspath(os.path.join(os.getcwd(), args.save_dir))}'")

    assert (args.lm_with_token_labels == False or args.lm == True), "If using --lm_with_token_labels, must also enable --lm"



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_load_path", type=str, default="",
                        help="Path of model to resume training, or ")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="")
    parser.add_argument("--dataset", type=str, default="cogs")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--reg_only", action="store_true", help="Just calculate the parses, quick fix")
    parser.add_argument("--dump_errs", action="store_true")
    parser.add_argument("--dump_file", type=str, default="")
    parser.add_argument("--vec_dim", type=int, default=512)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--encoder_n_layers", type=int, default=6)
    parser.add_argument("--mode", type=str, default="enc_dec")
    parser.add_argument("--decoder_n_layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start_lr", type=float, default=6e-4)
    parser.add_argument("--end_lr", type=float, default=0)
    parser.add_argument("--relative", type=bool, default=False)
    parser.add_argument("--rotary", type=bool, default=False)
    parser.add_argument("--pack", action="store_true", help="Implement packing to max_seq_len.")
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--hf", action="store_true", help="Use a model from HuggingFace.")
    parser.add_argument("--hf_model_name", type=str, default="princeton-nlp/Sheared-LLaMA-1.3B")
    parser.add_argument("--lm", type=bool, default=False)
    parser.add_argument("--enc", type=bool, default=False)
    parser.add_argument("--lm_with_token_labels", type=bool, default=False, help="If enabled, LM backprops from per-token labels provided by dataset, instead of the shifted sequence")
    parser.add_argument("--use_amp", action="store_true", help="Mixed Precision Training")

    parser.add_argument("--regularize", action="store_true", help="Turn on tree regularization.")
    parser.add_argument("--randomize", action="store_true", help="Randomize parses for a baseline.")
    parser.add_argument("--mixture", action="store_true", help="Use both greedy and optimal decoding. DO NOT USE CKY WITH THIS.")
    parser.add_argument("--parse_portion", type=float, default=1.0, help="Percentage of data that is parsed.")
    parser.add_argument("--mean_regularize", type=bool, default=False, help="Use mean version of regularizer.")
    parser.add_argument("--distance_fn", type=str, default="cosine", help="Distance function used by regularization.")
    parser.add_argument("--regularizer_rel_wt_init", type=float, default=0.1, help="Initial relative weight of regularizer wrt objective loss.")
    parser.add_argument("--regularizer_rel_wt_end", type=float, default=-1.0, help="Final relative weight of regularizer wrt objective loss.")
    parser.add_argument("--regularizer_steps", type=int, default=2, help="Regularize every regularizer_steps training steps.")
    parser.add_argument("--change_steps", type=int, default=500, help="Increase relative weight of regularizer after this number of regularization steps")
    parser.add_argument("--reg_sample_num", type=int, default=-1, help="Number of phrases sampled for regularization")
    parser.add_argument("--reg_sample_len", type=int, default=10, help="Length of phrases sampled for regularization")
    parser.add_argument("--sample_subtrees", action="store_true", help="Sample entire subtrees of parse tree instead of random phrases")
    parser.add_argument("--layer_id", type=int, default=-1, help="Depth limited SCI computation")
    parser.add_argument("--start_relax_layer", type=int, default=0, help="Context-free lower layers")
    parser.add_argument("--end_relax_layer", type=int, default=0, help="Context-free later layers")
    parser.add_argument("--reg_single", action="store_true", help="Only top level decision for SCI")
    parser.add_argument("--margin", type=int, default=-1, help="Margin for margin loss tree reg")
    parser.add_argument("--retain_positions", action="store_true", help="Compute phrase representation in-position")
    parser.add_argument("--causal_only", action="store_true", help="Use version of code that only makes causal calls to the LM")
    parser.add_argument("--use_gold", action="store_true", help="Sample using gold parses")
    parser.add_argument("--enforce_gold", action="store_true", help="Regularize as per gold parses")
    parser.add_argument("--batch_limit", type=int, default=-1, help="Only compute tree reg on this many strings in the batch. Intended for use with full tree reg")
    parser.add_argument("--use_difference", action="store_true", help="Use difference to compute vectors in SCI score")
    parser.add_argument("--use_gumbel", action="store_true", help="Use gumbel softmax to encourage exploration")
    parser.add_argument("--tau_init", type=float, default=1.0, help="Initial Temperature for gumbel/annealing")
    parser.add_argument("--tau_final", type=float, default=0.1, help="Final temperature for gumbel/annealing")
    parser.add_argument("--use_orthogonal", action="store_true", help="SCI score = norm of orthogonal component")
    parser.add_argument("--balance", action="store_true", help="New idea for orthogonal SCI")
    parser.add_argument("--print_parse", action="store_true", help="Print the SCI parse")
    parser.add_argument("--neg_samples", type=int, default=-1, help="Number of negative samples to penalize SCI for bad constituents")
    parser.add_argument("--neg_rel_wt", type=float, default=0., help="Relative weight of negative samples to penalize SCI for bad constituents")
    
    parser.add_argument("--orth_single", action="store_true", help="Use endpoint of span for SCI computation")
    parser.add_argument("--orth_comp", action="store_true", help="Use only cosine for SCI computation")
    parser.add_argument("--orth_bidir", action="store_true", help="Bidirectional SCI computation")
    parser.add_argument("--proj", action="store_true", help="Projections in SCI computation")
    parser.add_argument("--ce", action="store_true", help="Use cross entropy in (gold) SCI computation")
    parser.add_argument("--sci_heads", type=float, default=-1., help="Proportion of attention heads to be used in SCI computation")
    parser.add_argument("--cky", action="store_true", help="Train tree reg using CKY scores. Always use single for this.")
    parser.add_argument("--cky_dec", action="store_true", help="Decision level CKY.")
    parser.add_argument("--neg_only", action="store_true", help="SCI where random negatives are sampled for training")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--accum_steps", type=int, default=1)
    parser.add_argument("--embedding_dropout", type=float, default=-1.0)
    parser.add_argument("--output_dropout", type=float, default=-1.0)
    parser.add_argument("--batch_size_eval", type=int, default=32)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--eval_every", type=int, default=10000)
    parser.add_argument("--max_train_steps", type=int, default=20000)

    parser.add_argument("--save_model", action="store_true", default=False)
    parser.add_argument("--save_interval", type=int, default=10000)
    # this is only used if args.dataset == pcfg
    parser.add_argument("--base_folder", type=str, default="m-pcfgset")
    parser.add_argument("--tree_transform", action="store_true")
    # evaluating can be time consuming so we can do that later...
    parser.add_argument("--dyck_vocab", type=int, default=20)
    parser.add_argument("--wandb_user", type=str, default=None,
                        required=True, choices=['ananjan', 'derek', 'kyle'])

    parser.add_argument("--callback", action="store_true")
    # args for ds-addmult-mod10
    parser.add_argument("--dsam_data_file", type=str, default="data_utils/ds_addmult_mod10_data/data-addmult-231102-1.5m.csv")
    parser.add_argument("--dsam_hf_file", type=str, default="")
    parser.add_argument("--dsam_min_tree_height", type=int, default=1)
    parser.add_argument("--dsam_max_tree_height", type=int, default=4)
    parser.add_argument("--dsam_max_tree_width", type=int, default=80)
    parser.add_argument("--dsam_hold_out_n_unique_examples", type=int, default=0, help="Hold out this many unique examples and use them as the test set.")
    parser.add_argument("--dsam_hold_out_regex", type=str, default=None, help="Hold out examples which match this regex and use them as the test set. If using >1 holdout option, the union of the two is used as the test set. Accepts unescaped regexes, e.g. (+(*3(+..))(...))")
    parser.add_argument("--dsam_hold_out_p_subtrees", type=float, default=0.0, help="Hold out examples which match this proportion of unique subtrees and use them as the test set.")
    parser.add_argument("--dsam_max_held_examples", type=int, default=None, help="If specified, randomly cut held out set down to this many examples. Useful when holding out subtrees with a lot of data.")
    parser.add_argument("--dsam_balance_depths", action="store_true")

    args = parser.parse_args()

    validate_args(args)

    # Main program
    init_wandb(args)
    set_seed(args)
    main_lm(args)
