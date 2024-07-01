# Running Tree Regularization

## Code Structure

1. `train_transformers.py` is used to run all training. All command line arguments to this file can be found in the `__main__` function. I will also provide examples of commands to run the code for specific experiments as they come up.
2. `get_datasets_and_vocab` loads the relevant datasets for experiments, as well as the input vocabulary. The BLLIP datasets can be loaded with compatibility both for training a LM from scratch (as in the original codebase) and for continued pre-training of a HF model. This logic is handled by `build_datasets_pushdown`. Since this function contains a lot of important functionality, I will discuss this in a separate section below.
3. Next, the model is initialized.`get_base_transformer_hf` initializes HF models and `get_base_transformer_lm` initializes a LM from scratch. The `TransformerLMInterface` provides some basic functionalities such as computation of the LM loss on given input, as well handling of EOS and SOS tokens. This part of the code handles input differently based on whether it is packed or not. `TransformerHFInterface` is the counterpart for HF models, providing the same functionality (the key point of difference is that there is an additional `generate_mask` function to handle padding, this function is required by a lot of other functions and is part of the model definition for the `Transformer` defined in the codebase). 
4. Next, the callback functions are initialized. This is `callback_pushdown` for LM training. It does the following things: i) Loads the BLLIP test/dev set (and processes it appropriately if HF training is being used). ii) Get parsing accuracy against gold parse trees. Parses are obtained one sentence at a time by calling `get_parse` from the regularizer. Setting `override` to True returns CKY parses, greedy parses are returned otherwise. iii) Get perplexity over the test/dev set by calling `eval_base_model`. iv) 
Get performance on BLIMP. This is done by loading (and reformatting, if required) the GPT-2 tokenized BLIMP trees, and checking if logprobs for the correct tree is higher than logprobs for the incorrect counterpart. `make_preds_base_model` is an important function here, discussed in greater detail below. 
5. Finally, the regularizer is initialized and the training started inside `train_loop` (`training_utils.py`). 
6. Inside the training loop, first the optimizer and scheduler are initialized. The LR scheduler follows a linear warmup followed by cosine decay (`get_scheduler`).
7. If packing is to be done, a PackedDataset is initialized to created the packed batches. More details below.
8. A DataLoader is created for the train dataset. In case we wish to only use parses from a portion of the dataset, a second `parse_dataset` is created containing the desired subset of examples in the dataset. A dataloader is created for this dataset that is analogous to the train dataloader. Examples to be passed to the tree regularizer are sampled from this `parse_dataset`.
9. See `Train Loop` below for a description of how each batch is processed in the train loop.

### `build_datasets_pushdown`

This function can be used to load the BLLIP datasets from disk, make the trees in the dataset consistent with a HF model if required, or replace the trees with randomly-generated counterparts. 

1. The trees are first loaded from disk as NLTK objects, in `read_data`. There is also functionality for using random trees instead, see `build_random_tree`.
2. If we are planning to train a HF model, we need to reformat the dataset according to the HF tokenizer. Two helper functions are used for this: i) `reformat_sentence` ensures that the tokens corresponding to start of every word after the first one starts with # This can be ignored, SCI(i,k,j) = norm(orth(orth(k,i), orth(j,i))) and that the sentence is tokenized according to the HF tokenizer (after being flattened down from the GPT-2 tokenized format). ii) `reformat_tree` takes the input tree that is tokenized according to GPT-2, and converts it so that the leaves corresponding to tokens from the HF tokenizer. This is done by traversing the tree until we hit a subtree representing a single word, recovering the original word by joining the leaves, and retokenizing it according to the HF tokenizer. The newly obtained tokens are then used as the leaves of the new subtree. 
3. Finally, we return the tokenized version of the input sentences (to be passed to LM training), appropriately-processed string version of the input sentences and corresponding parse trees (to be passed to tree regularizer). The parses are represented as a dictionary indexed by string representations of span start and end indices (look into `tree_to_parse_decisions` for details). 

### `make_preds_base_model`

This function is used to get logprobs for a LM on a set of input sentences. It can handle LMs from HuggingFace as well (which require different handling for the tokenization, as well as a different `generate_mask` function). The output format is also different for HF models. Specifically, the function does the following:

1. Tokenizes the input sentences after batching them with a constant batch size.
2. Passed input batches to the LM.
3. Extracts the resulting logits from the output, excluding padding tokens.
4. Calls `compute_per_token_logprob` to get the log-probabilities of the target tokens (= input tokens shifted by one) and return them. 

### Train Loop

1. A batch from the train dataloader is sent to the GPU and passed to the model `Interface`, and the LM loss is returned. Mixed precision training is supported with the `--use_amp` argument.
2. Every `regularizer_steps` of training, the tree regularizer is invoked. This involves sampling a batch from the parse dataloader (=the train dataloader when all of the parses are to be used). All strings in the current batch are passed as a list to the tree regularizer (packed strings are unpacked). 
3. There is a provision for a `batch_limit` argument, which can be used to restrict the number of strings passed at once to the tree regularizer, as well as the maximum length of a passed string. This was required with the old version of tree regularization, and is mostly ignored now.
4. The regularizer involves a first call to a `build_scores` method to build the SCI score chart (and optionally perform phrase sampling on the input), and a second call to `get_score` to get the actual SCI loss. The internal logic of these functions is covered separately.
5. There is a `mixture` argument that can be set to train both using CKY and greedy parsing at the same time. In this case, the SCI loss for CKY and greedy are simply added to each other.
6. Finally, backward is called on the LM and SCI loss (divided by `accum_steps` for gradient accumulation). Every `accum_steps`, the following things happen: i) the gradients are clipped ii) The grad norms, LM and SCI losses are logged iii) `bracket_acc_train` is logged. This is intended to provide a more granular view at how the parsing accuracy evolves over the course of training. When the gold parse tree is traversed for SCI loss computation, I keep track at every node whether the highest-scoring split point is the gold parse split point. `bracket_acc_train` is then the proportion of nodes for which the gold and high-scoring split point coincides. iv) The optimizer and scheduler are updated and the model grads are zeroed.
7. Every `save_interval` steps, the model is saved. Alternatively, if it is set to 0, the model checkpoint is saved every 10k steps, and the previous checkpoint is overwritten (this odd behavior is just to save disk space without interfering with any existing functionality.)
8. Every `eval_every` steps, the callback function is called and the results logged. Callbacks are called on the train, test and valid sets, although perplexity computation only happens on the test and valid sets, and the only first 3k examples are considered for train parse accuracy.

### Packing

The PackedDataset class largely handles the logic required for the packing implementation (although the loss computation for packed batches is handled inside the relevant `Interface` class). It does the following: i) Add SOS to all strings in the dataset. ii) Sort all strings in the dataset by length. iii) In `get_packed_indices`, traverse the sorted list and create batches of indices such that the total length of strings in each batch is as close as possible to `max_sequence_len` without exceeding it. iv) Return batches containing tokenized as well as string representations of the packed input, the length of each packed string, as well as the strings packed in the current batch with corresponding parses (for each string individually).

## Data Contracts

I provide an overview of the input formats expected by different parts of the code.

1. The most important data contract involves the tree regularizer. The tree regularizer expects three things: i) The input string should be a space-separated join of the input tokenized according to the appropriate model (i.e. `" ".join(vocab.tokenize(input))`) ii) The token demarcating the beginning of every word except for the first one should start with 'Ġ' (like GPT-2 tokenization) iii) Each token in the input string should correspond to one leaf of the input parse tree.
2. In contrast, `make_preds_base_model` assumes that the input is simply flattened sentences, with no processing (as word information is not required).

## Tree Regularization

The tree regularizer consists of a `Chart`, `PhraseSampler` and `ChartComputer` object. The `Chart` is a wrapper that handles SCI chart computation (through the `PhraseSampler` and `ChartComputer`) as well as SCI score computation. The `PhraseSampler` samples phrases from input strings following a variety of heuristics. The `ChartComputer` has several different versions, computing the SCI chart according to a variety of methods.  A description of the different settings for the tree regularization computation can be found in the `Chart` init function.

### `PhraseSampler`

This can mostly be ignored, it was relevant to the previous version of tree regularization. If necessary, it can return phrases sampled from input strings according to a variety of heuristics, and corresponding parses, and some other book-keeping information.

### `ChartComputer`

The only relevant computer is the `OrthogonalChartComputer`. `build_scores` does the following:

1. Calls the `PhraseSampler` to return sampled phrases.
2. If HF model is being used, the text used to compute hidden states cannot have the 'Ġ' character. Therefore, they are cleaned and the original string is recovered.
3. The input sentences are tokenized, and a `idxs` dict is created that maps each token to its span in the tokenized result.
4. `get_all_hidden_states_scratch` is called to get hidden states for the input strings, after `layer_id`. This is very similar to `make_preds_base_model`, but supports both packing and getting hidden states from arbitrary layers of the model. 
5. The model computes the orthogonals for all spans `(i,j)` in the input. This is done individually for all of the strings in the input. The orthogonal computation for all spans starting at `i` is batched together. I compute the maximum end point `j` of a span starting at `i`, and compute orthogonals for all spans `(i,i+1) ... (i,j)` at the same time. `get_orthogonal_score` contains the logic for how this is done

**NOTE:** This will change once I implement the vectorization trick for orthogonal computation. 

### `Chart`

After the `ChartComputer` returns the SCI chart, `get_score` is where the SCI score computation happens. There are a lot of different scoring functions here, related to different ways of computing the SCI scores. I will provide a short description of all of them, there is more information in the comments inside the code. `gold_recurse` and `cky_reg` are the important functions here. For all of the different scoring functions, `get_penalty` implements common logic for getting `SCI(st, k, en)` from the chart scores. Note that chart indices are off by 1: `chart[(k,k)] = norm(orth(k,k-1))`. All scoring functions contain provisions for additionally negative sampling some constituents and driving their SCI scores down as part of the loss function.

`recurse`: unsupervised greedy recursive tree regularization (computes SCI score for the full tree). Also contains some very messy logic for supervised tree regularization on SimPL.

`gold_recurse`: supervised greedy recursive tree regularization (SCI score for full tree, with reference to a gold parse). Comments in the code for more details.

`get_score_single`: unsupervised greedy tree regularization at phrase level. Essentially only computes scores for each phrase sampled in the input, with no recursion.

`get_score_single_gold`: same as above, but with gold parse supervision.

`cky_decision_level_loss`: CKY loss with supervision at each decision. For each phrase on the gold parse tree, enforce the correct split.

`cky_reg`: CKY loss with supervision at the tree-level. Score of gold parse tree according to CKY with random trees as negative samples. The CKY chart is computed on CPU, and transferred to GPU for loss computation.

`negative_sample_reg`: Simplest version of tree regularization, just use constituents in the gold parse as positive samples and randomly sampled constituents not in the gold parse as negative samples. Loss = Score for positive constituents - alpha * score for negative constituents.

`get_parse` gets greedy parses for an input string. `get_parse_beam` implements some complicated logic to get parses through beam search for an input string. `get_parse_cky` gets CKY parses for an input string.

## Evaluation Scripts

### `eval_utils/eval_surprisal.py`

Runs the SyntaxGym test suite on the model, with support for HF models as well (compatibility through `make_preds_base_model`). It is also possible to print out parses for sentences in the test suite, although right now this is only done for `fgd` and `center_embed`.

Examples of use:

```
python eval_utils/eval_surprisal.py --model_load_path /nlp/scr/ananjan/llama-bllip-lg-base/state.pt --relative --layer_id 20 --sci_heads 0.25 --use_orthogonal --orth_bidir --orth_single --hf --hf_model_name princeton-nlp/Sheared-LLaMA-1.3B

python eval_utils/eval_surprisal.py --model_load_path /nlp/scr/ananjan/bllip-lg-base/state.pt --relative
```

### `eval_utils/eval_ptb.py`

Compute parsing accuracy on PennTreeBank with greedy, CKY and beam parses. Also computes perplexities for sentences parsed, along with a bunch of other stats such as the variation of parsevals with sentence lengths. Prints out parses with parsevals less than a certain threshold.

Examples of use:

```
python eval_utils/eval_ptb.py --model_load_path /nlp/scr/ananjan/llama-bllip-lg-base/state.pt --relative --layer_id 20 --sci_heads 0.25 --use_orthogonal --orth_bidir --orth_single --hf --hf_model_name princeton-nlp/Sheared-LLaMA-1.3B

python eval_utils/eval_ptb.py --model_load_path /nlp/scr/ananjan/bllip-lg-base/state.pt --relative

python eval_utils/eval_ptb.py --model_load_path /nlp/scr/ananjan/bllip-lg-treereg/state.pt --relative --layer_id 20 --sci_heads 0.25 --use_orthogonal --orth_bidir --orth_single
```

## Sample Model Calls (all on BLLIP-LG)

### Train a LM from Scratch without Tree Reg

```
python train_transformers.py --dataset=bllip-lg --save_dir /nlp/scr/ananjan/bllip-lg-16layer-base-relative/ --encoder_n_layers 16 --seed 10 --callback --lm True --max_train_steps 10000 --wandb_user ananjan --save_model --save_interval 0 --eval_every 1000 --batch_size 16 --accum_steps 10 --start_lr 1e-4 --relative True
```

### Train a LM from Scratch without Tree Reg with Packing up to 512 tokens and Mixed Precision

```
python train_transformers.py --dataset=bllip-lg --save_dir /nlp/scr/ananjan/bllip-lg-16layer-base-relative-pack512/ --encoder_n_layers 16 --seed 10 --callback --lm True --max_train_steps 10000 --wandb_user ananjan --save_model --save_interval 0 --eval_every 1000 --pack --max_seq_len 512 --batch_size 16 --accum_steps 10 --start_lr 1e-4 --relative True --use_amp
```

### Train a LM from Scratch with Tree Reg

```
python train_transformers.py --dataset=bllip-lg --save_dir /nlp/scr/ananjan/bllip-lg-16layer-1.0-l12-a0.25-ce-bidir/ --encoder_n_layers 16 --seed 10 --callback --lm True --max_train_steps 60000 --wandb_user ananjan --save_model --save_interval 0 --eval_every 1000 --batch_size 32 --accum_steps 5 --start_lr 1e-4 --end_lr 6e-5 --relative True --use_amp --regularize --regularizer_rel_wt_init 1.0 --regularizer_steps 10 --embedding_dropout 0.1 --output_dropout 0.1 --use_gold --enforce_gold --use_orthogonal --orth_single --orth_bidir --layer_id 12 --sci_heads 0.25 --ce
```

### Train a LM from Scratch with Tree Reg (CKY)

```
python train_transformers.py --dataset=bllip-lg --save_dir /nlp/scr/ananjan/bllip-lg-16layer-1.0-l4-a0.125-ce-cky-bidir/ --encoder_n_layers 16 --seed 10 --callback --lm True --max_train_steps 50000 --wandb_user ananjan --save_model --save_interval 0 --eval_every 1000 --batch_size 12 --accum_steps 10 --lr 1e-4 --relative True --regularize True --regularizer_rel_wt_init 1.0 --regularizer_steps 5 --embedding_dropout 0.1 --output_dropout 0.1 --use_gold --enforce_gold --use_orthogonal --orth_single --orth_bidir --layer_id 4 --sci_heads 0.25 --ce --cky --cky_dec --use_amp
```

### Train a LM from Scratch with Tree Reg on 10% of the Parses

```
python train_transformers.py --dataset=bllip-lg --save_dir /nlp/scr/ananjan/bllip-lg-16layer-1.0-l12-a0.25-ce-bidir/ --encoder_n_layers 16 --seed 10 --callback --lm True --max_train_steps 60000 --wandb_user ananjan --save_model --save_interval 0 --eval_every 1000 --batch_size 32 --accum_steps 5 --start_lr 1e-4 --end_lr 6e-5 --relative True --use_amp --regularize --regularizer_rel_wt_init 1.0 --regularizer_steps 10 --embedding_dropout 0.1 --output_dropout 0.1 --use_gold --enforce_gold --use_orthogonal --orth_single --orth_bidir --layer_id 12 --sci_heads 0.25 --ce --parse_portion 0.1
```

### Train a LM from Scratch with Tree Reg on 20 sampled phrases per input string, each with max length up to 100 characters

```
python train_transformers.py --dataset=bllip-lg --save_dir /nlp/scr/ananjan/bllip-lg-16layer-1.0-l8-a0.25-ce-20s100-bidir/ --encoder_n_layers 16 --seed 10 --callback --lm True --max_train_steps 50000 --wandb_user ananjan --save_model --save_interval 0 --eval_every 1000 --batch_size 12 --accum_steps 10 --lr 1e-4 --relative True --regularize True --regularizer_rel_wt_init 1.0 --regularizer_steps 5 --reg_single --reg_sample_num 20 --reg_sample_len 100 --embedding_dropout 0.1 --output_dropout 0.1 --use_gold --enforce_gold --use_orthogonal --orth_single --orth_bidir --layer_id 8 --sci_heads 0.25 --ce
```

### Train a LM from Scratch with Tree Reg with Packing up to 512 tokens and Mixed Precision

```
python train_transformers.py --dataset=bllip-lg --save_dir /nlp/scr/ananjan/bllip-lg-16layer-relative-pack512-1.0-l12-a0.25-ce-bidir/ --encoder_n_layers 16 --seed 10 --callback --lm True --max_train_steps 10000 --wandb_user ananjan --save_model --save_interval 0 --eval_every 100 --pack --max_seq_len 512 --batch_size 16 --accum_steps 5 --start_lr 1e-4 --end_lr 6e-5 --relative True --use_amp --regularize --regularizer_rel_wt_init 1.0 --regularizer_steps 10 --embedding_dropout 0.1 --output_dropout 0.1 --use_gold --enforce_gold --use_orthogonal --orth_single --orth_bidir --layer_id 12 --sci_heads 0.25 --ce
```

### Train Sheared Llama without Tree Reg

```
python train_transformers.py --dataset=bllip-lg --save_dir /nlp/scr/ananjan/llama-bllip-lg-base/ --seed 10 --callback --lm True --max_train_steps 10000 --wandb_user ananjan --save_model --save_interval 0 --eval_every 1000 --batch_size 12 --accum_steps 25 --start_lr 1e-5 --relative True --embedding_dropout 0.1 --output_dropout 0.1 --hf --hf_model_name princeton-nlp/Sheared-LLaMA-1.3B
```

### Train Sheared Llama with Tree Reg

```
python train_transformers.py --dataset=bllip-lg --save_dir /nlp/scr/ananjan/llama-bllip-lg-16layer-1.0-l12-a0.25-ce/ --encoder_n_layers 16 --seed 10 --callback --lm True --max_train_steps 10000 --wandb_user ananjan --save_model --save_interval 0 --eval_every 1000 --batch_size 12 --accum_steps 25 --start_lr 1e-4 --relative True --regularize --regularizer_rel_wt_init 1.0 --regularizer_steps 5 --embedding_dropout 0.1 --output_dropout 0.1 --use_gold --enforce_gold --use_orthogonal --orth_single --layer_id 12 --sci_heads 0.25 --ce --hf --hf_model_name princeton-nlp/Sheared-LLaMA-1.3B
```