from typing import Tuple, Optional
from datasets import load_dataset, DatasetDict, concatenate_datasets
from vocabulary import CharVocabulary
import torch
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler,
)
import re, random
import collate
from tqdm import tqdm
from util import test_continuations


def build_dataset_addmult_mod10(
    data_file: str, 
    min_tree_height: int = 1,
    max_tree_height: int = 4, 
    max_tree_width: int = 80, 
    hold_out_n_unique_examples: int = 0,
    hold_out_regex: Optional[str] = None,
    hold_out_p_subtrees: float = 0.0,
    max_held_examples: int = None,
    lm_mode: bool = False,
    use_intermediates: bool = False,
    balance_depths: bool = False,
) -> Tuple[DatasetDict, CharVocabulary]:
    """
    Build an addmult mod10 dataset with specific constraints and tokenize the examples.

    This function loads a dataset from a CSV file, filters examples based on certain conditions,
    splits the dataset into training, validation, and test subsets, and tokenizes the examples
    using a character vocabulary. The tokenization is based on a specific set of characters
    relevant to the dataset.

    Parameters:
    data_file (str): The path to the CSV file to load the dataset from.
    max_tree_height (int): The maximum height for the trees in the dataset.
    max_tree_width (int): The maximum width for the trees in the dataset.
    hold_out_n_unique_examples (int): Take this many unique examples and use them as the test set.
    hold_out_regex (Optional[str]): Take examples that match this regex and use them as the test set.
    hold_out_p_subtrees (int): Sample this percent of all unique subtrees, and use all items that match them as the test set.
    max_held_examples (int): If provided, sample this many examples from the held out set.
    use_intermediates (bool): If True, use intermediate labels instead of just final label.

    Returns:
    Tuple[DatasetDict, CharVocabulary]: A tuple containing the processed huggingface dataset and the tokenizer used.
    """

    # Sanity checks
    if use_intermediates:
        assert(lm_mode)

    # Load dataset
    dataset = load_dataset("csv", data_files=data_file, split="all")

    # Filter to specific sizes. Do this first, it's very fast
    # max height 4
    dataset = dataset.filter(lambda example: example['height'] <= max_tree_height and example['height'] >= min_tree_height)
    # max width 80
    dataset = dataset.filter(lambda example: example['width'] <= max_tree_width)

    # Balance depths
    if balance_depths:
        datasets_by_depth = {}
        for i in range(min_tree_height, max_tree_height + 1):
            datasets_by_depth[i] = dataset.filter(lambda example: example['height'] == i)

        min_depth = min([len(datasets_by_depth[i]) for i in range(min_tree_height, max_tree_height + 1)])
        print(f"Balancing depths. Min examples in any depth: {min_depth}")

        # sample min_depth examples from each depth
        for i in range(min_tree_height, max_tree_height + 1):
            datasets_by_depth[i] = datasets_by_depth[i].select(range(min_depth))

        # concatenate all the datasets
        dataset = concatenate_datasets([datasets_by_depth[i] for i in range(min_tree_height, max_tree_height + 1)])
        print(f"Total examples after balancing depths: {len(dataset)}")

    # Held out elements: Use as test set if provided
    dataset_held = None
    if hold_out_n_unique_examples > 0:
        ds_uniques = set(dataset.unique('example'))
        held_out_examples = set(list(ds_uniques)[:hold_out_n_unique_examples])
        dataset_held = dataset.filter(lambda example: example['example'] in held_out_examples, num_proc=8)
        dataset_remainder = dataset.filter(lambda example: example['example'] not in held_out_examples, num_proc=8)
        print(f"# held out random examples: {len(dataset_held)}")
        dataset = dataset_remainder

    if hold_out_regex is not None:
        # escape here because providing escaped regex to launch.json and sweeps is messy (they have their own backslash escaping)
        hold_out_regex = hold_out_regex.replace('(', '\(').replace(')', '\)').replace('+', '\+').replace('*', '\*')
        dataset_held_regex = dataset.filter(lambda example: re.search(hold_out_regex, example['example']) is not None, num_proc=8)
        dataset_remainder = dataset.filter(lambda example: re.search(hold_out_regex, example['example']) is None, num_proc=8)
        dataset = dataset_remainder
        print(f"# held out examples via regex: {len(dataset_held_regex)}")
        if len(dataset_held_regex) > 0:
            if dataset_held is None:
                dataset_held = dataset_held_regex
            else:
                dataset_held = concatenate_datasets([dataset_held, dataset_held_regex])

    if hold_out_p_subtrees > 0.0:
        # sample unique subtrees
        ds_uniques = set(dataset.unique('tree_sig'))
        n_subtrees = int(len(ds_uniques) * hold_out_p_subtrees)
        held_out_subtrees = set(random.sample(ds_uniques, n_subtrees))

        # hold out all examples that match those subtrees
        dataset_held_subtrees = dataset.filter(lambda example: example['tree_sig'] in held_out_subtrees, num_proc=8)
        dataset_remainder = dataset.filter(lambda example: example['tree_sig'] not in held_out_subtrees, num_proc=8)
        print(f"{len(held_out_subtrees)}/{len(ds_uniques)} randomly held out subtrees: {held_out_subtrees}")
        dataset = dataset_remainder
        print(f"# held out examples via subtree exclusion: {len(dataset_held_subtrees)}")
        if len(dataset_held_subtrees) > 0:
            if dataset_held is None:
                dataset_held = dataset_held_subtrees
            else:
                dataset_held = concatenate_datasets([dataset_held, dataset_held_subtrees])

    if dataset_held is not None and max_held_examples is not None:
        max_held_examples = min(max_held_examples, len(dataset_held)) # exception if max_held_examples > len(dataset_held)
        dataset_held = dataset_held.train_test_split(train_size=max_held_examples, shuffle=True)['train']
        print(f"Cutting down held out set tp max_held_examples: {len(dataset_held)}")

    if dataset_held is None:
        train_valtest = dataset.train_test_split(test_size=0.2, shuffle=False)
        val_test = train_valtest['test'].train_test_split(test_size=0.5, shuffle=False)
        val, test = val_test['train'], val_test['test']
    else:
        train_valtest = dataset.train_test_split(test_size=0.1, shuffle=False)
        val = train_valtest['test']
        test = dataset_held

    dataset = DatasetDict({
        'train': train_valtest['train'],
        'val': val,
        'test': test,
    })

    tokenizer = CharVocabulary(chars=set('0123456789+*()[]xyzL=_'), ignore_char='_', ignore_char_idx=0)
    remove_columns = ['height', 'width', 'example', 'answer', 'ans_mod10', 'ans_sublabels', 'tree_sig']
    if lm_mode and not use_intermediates:
        dataset = dataset.map(lambda example, idx: {  
            'in': tokenizer(example['example'] + '=' + str(example['ans_mod10'])),
            'in_len': len(tokenizer(example['example'] + '=' + str(example['ans_mod10']))),
            'idxs': idx,
            'string': example['example']
        }, with_indices=True,
        remove_columns=remove_columns)
    elif lm_mode and use_intermediates:
        dataset = dataset.map(lambda example, idx: {  
            'in': tokenizer(example['example']),
            'in_len': len(tokenizer(example['example'])),
            'idxs': idx,
            'labels': tokenizer(str(example['ans_sublabels'])),
            'labels_len': len(str(example['ans_sublabels'])) - str(example['ans_sublabels']).count('_'), # number of close brackets
        }, with_indices=True,
        remove_columns=remove_columns)
    else:
        dataset = dataset.map(lambda example, idx: {  
            'in': tokenizer(example['example']),
            'in_len': len(tokenizer(example['example'])),
            'labels': example['ans_mod10'],
            'idxs': idx,
            'string': example['example']
        }, with_indices=True,
        remove_columns=remove_columns)

    return dataset, tokenizer


def eval_callback_mod10(model, in_vocab, split, datasets, eval_batch_size=32, has_token_labels=False, dump=False, dump_file_path=""):
    """
    PROTOTYPE. Not finished
    
    dump, dump_file_path: if dump is True, dump all incorrectly predicted
        examples to dump_file_path. Only works for EncDec models.
    """
    # Assuming 'datasets' is a dictionary containing your data splits (train, val, test, etc.)
    # and 'split' is the key for the data split you want to evaluate on (e.g., 'val', 'test').
    # If has_token_labels is True, dataset includes labels for each token, not just one final label.
    data = datasets[split]

    # Initialize counters
    total_count = 0
    correct_count = 0

    # Set the model to evaluation mode
    eval_data_collator = collate.VarLengthCollate(None)
    model.eval()

    with torch.no_grad():  # Disable gradient calculations during evaluation
        eval_dataloader = DataLoader(
            data,
            sampler=SequentialSampler(data),
            batch_size=eval_batch_size,
            collate_fn=eval_data_collator,
        )

        for batch in tqdm(eval_dataloader):
            # Move tensors to the same device as the model
            device = next(model.parameters()).device
            input = batch['in'].transpose(0,1).to(device)
            input_len = batch['in_len'].long().to(device)
            label = batch['labels'].long().to(device)

            # Get the model's predictions
            if has_token_labels: # TransformerLM
                prefix_sos = (torch.ones((input.shape[0],1)).long() * model.encoder_sos).to(device)
                input = torch.cat((prefix_sos, input), dim=1) # add SOS token
                input_len += 1

                outputs = model(input, input_len)
                _, preds = torch.max(outputs['data'], dim=2)

                labels_cmp = torch.cat((label.T, prefix_sos), dim=1)
                labels_match = (preds == labels_cmp) * (input != 0) * (labels_cmp != 0)

                # Update accuracy counters
                total_count += batch['labels_len'].sum().item()
                correct_count += labels_match.sum().item()

            else: # EncDec
                outputs = model(input, input_len, None, None, None)
                _, predicted_class = torch.max(outputs, dim=1)
                # Update accuracy counters
                total_count += label.size(0)
                correct_count += (predicted_class == label).sum().item()

                if (dump):
                    with open(dump_file_path, 'w') as dump_file:
                        strings = batch['string']
                        label = label.cpu().numpy()
                        predicted_class = predicted_class.cpu().numpy()
                        for idx in range(len(label)):
                            if (label[idx] != predicted_class[idx]):
                                dump_file.write(strings[idx] + '\n')


    # Calculate accuracy
    accuracy = correct_count / total_count
    print(f'Accuracy on {split} data: {accuracy:.2f}')
    return accuracy

def eval_callback_mod10_lm(lm, in_vocab, split, datasets, eval_batch_size=32, dump=False, dump_file_path=""):
    """PROTOTYPE. Not finished"""
    # Assuming 'datasets' is a dictionary containing your data splits (train, val, test, etc.)
    # and 'split' is the key for the data split you want to evaluate on (e.g., 'val', 'test').
    def tokenizer(s):
        return [lm.encoder_sos] + in_vocab(s)
    
    data = datasets[split]

    # Create the evaluation setting
    queries = []
    targets = []

    for ex in data:
        stringified = in_vocab.ind_to_str(ex['in'])
        [qs, ts] = stringified.split('=')
        queries.append(qs + '=')
        targets.append(int(ts))

    out = test_continuations(tokenizer, lm, queries, 0, batch_size=eval_batch_size)
    desired_out = '0123456789'
    desired_out_idx = [in_vocab(w) for w in desired_out]
    out = out[:, desired_out_idx]
    decoded = out.argmax(dim = 1)
    acc = [int(desired_out[i]) == target for i, target in zip(decoded, targets)]

    if (dump):
        with open(dump_file_path, 'w') as dump_file:
            decoded = decoded.cpu().numpy()
            for idx in range(len(targets)):
                if (targets[idx] != int(desired_out[decoded[idx][0]])):
                    dump_file.write(queries[idx][:-1] + '\n')

    acc = sum(acc)/len(targets)
    return acc