from typing import Tuple, Optional
from datasets import load_dataset, DatasetDict, concatenate_datasets
from vocabulary import CharVocabulary
import torch
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler,
)
import re
import collate
from tqdm import tqdm
from util import test_continuations


def build_dataset_addmult_mod10(
    data_file: str, 
    min_tree_height: int = 1,
    max_tree_height: int = 4, 
    max_tree_width: int = 80, 
    hold_out_n_unique_examples: int = 0,
    hold_out_percent: float = 0.0,
    hold_out_regex: Optional[str] = None,
    lm_mode: bool = False,
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
    make_unique (bool): If True, before doing anything else, drop all duplicate examples from the dataset.

    Returns:
    Tuple[DatasetDict, CharVocabulary]: A tuple containing the processed huggingface dataset and the tokenizer used.
    """

    # Load dataset
    dataset = load_dataset("csv", data_files=data_file, split="all")

    # Filter to specific sizes. Do this first, it's very fast
    # max height 4
    dataset = dataset.filter(lambda example: example['height'] <= max_tree_height and example['height'] >= min_tree_height)
    # max width 80
    dataset = dataset.filter(lambda example: example['width'] <= max_tree_width)

    # Held out elements: Use as test set if provided
    dataset_held = None
    if hold_out_n_unique_examples > 0:
        ds_uniques = set(dataset.unique('example'))
        held_out_examples = set(list(ds_uniques)[:hold_out_n_unique_examples])
        dataset_held = dataset.filter(lambda example: example['example'] in held_out_examples, num_proc=8)
        dataset_remainder = dataset.filter(lambda example: example['example'] not in held_out_examples, num_proc=8)
        print(f"# held out random examples: {len(dataset_held)}")
        dataset = dataset_remainder

    if hold_out_percent > 0:
        ds_uniques = set(dataset.unique('example'))
        held_out_examples = set(list(ds_uniques)[:int(hold_out_percent*len(ds_uniques))])
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



    # # demo: hold out examples with a certain string (we aren't doing this yet)
    # if held:
    #     dataset_held = dataset.filter(lambda example: held in example['example'])
    # if remainder:
    #     dataset_remainder = dataset.filter(lambda example: remainder not in example['example'])


    # split into train, val, test

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

    # optionally: add arg above to remove the unnecessary columns:
    # remove_columns=['height', 'width', 'example', 'answer', 'ans_mod10'])
    tokenizer = CharVocabulary(chars=set('0123456789+*()='))
    remove_columns = ['height', 'width', 'example', 'answer', 'ans_mod10']
    if lm_mode:
        dataset = dataset.map(lambda example, idx: {  
            'in': tokenizer(example['example'] + '=' + str(example['ans_mod10'])),
            'in_len': len(tokenizer(example['example'] + '=' + str(example['ans_mod10']))),
            'idxs': idx,
            'string': example['example']
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

def eval_callback_mod10(model, in_vocab, split, datasets, eval_batch_size=32, dump=False, dump_file_path=""):
    """PROTOTYPE. Not finished"""
    # Assuming 'datasets' is a dictionary containing your data splits (train, val, test, etc.)
    # and 'split' is the key for the data split you want to evaluate on (e.g., 'val', 'test').
    data = datasets[split]

    # Initialize counters
    total_count = 0
    correct_count = 0

    # Set the model to evaluation mode
    eval_data_collator = collate.VarLengthCollate(None)
    model.eval()

    if (dump):
        dump_file = open(dump_file_path, 'w')

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
            outputs = model(input, input_len, None, None, None)
            _, predicted_class = torch.max(outputs, dim=1)

            # Update accuracy counters
            total_count += label.size(0)
            correct_count += (predicted_class == label).sum().item()

            if (dump):
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
    
    if (dump):
        dump_file = open(dump_file_path, 'w')

    out = test_continuations(tokenizer, lm, queries, 0, batch_size=eval_batch_size)
    desired_out = '0123456789'
    desired_out_idx = [in_vocab(w) for w in desired_out]
    out = out[:, desired_out_idx]
    decoded = out.argmax(dim = 1)
    acc = [int(desired_out[i]) == target for i, target in zip(decoded, targets)]

    if (dump):
        decoded = decoded.cpu().numpy()
        for idx in range(len(targets)):
            if (targets[idx] != int(desired_out[decoded[idx][0]])):
                dump_file.write(queries[idx][:-1] + '\n')

    acc = sum(acc)/len(targets)
    return acc