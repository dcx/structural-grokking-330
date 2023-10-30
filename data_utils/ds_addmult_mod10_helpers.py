from typing import Tuple, Optional
from datasets import load_dataset, DatasetDict
from vocabulary import CharVocabulary
import torch
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler,
)
import collate
from tqdm import tqdm

def build_dataset_addmult_mod10_lm(
    data_file: str, 
    max_tree_height: int = 4, 
    max_tree_width: int = 80, 
    held: Optional[str] = None, 
    remainder: Optional[str] = None
) -> Tuple[DatasetDict, CharVocabulary]:
    """
    Build an addmult mod10 dataset with specific constraints and tokenize the examples.

    This function loads a dataset from a CSV file, filters examples based on certain conditions,
    splits the dataset into training, validation, and test subsets, and tokenizes the examples
    using a character vocabulary. The tokenization is based on a specific set of characters
    relevant to the dataset. The datasets are processed for the LM task. 

    Parameters:
    data_file (str): The path to the CSV file to load the dataset from.
    max_tree_height (int): The maximum height for the trees in the dataset.
    max_tree_width (int): The maximum width for the trees in the dataset.
    held (Optional[str]): A string to filter examples that contain this substring. If None, no filtering is applied.
    remainder (Optional[str]): A string to filter examples that do not contain this substring. If None, no filtering is applied.

    Returns:
    Tuple[DatasetDict, CharVocabulary]: A tuple containing the processed huggingface dataset and the tokenizer used.
    """

    # Load dataset
    dataset = load_dataset("csv", data_files=data_file, split="all")

    # filter to specific sizes
    # max height 4
    dataset = dataset.filter(lambda example: example['height'] <= max_tree_height)
    # max width 80
    dataset = dataset.filter(lambda example: example['width'] <= max_tree_width)

    # demo: hold out examples with a certain string (we aren't doing this yet)
    if held:
        dataset_held = dataset.filter(lambda example: held in example['example'])
    if remainder:
        dataset_remainder = dataset.filter(lambda example: remainder not in example['example'])

    # split into train, val, test
    train_testval = dataset.train_test_split(test_size=0.2, shuffle=False)
    test_val = train_testval['test'].train_test_split(test_size=0.5, shuffle=False)
    dataset = DatasetDict({
        'train': train_testval['train'],
        'val': test_val['test'],
        'test': test_val['train']
    })

    tokenizer = CharVocabulary(chars=set('0123456789+*()='))
    dataset = dataset.map(lambda example, idx: {  
        'in': tokenizer(example['example'] + '=' + str(example['ans_mod10'])),
        'in_len': len(tokenizer(example['example'] + '=' + str(example['ans_mod10']))),
        # 'str': example['example'] + '=' + str(example['ans_mod10']),
        #'out_len': 1,
        #'out': example['ans_mod10'],
        # 'labels': example['ans_mod10'],
        'idxs': idx,
    }, with_indices=True,
    remove_columns=['height', 'width', 'example', 'answer', 'ans_mod10'])

    # optionally: add arg above to remove the unnecessary columns:
    # remove_columns=['height', 'width', 'example', 'answer', 'ans_mod10'])

    return dataset, tokenizer


def eval_callback_mod10(model, in_vocab, split, datasets):
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

    with torch.no_grad():  # Disable gradient calculations during evaluation
        eval_batch_size = 8
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


    # Calculate accuracy
    accuracy = correct_count / total_count
    print(f'Accuracy on {split} data: {accuracy:.2f}')
    return accuracy

def eval_callback_mod10_lm(lm, in_vocab, split, datasets):
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

    out = test_continuations(tokenizer, lm, queries, 0)
    desired_out = '0123456789'
    desired_out_idx = [in_vocab(w) for w in desired_out]
    out = out[:, desired_out_idx]
    decoded = out.argmax(dim = 1)
    acc = [int(desired_out[i]) == target for i, target in zip(decoded, targets)]

    acc = sum(acc)/len(targets)
    return acc