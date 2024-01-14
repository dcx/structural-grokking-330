from dataclasses import dataclass, asdict

import torch
import pandas as pd
import random
from sklearn.model_selection import train_test_split

pad_token_id = 30
sep_token_id = 31

def prepare_single_example(y, pad_token_id=pad_token_id):
    """
    Prepares and pads a single example. 
    """
    # Convert (x, y) to tensors
    y_tensor = torch.tensor(y, dtype=torch.long)

    # Determine the longest sequence between x and y
    longest_seq = len(y_tensor)

    # Initialize padded versions of x and y
    y_padded = torch.zeros(longest_seq, dtype=torch.long) + pad_token_id

    # Copy the original data into the padded tensors
    y_padded[:len(y)] = y_tensor

    # Add an extra dimension to simulate batch_size of 1
    y_padded = y_padded.unsqueeze(0)

    # Create the padding mask for y
    padding_mask = (y_padded == pad_token_id)

    return y_padded, padding_mask

def collate_fn(batch):
    """
    Pads batch of variable length.
    """

    longest_seq = max([max(len(x),len(y)) for x,y in batch])
    n_items = len(batch)

    x_batch = torch.zeros((n_items, longest_seq), dtype=torch.long) + pad_token_id
    y_batch = torch.zeros((n_items, longest_seq), dtype=torch.long) + pad_token_id

    for i, (x, y) in enumerate(batch):
        x_batch[i, :len(x)] = torch.tensor(x)
        y_batch[i, :len(y)] = torch.tensor(y)

    return x_batch, y_batch, (y_batch == pad_token_id)


def make_datasets(csv_file, holdout_trees_frac=0.1, train_frac=0.8, val_frac=0.2, test_max_items=2000,
                  use_cur_action=True, use_cur_action_result=True, use_next_action=False):
    """
    Given a dataset CSV, creates a triplet of datasets, in which:
    1. A random set of trees are excluded from training and val, and only used in test.
    2. The remaining trees are split into training and val.
    """
    csv_data = pd.read_csv(csv_file)

    # roll held out trees
    all_trees = csv_data.tree_sig.unique()
    random.shuffle(all_trees)
    n_holdout_trees = int(len(all_trees) * holdout_trees_frac)
    holdout_trees = all_trees[:n_holdout_trees]

    # make test dataframe: only held out trees
    test = csv_data[csv_data.tree_sig.isin(holdout_trees)]
    if test_max_items is not None:
        test = test[:test_max_items]




    # make train/val dataframe: only non-held out trees
    train_val_df = csv_data[~csv_data.tree_sig.isin(holdout_trees)]
    train, val = train_test_split(train_val_df, train_size=train_frac, test_size=val_frac)


    ds_train = PlanDataset(train, use_cur_action, use_cur_action_result, use_next_action)
    ds_val = PlanDataset(val, use_cur_action, use_cur_action_result, use_next_action)
    ds_test = PlanDataset(test, use_cur_action, use_cur_action_result, use_next_action)
    eval_dataset = EvalDataset(test, use_cur_action, use_cur_action_result, use_next_action)

    return ds_train, ds_val, ds_test, eval_dataset


def detokenize(tokens, pad_token_id=pad_token_id, sep_token_id=sep_token_id):

    char_map = {
            '(': 0, ')': 1, '[': 2, ']': 3, 'L': 4, 'I': 5, 'F': 6, 'D': 7,
            'a': 8, 'b': 9, 'c': 10, 'x': 11, 'y': 12, 'z': 13,
            '0': 14, '1': 15, '2': 16, '3': 17, '4': 18, '5': 19, '6': 20, '7': 21, '8': 22, '9': 23,
            '+': 24, '*': 25, '<': 26, '=': 27, '>': 28, ' ': 29, 
            '#': pad_token_id, # pad token 
            '|': sep_token_id, # separator token           
        }
    rev_char_map = {v: k for k, v in char_map.items()}
    
    
    return ''.join(rev_char_map.get(token, '') for token in tokens)


class PlanDataset(torch.utils.data.Dataset):
    def __init__(self, pd_dataframe, use_cur_action=True, use_cur_action_result=True, use_next_action=False):
        "note: use_next_action only works if use_cur_action is True"
        self.csv_data = pd_dataframe
        self.char_map = {
            '(': 0, ')': 1, '[': 2, ']': 3, 'L': 4, 'I': 5, 'F': 6, 'D': 7,
            'a': 8, 'b': 9, 'c': 10, 'x': 11, 'y': 12, 'z': 13,
            '0': 14, '1': 15, '2': 16, '3': 17, '4': 18, '5': 19, '6': 20, '7': 21, '8': 22, '9': 23,
            '+': 24, '*': 25, '<': 26, '=': 27, '>': 28, ' ': 29, 
            '#': pad_token_id, # pad token 
            '|': sep_token_id, # separator token           
        }
        self.use_cur_action = use_cur_action
        self.use_cur_action_result = use_cur_action_result
        self.use_next_action = use_next_action


    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        row = self.csv_data.iloc[idx]
        x, y = self.get_curr_and_next_state(row)
        return x, y

    def tokenize(self, s):
        a = [self.char_map[c] for c in s]
        return a

    def detokenize(self, tokens):
        return ''.join(self.rev_char_map.get(token, '') for token in tokens if token not in [pad_token_id, sep_token_id])
    

    def get_curr_and_next_state(self, row, predicted=None):
        if predicted:
            cur_state = self.tokenize(predicted)
        else:
            cur_state = self.tokenize(row.cur_state)
            
        next_state = self.tokenize(row.next_state)
        (x,y) = cur_state, next_state

        if self.use_cur_action:
            cur_action = self.tokenize(row.cur_action_aligned)
            if self.use_next_action:
                next_action = self.tokenize(row.next_action_aligned)
            else:
                next_action = self.tokenize(' ')*len(cur_action)
            x += [sep_token_id] + cur_action
            y += [sep_token_id] + next_action

        if self.use_cur_action_result:
            cur_action_res = self.tokenize(row.cur_action_res)
            next_action_res = self.tokenize(' ') # we don't know this
            x += [sep_token_id] + cur_action_res
            y += [sep_token_id] + next_action_res

        return x, y


        
@dataclass
class DebugData:
    final_answer: int 
    current_operator: str
    current_step: int
    final_step: int
    current_depth: int
    is_done: bool
    next_action_aligned: any

class EvalDataset(PlanDataset):
     
    def __init__(self, pd_dataframe, use_cur_action=True, use_cur_action_result=True, use_next_action=False):
        """
        Inherit from PlanDataset, but use different __getitem__ method.
        """
        super().__init__(pd_dataframe, use_cur_action, use_cur_action_result, use_next_action)


    def __getitem__(self, idx):

        self.csv_data.reset_index(drop=True, inplace=True)

        row = self.csv_data.iloc[idx]
        # final state
        final_step_num = row.final_step
        sequence_id = row.new_sequence

        # Retrieve the row for the final step of the current idx
        final_step_row = self.csv_data[(self.csv_data['idx'] == row.idx) & (self.csv_data['step'] == final_step_num)].iloc[0]

        if final_step_row.is_done != 1:
            raise ValueError(f"Data inconsistency: 'is_done' is not 1 at the final step for sequence {sequence_id}")

        x, y = self.get_curr_and_next_state(row)

        debug_data = DebugData(
            final_answer = int(final_step_row.next_state),
            current_operator = row.cur_action_type,
            current_step = row.step,
            final_step = final_step_num,
            current_depth = row.height,
            is_done = final_step_row.is_done,
            next_action_aligned = row.next_action_aligned
        )
        
        return x, y, asdict(debug_data)
        
    def get_indices_of_new_sequences(self):
        self.csv_data.reset_index(drop=True, inplace=True)

        # Find all indices where step is 0, reset indices.
        new_sequence_indices = self.csv_data[self.csv_data['step'] == 0].index.tolist()
        return new_sequence_indices