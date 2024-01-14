import torch
import pandas as pd
import random
from sklearn.model_selection import train_test_split

pad_token_id = 30 # TODO: Parameterize
sep_token_id = 31 # TODO: Parameterize

def collate_fn(batch):
    """
    Pads batch of variable length.
    """
    longest_seq = max([len(x) for _,x,_ in batch]) # o,x,y same length
    n_items = len(batch)

    o_batch = torch.zeros((n_items, longest_seq), dtype=torch.long) + pad_token_id
    x_batch = torch.zeros((n_items, longest_seq), dtype=torch.long) + pad_token_id
    y_batch = torch.zeros((n_items, longest_seq), dtype=torch.long) + pad_token_id

    for i, (o, x, y) in enumerate(batch):
        o_batch[i, :len(o)] = torch.tensor(o)
        x_batch[i, :len(x)] = torch.tensor(x)
        y_batch[i, :len(y)] = torch.tensor(y)

    return o_batch, x_batch, y_batch, (y_batch == pad_token_id)


def make_datasets(csv_file, holdout_trees_frac=0.1, train_frac=0.8, val_frac=0.2, train_max_items=None, val_max_items=2500, test_max_items=2500,
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

    val_size = val_frac if val_max_items is None else val_max_items
    train_size = 1 - val_frac if val_max_items is None else len(train_val_df) - val_max_items
    train, val = train_test_split(train_val_df, train_size=train_size, test_size=val_size)

    ds_train = PlanDataset(train, use_cur_action, use_cur_action_result, use_next_action)
    ds_val = PlanDataset(val, use_cur_action, use_cur_action_result, use_next_action)
    ds_test = PlanDataset(test, use_cur_action, use_cur_action_result, use_next_action)

    return ds_train, ds_val, ds_test




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
        self.char_unmap = {v:k for k,v in self.char_map.items()}

        self.use_cur_action = use_cur_action
        self.use_cur_action_result = use_cur_action_result
        self.use_next_action = use_next_action


    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        row = self.csv_data.iloc[idx]
        x_raw  = self.tokenize(row.example)
        cur_state = self.tokenize(row.cur_state)
        next_state = self.tokenize(row.next_state)
        (o,x,y) = x_raw, cur_state, next_state

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

        return o, x, y

    def tokenize(self, s):
        a = [self.char_map[c] for c in s]
        return a

