import torch
import pandas as pd

pad_token_id = 30

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


class PlanDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file):
        self.csv_data = pd.read_csv(csv_file)
        self.char_map = {
            '(': 0, ')': 1, '[': 2, ']': 3, 'L': 4, 'I': 5, 'F': 6, 'D': 7,
            'a': 8, 'b': 9, 'c': 10, 'x': 11, 'y': 12, 'z': 13,
            '0': 14, '1': 15, '2': 16, '3': 17, '4': 18, '5': 19, '6': 20, '7': 21, '8': 22, '9': 23,
            '+': 24, '*': 25, '<': 26, '=': 27, '>': 28, ' ': 29, 
            '#': pad_token_id, # pad token            
        }

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        row = self.csv_data.iloc[idx]
        cur_state = self.tokenize(row.cur_state)
        next_state = self.tokenize(row.next_state)

        return cur_state, next_state

    def tokenize(self, s):
        return [self.char_map[c] for c in s]

    def detokenize(self, t):
        return ''.join([list(self.char_map.keys())[i] for i in t.tolist()])