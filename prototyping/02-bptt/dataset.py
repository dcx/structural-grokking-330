import torch
import random
import string

class ToyDataset(torch.utils.data.Dataset):
    """
    Toy dataset for exploring backpropagation through time (BPTT) using
    1-D cellular automata.

    Dynamically generates a dataset of random binary strings. 
    The model's task at each timestep is to predict the next generation
    of the string, given the current generation.
    """

    def __init__(self, n_examples, len_from, len_to, n_steps=1, ca_ruleset=30):
        self.n_examples = n_examples
        self.len_from = len_from
        self.len_to = len_to
        self.rules = self.make_ca_rules(ca_ruleset)

        # generate the dataset
        self.data = []
        for _ in range(n_examples):
            length = random.randint(len_from, len_to)
            ca_in = ''.join(random.choice('01') for _ in range(length))
            ca_out = self.automata_step(ca_in, self.rules, n_steps=n_steps)
            ca_in = self.tokenize(ca_in)
            ca_out = self.tokenize(ca_out)
            self.data.append((ca_in, ca_out))

    def __len__(self):
        return self.n_examples

    def __getitem__(self, idx):
        return self.data[idx]

    def make_ca_rules(self, ruleset_no):
        # unpack automata ruleset into a dict
        rules = {}
        for i in range(8):
            rules[i] = ruleset_no % 2
            ruleset_no = ruleset_no // 2
            #print(f'{i}: {rules[i]}')
        return rules

    def automata_step(self, ca_string, rules, n_steps=1):
        """
        Given a string of 0s and 1s, return the next generation of the string.
        Ruleset is an int, representing the Wolfram code for the CA ruleset.
        """
        for _ in range(n_steps):
            new_string = ''
            for i in range(len(ca_string)-2):
                int_val = int(ca_string[i:i+3], 2)
                new_string += str(rules[int_val])
            new_string = '0' + new_string + '0'
            ca_string = new_string

        return ca_string

    def tokenize(self, s):
        a = [int(c) for c in s]
        return a


def make_datasets(n_train, n_val, n_steps):
    ds_train = ToyDataset(n_train, 8, 16, n_steps=n_steps)
    ds_val = ToyDataset(n_val, 8, 16, n_steps=n_steps)

    return ds_train, ds_val


pad_token_id = 2
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


if __name__ == '__main__':
    ds = ToyDataset(10, 8, 16)
    print(ds.data)


