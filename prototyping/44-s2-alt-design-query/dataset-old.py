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

    def __init__(self, n_examples, len_from, len_to, n_steps, ca_rule_no):
        self.n_examples = n_examples
        self.len_from = len_from
        self.len_to = len_to

        # make the ruleset
        self.rules_cache = {ca_rule_no: self.make_ca_rules(ca_rule_no)}
        ca_rule_seq = [ca_rule_no] * n_steps
        ca_rule_token_map = {
            30: 4,
        }

        # generate the dataset
        self.data = []
        for _ in range(n_examples):
            # generate a random binary string
            length = random.randint(len_from, len_to)
            ca_start = ''.join(random.choice('01') for _ in range(length))
            ca_sequence = self.automata_tick(ca_start, ca_rule_seq, self.rules_cache, n_steps, include_substeps=True)

            ca_sequence = [ca_start] + ca_sequence
            cur_row = []
            for i,ca_rule in enumerate(ca_rule_seq):
                cur_row.append(self.tokenize(ca_sequence[i]) + \
                               [ca_rule_token_map[ca_rule]])
            cur_row.append(self.tokenize(ca_sequence[-1]) + \
                           [ca_rule_token_map[30]]) # dummy final rule - for now, map to pad token for simplicty

            self.data.append(cur_row)

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

    def automata_tick(self, ca_input, ca_rule_seq, rules_cache, n_steps, include_substeps=False):
        """
        Step a cellular automata forward according to some ruleset.

        - ca_input is a string of 0s and 1s.
        - ca_rule_seq contains the Wolfram code for the CA ruleset to apply at
          each step. If it is an int, it is the same ruleset for each step.
          If it is a list of ints, each int is the ruleset for that step.
        - rules_cache is a dict of rulesets, keyed by ruleset number.
          We assume every rule on ca_rule_seq is in rules_cache.
        - n_steps is an int, the number of steps to tick the automata forward.
        - include_substeps is a bool, whether to return the intermediate steps.
          if True, returns a list of strings, each string representing a step.
          if False, returns a single string, the final step.  
        """
        ca_string = ca_input
        ca_sequence = []
        for ca_rule_no in ca_rule_seq:
            rules = rules_cache[ca_rule_no]

            new_string = ''
            for i in range(len(ca_string)-2):
                int_val = int(ca_string[i:i+3], 2)
                new_string += str(rules[int_val])
            new_string = '0' + new_string + '0'
            ca_string = new_string
            ca_sequence.append(ca_string)

        if include_substeps:
            return ca_sequence
        else:
            return ca_string

    def tokenize(self, s):
        a = [int(c) for c in s]
        return a


def make_datasets(n_train, n_val, n_steps):
    ds_train = ToyDataset(n_train, 8, 16, n_steps=n_steps)
    ds_val = ToyDataset(n_val, 8, 16, n_steps=n_steps)

    return ds_train, ds_val


def make_collate_fn(pad_token_id):

    def collate_fn(batch):
        """
        Pads batch of variable length.
        """
        lengths = [len(x[0]) for x in batch]
        longest_seq = max(lengths)
        shortest_seq = min(lengths)
        n_items = len(batch)
        n_steps = len(batch[0])

        batch_tensor = torch.zeros((n_items, n_steps, longest_seq), dtype=torch.long) + pad_token_id

        for i, b in enumerate(batch):
            for j in range(n_steps):
                batch_tensor[i, j, :len(b[j])] = torch.tensor(b[j])

        return batch_tensor, (batch_tensor == pad_token_id)
    
    return collate_fn


if __name__ == '__main__':
    ds = ToyDataset(n_examples=10, len_from=8, len_to=16, n_steps=9, ca_rule_no=30)
    print(ds.data)


