import torch
import os, random
from datasets import load_dataset
import tokenizers

dataset_path = '../data/chess/lichess_6gb.csv'
# dataset_path = '/dev/shm/lichess_100mb.csv'


# define data format
itos = {
    0: ' ', 1: '#', 2: '+', 3: '-', 4: '.', 5: '0', 6: '1', 7: '2', 
    8: '3', 9: '4', 10: '5', 11: '6', 12: '7', 13: '8', 14: '9', 
    15: ';', 16: '=', 17: 'B', 18: 'K', 19: 'N', 20: 'O', 21: 'Q', 
	22: 'R', 23: 'a', 24: 'b', 25: 'c', 26: 'd', 27: 'e', 28: 'f',
	29: 'g', 30: 'h', 31: 'x',
}

stoi = {
	' ': 0, '#': 1, '+': 2, '-': 3, '.': 4, '0': 5, '1': 6,
	'2': 7, '3': 8, '4': 9, '5': 10, '6': 11, '7': 12, '8': 13,
	'9': 14, ';': 15, '=': 16, 'B': 17, 'K': 18, 'N': 19, 'O': 20,
	'Q': 21, 'R': 22, 'a': 23, 'b': 24, 'c': 25, 'd': 26, 'e': 27,
	'f': 28, 'g': 29, 'h': 30, 'x': 31,
}

pad_token_id = stoi[';']

# # add chess positions as separate tokens
# for letter in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']:
#     for num in range(1, 9):
#         position = letter + str(num)
#         itos[len(itos)] = position
#         stoi[position] = len(stoi)

# from tokenizers import normalizers
# from tokenizers.normalizers import NFD, StripAccents
# normalizer = normalizers.Sequence([NFD(), StripAccents()])


# setup tokenizer
tokenizer_model = tokenizers.models.WordLevel(stoi)
tokenizer = tokenizers.Tokenizer(tokenizer_model)

# tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Sequence([
#     tokenizers.pre_tokenizers.Split(tokenizers.Regex("[a-h][1-8]"), behavior="isolated", invert=True),
#     tokenizers.pre_tokenizers.Whitespace(),
#     tokenizers.pre_tokenizers.Split('x', behavior="isolated"),
#     tokenizers.pre_tokenizers.Split(tokenizers.Regex("[A-Z]"), behavior="isolated"),
#     tokenizers.pre_tokenizers.Split(tokenizers.Regex("^[0-9]"), behavior="isolated"),
# ])
# test_tok = tokenizer.pre_tokenizer.pre_tokenize_str("1.e4 g6 2.d4 Bg7 3.Nc3 c5 4.Be3 cxd4 5.Bxd4 Nf6 6.f3 Nc6 7.Be3 O-O 8.Qd2 a6 9.O-O-O b5 10.Bh6 b4 11.Nd5 Bb7 12.Bxg7 Kxg7 13.h4 h5 14.g4 a5 15.Nxf6 Kxf6 16.gxh5 Rh8 17.Qg5+ Kg7 18.hxg6 f6 19.Qg3 Ne5 20.h5 Rh6 21.Nh3 Qc7 22.Nf4 Rc8 23.Rh2 a4 24.Kb1 a3 25.Qe1 axb2 26.Kxb2 Nc4+ 27.Bxc4 Qxc4 28.Rxd7 Ra8 29.Rxe7+ Kf8 30.Rf7+ Kg8 31.Qa1 Qc3+ 32.Kb1 Qe1+ 33.Kb2 Qc3+ 34.Kb1 Qe3 35.Ne6 Qg1+ 36.Kb2 Qxh2 37.Rxb7 Qe5+ 38.Kb1 Qxe6 39.Qb2 Qe5")
# print(test_tok)
# for tok in test_tok:
#     if len(tok[0]) > 1: 
#         print(tok)

tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Split('', behavior="isolated", invert=True)
tokenizer.enable_padding(pad_id=pad_token_id) # do not pad when tokenizing at the load step

def tokenize_csv_rows(rows):
    tokenized = tokenizer.encode_batch(rows['transcript'])
    tok_ids = [t.ids for t in tokenized]
    lengths = [len(t.ids) for t in tokenized]
    return {'input_ids': tok_ids, 'lengths': lengths}

def detokenize(id_tensor):
    return tokenizer.decode_batch(id_tensor.tolist())

# s = "1.e4 g6 2.d4 Bg7 3.Nc3 c5 4.Be3 cxd4 5.Bxd4 Nf6 6.f3 Nc6 7.Be3 O-O 8.Qd2 a6 9.O-O-O b5 10.Bh6 b4 11.Nd5 Bb7 12.Bxg7 Kxg7 13.h4 h5 14.g4 a5 15.Nxf6 Kxf6 16.gxh5 Rh8 17.Qg5+ Kg7 18.hxg6 f6 19.Qg3 Ne5"

# all_chars = set(stoi.keys())
# for char in s:
#     if char not in all_chars:
#         print(char)

# a = tokenizer.encode(s)


# load raw CSV
def make_datasets(n_train, n_val, random_seed=2357):
    """
    n_train: number of training examples
    n_val: number of validation examples
    """
    dataset = load_dataset("csv", data_files=dataset_path, split=f"train[:{n_train+n_val}]")

    dataset = dataset.map(tokenize_csv_rows, batched=True, num_proc=24) # , load_from_cache_file=False)
    dataset = dataset.select_columns(["input_ids", "lengths"])
    dataset.set_format(type="torch", columns=["input_ids", "lengths"])

    dataset = dataset.train_test_split(
        train_size=n_train, test_size=n_val, 
        seed=random_seed, shuffle=True
    )

    ds_train = dataset["train"]
    ds_val   = dataset["test"]

    return ds_train, ds_val



def collate_fn(batch):

    lengths = [len(x['input_ids']) for x in batch]
    longest_seq = min(max(lengths), 512) # longest supported sequence
    n_items = len(batch)

    batch_x = torch.zeros((n_items, longest_seq), dtype=torch.int8) + pad_token_id
    batch_y = torch.zeros((n_items, longest_seq), dtype=torch.int8) + pad_token_id

    for i, b in enumerate(batch):
        batch_x[i, :lengths[i]-1] = b['input_ids'][:-1][:longest_seq] # TODO: Capture full game, don't lose a move
        batch_y[i, :lengths[i]-1] = b['input_ids'][1:][:longest_seq]

    return batch_x, batch_y



class SeqLenAwareBatchSampler(torch.utils.data.Sampler):
    def __init__(self, data, batch_size, shuffle):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
    def __len__(self):
        return (len(self.data) + self.batch_size - 1) // self.batch_size
    def __iter__(self):
        sizes = self.data['lengths']
        index_chunks = list(torch.chunk(torch.argsort(sizes), len(self)))
        if self.shuffle:
            random.shuffle(index_chunks)
        for batch in index_chunks:
            yield batch.tolist()

