import torch
import os, random
from datasets import load_dataset
import tokenizers

dataset_path = '../data/test-nostep-am-500k-d46.csv'
# dataset_path = '/dev/shm/lichess_100mb.csv'


# define data format
supported_chars = '0123456789+*()[]abcxyzLFI<=_'
itos = {i:c for i, c in enumerate(supported_chars)}
stoi = {c:i for i, c in enumerate(supported_chars)}
pad_token_id = stoi['_']

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
# tokenizer.enable_padding(pad_id=pad_token_id) # do not pad when tokenizing at the load step

def tokenize_csv_rows(rows):
    # cur_state_tight = []
    # for cur_state in rows['cur_state']:
    #     cur_state_tight.append(cur_state.replace(" ", ""))
    # tokenized = tokenizer.encode_batch(cur_state_tight)
    # #tokenized = tokenizer.encode_batch(rows['example'])
    # tok_ids = [t.ids for t in tokenized]
    # lengths = [len(t.ids) for t in tokenized]
    # answers = rows['ans_mod10']

    tokenized = tokenizer.encode_batch(rows['example'])
    tok_ids = [t.ids for t in tokenized]

    # next_state_tight = []
    # for next_state in rows['next_state']:
    #     next_state_tight.append(next_state.replace(" ", ""))
    # tok_next = tokenizer.encode_batch(next_state_tight)
    # tok_ids_next = [t.ids for t in tok_next]



    # next_action_tight = []
    # for nat in rows['cur_action_tight']:
    #     if nat is None:
    #         next_action_tight.append("_")
    #     else:
    #         next_action_tight.append(nat.replace(" ", "_"))
    # subitems_tok = tokenizer.encode_batch(next_action_tight)
    # tok_ids_subitem = [t.ids for t in subitems_tok]
    # lengths_subitem = [len(t.ids) for t in subitems_tok]

    # next_action_res = []
    # for nar in rows['cur_action_res']:
    #     if nar is None:
    #         next_action_res.append(0)
    #     else:
    #         next_action_res.append(int(nar.replace(" ", "0")))

    #ans_sublabels = tokenizer.encode_batch(rows['ans_sublabels'])
    #ans_sublabels = [t.ids for t in ans_sublabels]
    return {
        'heights': rows['height'],
        'input_ids': tok_ids,
        'answers': rows['answer'],
    }
        # 'ans_sublabels': ans_sublabels}


def detokenize(id_tensor):
    return tokenizer.decode_batch(id_tensor.tolist())

# height,width,example,answer,ans_mod10,tree_sig,ans_sublabels
# 2,5,(*14),4,4,(odd),____4
# 5,21,(+(+33)(+5(*0(+47)))),11,1,(o(odd)(od(od(odd)))),______6__________1051



# load raw CSV
def make_datasets(n_train, n_val, num_proc=8, random_seed=2357):
    """
    n_train: number of training examples
    n_val: number of validation examples
    """

    dataset = load_dataset("csv", data_files=dataset_path, split=f"train[:2500000]")

    dataset = dataset.map(tokenize_csv_rows, batched=True, num_proc=num_proc, load_from_cache_file=True)

    # filter: all step=0 (zeroth step is invalid for the way we're (ab)using the stepwise dataset)
    #dataset = dataset.filter(lambda x: int(x['step']) > 0)

    dataset = dataset.select_columns(["heights", "input_ids", "answers"])
    dataset.set_format(type="torch", columns=["heights", "input_ids", "answers"])



    dataset = dataset.train_test_split(
        train_size=n_train, test_size=n_val, 
        seed=random_seed, shuffle=True
    )

    ds_train = dataset["train"]
    ds_val   = dataset["test"]

    return ds_train, ds_val



def collate_fn(batch):

    #min_seq = 64 # hack so model.py's "trim n_rand_encs to be a multiple of bs" doesn't crash
    #max_seq = 512

    lengths = [len(x['input_ids']) for x in batch]
    # lengths_next_ids = [len(x['next_ids']) for x in batch]
    # lengths_subitem = [len(x['next_action_tight']) for x in batch]
    
    longest_seq = max(lengths) # longest supported sequence
    n_items = len(batch)

    batch_x = torch.zeros((n_items, longest_seq), dtype=torch.int8) + pad_token_id
    #batch_xnext = torch.zeros((n_items, longest_seq), dtype=torch.int8) + pad_token_id
    batch_y = torch.stack([x['answers'] for x in batch])
    #batch_xsi = torch.zeros((n_items, longest_seq), dtype=torch.int8) + pad_token_id
    #batch_ysi = torch.stack([x['next_action_res'] for x in batch])

    for i, b in enumerate(batch):
        batch_x[i, :lengths[i]] = b['input_ids']
        # batch_xnext[i, :lengths_next_ids[i]] = b['next_ids']
        # batch_xsi[i, :lengths_subitem[i]] = b['next_action_tight']
        #batch_sl[i,:lengths[i]] = b['ans_sublabels']

    batch_heights = torch.stack([x['heights'] for x in batch])

    return batch_heights, batch_x, batch_y # , batch_xnext, batch_xsi, batch_ysi



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

