from vocabulary import WordVocabulary
from datasets import Dataset as HFDataset
import sequence
import pickle
import random
from collections import Counter
import numpy as np
from tqdm import tqdm
from nltk import Tree
import json
import pdb
from transformers import GPT2Tokenizer


def flatten(parse, add_eos, clean=False):
    def helper(p):
        if type(p) == str:
            return p
        else:
            return " ".join(helper(x) for x in p)

    if type(parse) == Tree:
        words = " ".join(parse.leaves())
    else:
        words = helper(parse)

    if clean:
        words = words.split(" ")
        cleaned_words = []
        curr_word = ""
        for idx in range(len(words)):
            if words[idx][0] == 'Ġ':
                if idx != 0:
                    cleaned_words.append(curr_word)
                curr_word = words[idx][1:]
            else:
                curr_word += words[idx]
        cleaned_words.append(curr_word)
        words = " ".join(cleaned_words)
    
    if add_eos:
        return "{} <eos>".format(words)
    else:
        return words
    
class ParserPipeline:
    def __init__(self):
        self.gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    def post_process_op(self, tree):
        """
        Input: nltk tree where leaf nodes are strings
        Output: nltk tree but use gpt_tokenizer to tokenize leaf nodes and create a subtree corresponding to it.
        """

        def fix(t, is_first, label=None):
            if type(t) == str:
                tokenized = self.gpt_tokenizer.tokenize(
                    t, add_prefix_space=not is_first
                )
                if len(tokenized) == 1:
                    return Tree(label, [tokenized[0]])
                else:
                    return Tree(label, [Tree(label, [tok]) for tok in tokenized])
            elif len(t) == 1:
                return fix(t[0], is_first=is_first, label=t.label())
            else:
                return Tree(
                    t.label(),
                    [fix(c, is_first=is_first and idx == 0) for idx, c in enumerate(t)],
                )

        fixed_tree = fix(tree, is_first=True)
        fixed_tree.chomsky_normal_form()
        return fixed_tree

    def process(self, parse):
        ptree = Tree.fromstring(parse)
        ptree.chomsky_normal_form()
        return self.post_process_op(ptree)
        # return ptree

    def __call__(self, parse):
        return self.process(parse)


def get_elems(list_of_list):
    ### flatten out a list of lists
    ret = []
    for l in list_of_list:
        if type(l) == list:
            ret.extend(get_elems(l))
        else:
            ret.append(l)
    return ret


def binarize_tree(parse):
    if type(parse) == str:
        return parse
    else:
        if len(parse) == 1:
            return binarize_tree(parse[0])
        else:
            return (binarize_tree(parse[0]), binarize_tree(parse[1:]))

def stringify_tree(parse):
    if type(parse) == str:
        return parse
    else:
        if len(parse) == 1:
            return stringify_tree(parse[0])
        else:
            return "("+stringify_tree(parse[0])+") ("+stringify_tree(parse[1:])+")"
        
def tree_to_parse_decisions(parse, start, parse_dict):
    # Builds a dict containing index of gold split for all gold subtrees
    # Accumulates results in parse_dict
    if (len(parse.leaves()) <= 2):
        # There is only one possible split for phrases of length 2
        return len(parse.leaves())
    
    # We know that subwords are never split between constituents. 
    # If all tokens after the first one in the current span do not start with 'G.', it should not be split further (single word)
    single_word = True
    for idx, word in enumerate(parse.leaves()):
        if (idx == 0):
            continue
        if (word[0] == 'Ġ'):
            single_word = False
            break
    
    if single_word:
        return len(parse.leaves())
    
    s1 = len(parse[0].leaves())
    s2 = len(parse[1].leaves()) 

    tree_to_parse_decisions(parse[0], start, parse_dict)
    tree_to_parse_decisions(parse[1], start + s1, parse_dict)

    parse_dict[str(start) + ' ' + str(start + s1 + s2)] = start + s1

    return s1 + s2

def reformat_tree(tree, in_vocab, is_first):
    # "retokenizes" tree according to given vocabulary
    
    single_word = True
    for idx, word in enumerate(tree.leaves()):
        if (idx == 0):
            continue
        if (word[0] == 'Ġ'):
            single_word = False
            break
    
    if single_word:
        # this needs processing, subwords 
        recreated = "".join(tree.leaves())
        if not is_first:
            # remove first space
            recreated = recreated[1:]
        tokens = in_vocab.tokenize(recreated)
        if len(tokens) == 1:
            return Tree(tree.label(), ['Ġ' + recreated])
        else:
            return Tree(tree.label(), 
            [Tree(tree.label(), ['Ġ' + tokens[0][1:]])] + 
            [Tree(tree.label(), [_]) for _ in tokens[1:]])

    return Tree(
        tree.label(),
        [reformat_tree(t, in_vocab, is_first = is_first and idx == 0) for idx, t in enumerate(tree)]
    )

def build_random_tree(tree_nodes):
    if len(tree_nodes) == 1:
        return Tree('X', tree_nodes)
    
    single_word = True
    for idx, word in enumerate(tree_nodes):
        if (idx == 0):
            continue
        if (word[0] == 'Ġ'):
            single_word = False
            break
    
    if single_word:
        return Tree('X', [Tree('X', [tok]) for tok in tree_nodes])

    split_points = []
    for cand in range(0, len(tree_nodes) - 1):
        if tree_nodes[cand+1][0] == 'Ġ':
            split_points.append(cand+1)

    split_point = split_points[random.randint(0,len(split_points)-1)]

    return Tree('X', [build_random_tree(tree_nodes[:split_point]), build_random_tree(tree_nodes[split_point:])])

def reformat_sentence(sentence, in_vocab):
    # this is accompanying string
    ret_words = []
    for word in in_vocab.tokenize(sentence):
        if word[0] == '▁':
            ret_words.append('Ġ' + word[1:])
        else:
            ret_words.append(word)
    return " ".join(ret_words)

def build_datasets_pushdown(
    data_regime="normal",
    only_vocab=False,
    data_file_given=None,
    data_ratio = 1.0,
    in_vocab = None,
    hf = False,
    randomize = False
):
    def read_data(splits):
        in_sentences = []
        in_sentences_tok = []
        parses = []
        index_map = {split: [] for split in splits}
        for split in splits:
            split_file = split

            if not data_file_given:
                data_file = "bllip-lg-depth"
            else:
                data_file = data_file_given

            with open(
                "{}/{}.txt".format(
                    data_file,
                    split_file,
                ),
                "r",
            ) as reader:
                print("Reading trees for {}".format(split_file))
                if randomize:
                    data = [
                        build_random_tree(Tree.fromstring(l.strip()).leaves()) for l in tqdm(reader.readlines()[:int(data_ratio * 1755715)])
                    ]
                else:
                    data = [
                        Tree.fromstring(l.strip()) for l in tqdm(reader.readlines()[:int(data_ratio * 1755715)])
                    ]

            for sent in tqdm(data):
                if hf:
                    sent = reformat_tree(sent, in_vocab, True)
                index_map[split].append(len(in_sentences))
                if hf:
                    in_sentences.append(reformat_sentence(flatten(sent, add_eos=False, clean=True), in_vocab))
                    in_sentences_tok.append(flatten(sent, add_eos=False, clean=True))
                else:
                    in_sentences.append(flatten(sent, add_eos=False))
                if not isinstance(sent, Tree):
                    parses.append(binarize_tree(sent))
                else:
                    parses.append(sent)
        return in_sentences, in_sentences_tok, parses, index_map

    def get_subset(elem_list, idx_list):
        return [elem_list[idx] for idx in idx_list]

    splits = ["train", "val", "test"]
    ### NOTE: if math, sent and parses are the same.
    in_sentences, in_sentences_tok, parses, index_map = read_data(splits)
    print("num examples: {}".format(len(in_sentences)))

    # in_vocab = WordVocabulary(in_sentences, split_punctuation=False) # remove this
    # # pickle.dump(in_vocab, open('/afs/cs.stanford.edu/u/ananjan/grokking-330/structural-grokking-330/data_utils/blimp_vocab.pkl', 'wb'))
    # if only_vocab:
    #     return in_vocab

    dataset = {}
    for split in splits:
        print("Processing {} data".format(split))
        if data_regime == "small" and split == "train":
            max_sz = 10000
        elif data_regime == "tiny":
            max_sz = 1000
        else:
            max_sz = len(index_map[split])
        if len(index_map[split]) > max_sz:
            index_map[split] = random.sample(index_map[split], k=max_sz)
        in_subset = get_subset(in_sentences, index_map[split])
        in_parses = get_subset(parses, index_map[split])
        if hf:
            in_subset_tok = get_subset(in_sentences_tok, index_map[split])
            in_subset_tokenized = [in_vocab(s)["input_ids"][1:] for s in in_subset_tok] # remove sos
        else:
            in_subset_tokenized = [in_vocab(s) for s in in_subset]
        in_lens = [len(s) for s in in_subset_tokenized]

        parse_dicts = []
        for p in tqdm(in_parses):
            parse_dict = {}
            _ = tree_to_parse_decisions(p, 0, parse_dict)
            parse_dicts.append(json.dumps(parse_dict))
        data = {
            "in": in_subset_tokenized,
            "in_len": in_lens,
            "idxs": index_map[split],
            "string": in_subset,
            "parses": parse_dicts
        }

        dataset_curr = HFDataset.from_dict(data)
        dataset[split] = dataset_curr

    return dataset, in_vocab, in_sentences

def build_datasets_ptb(
    in_vocab,
    data_file_given=None,
    data_ratio = 1.0
):
    def read_data(splits):
        in_sentences = []
        parses = []
        index_map = {split: [] for split in splits}
        for split in splits:
            split_file = split

            if not data_file_given:
                data_file = "bllip-lg-depth"
            else:
                data_file = data_file_given

            with open(
                "{}/en_ptb3-revised_{}.mrg".format(
                    data_file,
                    split_file
                ),
                "r",
            ) as reader:
                print("Reading trees for {}".format(split_file))
                pipeline = ParserPipeline()
                data = [
                    pipeline(l.strip()) for l in tqdm(reader.readlines())
                ]
            for sent in tqdm(data):
                index_map[split].append(len(in_sentences))
                in_sentences.append(flatten(sent, add_eos=False))
                if not isinstance(sent, Tree):
                    parses.append(binarize_tree(sent))
                else:
                    parses.append(sent)
        return in_sentences, parses, index_map

    def get_subset(elem_list, idx_list):
        return [elem_list[idx] for idx in idx_list]

    splits = ["train", "val", "test"]
    ### NOTE: if math, sent and parses are the same.
    in_sentences, parses, index_map = read_data(splits)
    print("num examples: {}".format(len(in_sentences)))

    # in_vocab = WordVocabulary(in_sentences, split_punctuation=False)
    # pickle.dump(in_vocab, open('/afs/cs.stanford.edu/u/ananjan/grokking-330/structural-grokking-330/data_utils/blimp_vocab.pkl', 'wb'))

    dataset = {}
    for split in splits:
        print("Processing {} data".format(split))
        max_sz = len(index_map[split])
        if len(index_map[split]) > max_sz:
            index_map[split] = random.sample(index_map[split], k=max_sz)
        in_subset = get_subset(in_sentences, index_map[split])
        in_parses = get_subset(parses, index_map[split])
        in_subset_tokenized = []
        in_parses_subset = []
        indices = []
        in_subset_fin = []
        exceptions = 0
        for idx, s in enumerate(in_subset):
            try:
                in_subset_tokenized.append(in_vocab(s))
                in_parses_subset.append(in_parses[idx])
                indices.append(index_map[split][idx])
                in_subset_fin.append(in_subset[idx])
            except:
                exceptions += 1
        print(f'Examples excluded = {exceptions}')
        in_lens = [len(s) for s in in_subset_tokenized]

        parse_dicts = []
        for p in tqdm(in_parses_subset):
            parse_dict = {}
            _ = tree_to_parse_decisions(p, 0, parse_dict)
            parse_dicts.append(json.dumps(parse_dict))

        data = {
            "in": in_subset_tokenized,
            "in_len": in_lens,
            "idxs": index_map[split],
            "string": in_subset,
            "parses": parse_dicts
        }

        dataset_curr = HFDataset.from_dict(data)
        dataset[split] = dataset_curr

    return dataset, in_sentences