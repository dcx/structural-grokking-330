from nltk import Tree
import torch
from data_utils.text_helpers import flatten
from tqdm import tqdm
from util import set_seed, convert_tree_to_tuple
from eval_utils.eval_pushdown_model import get_parsing_accuracy
import pdb
from ..train_transformers import get_base_transformer_lm

def get_right_parse(sentence):
    if (len(sentence) == 2):
        return (sentence[0], sentence[1])
    elif (len(sentence) == 1):
        return (sentence[0])
    return (sentence[0], get_right_parse(sentence[1:]))

def get_right_parseval(split, data_folder_given=None):
    """Callback function on BLIMP for pushdown lm training."""
    if data_folder_given:
        folder_dir = data_folder_given
    else:
        folder_dir = "bllip-lg-depth"
    
    with open("{}/{}.txt".format(folder_dir, split)) as f:
        data = [Tree.fromstring(l.strip()) for l in f.readlines()]

    examples = [(d, flatten(d, add_eos=False)) for d in data]

    # get parsevals
    actual = 0
    gold_parses = []
    predicted_parses = []
    for d, sentence in tqdm(examples):
        if (len(sentence.split()) > 30):
            continue
        actual += 1
        predicted_parse = get_right_parse(sentence.split(" "))
        predicted_parses.append(predicted_parse)
        gold_parses.append(convert_tree_to_tuple(d))
    
    parsing_acc = get_parsing_accuracy(predicted_parses, gold_parses)

if __name__=='__main__':
    get_right_parseval('val', '/u/scr/smurty/pushdown-lm/data_utils/bllip-lg-depth')
    get_right_parseval('test', '/u/scr/smurty/pushdown-lm/data_utils/bllip-lg-depth')