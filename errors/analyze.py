# PROTOTYPE FILE

from datasets import load_dataset

TRAIN_FILE = "../data_utils/ds_addmult_mod10/data-addmult-231019.csv"
ERROR_FILE = "/sailhome/ananjan/grokking-330/structural-grokking-330/errors/dsmult_lm_e4_d4.txt"

def get_struct(ex):
    # Gets a representation for the structure corresponding to any example
    out_ex = ""
    for character in ex:
        if (character in "()*+"):
            out_ex += character
        else:
            out_ex += "a"
    return out_ex

def load_train(file_path):
    # Quick and dirty fix, not changing right now
    # Load training data for AddMult dataset
    # Fix loader code
    dataset = load_dataset("csv", data_files=file_path, split="all")
    # max height 4
    dataset = dataset.filter(lambda example: example['height'] <= 4)
    # max width 80
    dataset = dataset.filter(lambda example: example['width'] <= 80)

    train_testval = dataset.train_test_split(test_size=0.2, shuffle=False)
    train = train_testval['train']
    train_strings = []
    for ex in train:
        train_strings.append(ex['example'])
    return train_strings

def load_errors(file):
    # Load file of errors dumped during testing
    # Can be dumped by using --dump_errs --dump_file PATH/TO/FILE while running train_transformers.py in eval mode
    error_strings = []
    with open(file, 'r') as f:
        for line in f:
            error_strings.append(line.strip())
    return error_strings

if __name__ == '__main__':
    # Compares the distribution of structures in the training files and error examples
    # load_errors docstring contains details about creating error file

    train_strings = load_train(TRAIN_FILE)
    error_strings = load_errors(ERROR_FILE)

    train_structs = {}
    for _ in train_strings:
        struct = get_struct(_)
        if struct not in train_structs:
            train_structs[struct] = 0
        train_structs[struct] += 1

    not_in_train = 0
    for _ in error_strings:
        if (get_struct(_) not in train_structs):
            not_in_train += 1
    
    error_structs = {}
    for _ in error_strings:
        struct = get_struct(_)
        if struct not in error_structs:
            if struct in train_structs:
                error_structs[struct] = train_structs[struct]
            else:
                error_structs[struct] = 0
    
    error_struct_num = 0
    for _ in error_structs:
        error_struct_num += error_structs[_]
    
    not_in_train_tot = 0
    for _ in error_strings:
        if (_ not in train_strings):
            not_in_train_tot += 1

    # Just a bunch of statistics for quick validation, can ignore
    # Number of structures in error examples not present in train set
    print(not_in_train/len(error_strings))
    # Number of error examples not present in train set
    print(not_in_train_tot/len(error_strings))
    # Average number of examples per structure in train set
    print(len(train_strings)/len(train_structs))

