from datasets import load_dataset

def get_struct(ex):
    out_ex = ""
    for _ in ex:
        if (_ in "()*+"):
            out_ex += _
        else:
            out_ex += "a"
    return out_ex

def load_train(file_path):
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
    error_strings = []
    with open(file, 'r') as f:
        for line in f:
            error_strings.append(line.strip())
    return error_strings

if __name__ == '__main__':
    train_strings = load_train("../data_utils/ds_addmult_mod10/data-addmult-231019.csv")
    error_strings = load_errors('/sailhome/ananjan/grokking-330/structural-grokking-330/errors/dsmult_lm_e4_d4.txt')

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

    print(not_in_train/len(error_strings))
    print(not_in_train_tot/len(error_strings))
    print(len(train_strings)/len(train_structs))
    print(error_struct_num/len(error_structs))

