import pickle
from nltk import Tree
from tqdm import tqdm

def tree_to_sentence(tree):
    subwords = tree.leaves()
    curr_word = ""
    sentence = []
    for word in subwords:
        if (word[0] == 'Ġ'):
            sentence.append(curr_word)
            curr_word = word[1:]
        else:
            curr_word += word
    sentence.append(curr_word)

    return sentence

if __name__ == '__main__':
    with open("blimp.pkl", "rb") as f:
        blimp_data = [Tree.fromstring(t) for t in pickle.load(f)]
        
    with open('/u/scr/smurty/pushdown-lm/data_utils/bllip-lg-depth/train.txt', 'r') as f:
        train_data = [tree_to_sentence(Tree.fromstring(t)) for t in tqdm(f)]

    words = set()
    for _ in tqdm(train_data):
        for word in _:
            words.add(word)

    num_pairs = len(blimp_data)//2
    filtered_blimp_good = []
    filtered_blimp_bad = []
    for num in range(num_pairs):
        good_words = tree_to_sentence(blimp_data[num])
        bad_words = tree_to_sentence(blimp_data[num + num_pairs])

        skip = False
        for _ in good_words:
            if _ not in words:
                skip = True
                break
        
        if skip:
            continue

        for _ in bad_words:
            if _ not in words:
                skip = True
                break

        if skip:
            continue

        filtered_blimp_good.append(str(blimp_data[num]))
        filtered_blimp_bad.append(str(blimp_data[num + num_pairs]))


    pickle.dump(filtered_blimp_good + filtered_blimp_bad, open("blimp_filtered.pkl", "wb"))
