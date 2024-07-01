from nltk.tree import *
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss
import pickle

def phrase_dump(phrases, t, fil):
    try:
        t.label()
    except AttributeError:
        return 
    
    if t.label() == fil:
        phrases.add(" ".join(t.leaves()))

    for child in t:
        phrase_dump(phrases, child, fil)

def get_from_file(in_file, data_ratio):
    with open(in_file, 'r') as f:
        data = [
                    Tree.fromstring(l.strip()) for l in tqdm(f.readlines()[:int(data_ratio * 1755715)])
                ]
        
    return data

def create_dump(in_file, data_ratio, fil, out_file):
    data = get_from_file(in_file, data_ratio)

    phrases = set()
    for tree in tqdm(data):
        phrase_dump(phrases, tree, fil)

    phrases = list(phrases)
    print(len(phrases))

    with open(out_file, 'w') as f:
        for _ in phrases:
            f.write(f'{_}\n')

def get_embs(in_file, out_file):
    text = []
    with open(in_file, 'r') as f:
        for line in f:
            text.append(line.strip())

    encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    vectors = encoder.encode(text, batch_size=2048, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
    pickle.dump(vectors, open(out_file, 'wb'))

def create_ind(in_file, out_file):
    vectors = pickle.load(open(in_file, 'rb'))
    vector_dimension = vectors.shape[1]
    index = faiss.IndexFlatL2(vector_dimension)
    index.add(vectors)
    pickle.dump(index, open(out_file, 'wb'))

def create_permuted_sents(in_file, index_file, out_file):
    pass

if __name__ == '__main__':
    # create_dump("/u/scr/smurty/pushdown-lm/data_utils/bllip-lg-depth/train.txt",
    #             1.0, 'VP', 'BLLIP_VP.txt')
    get_embs("BLLIP_VP.txt", "/nlp/scr/ananjan/BLLIP_VP_embs.pkl")