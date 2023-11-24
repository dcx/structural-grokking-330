from streamlit.components.v1 import html

from bertviz import model_view, head_view

import os
import torch
import bertviz
import numpy as np
from bertviz import model_view, head_view
from bertviz.neuron_view import show
import matplotlib.pyplot as plt
import seaborn as sns

import streamlit as st


import sys
sys.path.append('../') # Dirty hack
from train_transformers import get_base_transformer_lm, get_datasets_and_vocab
from vocabulary import CharVocabulary
from data_utils import build_dataset_addmult_mod10
from data_utils.ds_addmult_gen.ds_addmult_gen import evaluate_single_expression

from evaluate_transformers import package_data, eval_single_input, decollate_batch


from util import test_continuations



def get_model_attn_matrix(model, data_tensor):
    src = data_tensor['in'].T
    src_len = data_tensor['in_len']

    mask = model.generate_len_mask(src.shape[1], src_len)
    src = model.pos_embed(model.input_embed(src), 0)
    attn_matrices = model.trafo.get_attn_matrices(
        src, tgt=None, src_length_mask=mask
    )
    attention_matrices = tuple([i for i in attn_matrices])
    return attention_matrices


def get_lm_output(in_vocab, lm, in_sentence, device='cpu'):

    def tokenizer(s):
        return [lm.encoder_sos] + in_vocab(s)

    # stringified = in_vocab.ind_to_str(in_sentence)
    
    query = in_sentence + '='


    out = test_continuations(tokenizer, lm, [query], gpu_id=0, batch_size=16, device=device)

    desired_out = '0123456789'
    desired_out_idx = [in_vocab(w) for w in desired_out]
    out = out[:, desired_out_idx]
    decoded = out.argmax(dim=1)
    return int(decoded[0])
    

def main():
    # Streamlit UI elements for user inputs
    st.sidebar.title("Settings")
    model_folder = st.sidebar.text_input("Enter model weight storage folder", '/Users/kylemcgrath/projects/structural-grokking-330/sample_runs/')
    model_file = st.sidebar.selectbox("Select Model", sorted(os.listdir(model_folder)))
    vec_dim = st.sidebar.number_input("Vector Dimension", 512)
    n_heads = st.sidebar.number_input("Number of Heads", 4, step=1)
    encoder_n_layers = st.sidebar.number_input("Number of Encoder Layers", 2, step=1)
    
    input_sentence = st.text_input("Input Sentence", '(+(+4(*32))(*45))')

    # if st.button("Run"):
    if True:
        MODEL_LOAD_PATH = os.path.join(model_folder, model_file)
        in_vocab = CharVocabulary(chars=set('0123456789+*()='))
        tokens = list(input_sentence)

        model, interface = get_base_transformer_lm(in_vocab, vec_dim, n_heads, encoder_n_layers, model_load_path=MODEL_LOAD_PATH)
        data_tensor = package_data([input_sentence], in_vocab)
        attn_matrices = get_model_attn_matrix(model, data_tensor)
        model_out = eval_single_input(interface, data_tensor).outputs
        model_view_html = model_view(attn_matrices, tokens, html_action='return')

        calculated = evaluate_single_expression(input_sentence)
        print(f"LARK calculated answer: {calculated}")
        print(f"model_output: {int(np.argmax(model_out))}")

        model_output_text = None

        output_probs = torch.nn.functional.softmax(model_out, dim=-1)
        output_indices = torch.argmax(output_probs, dim=-1)

        lm_out = get_lm_output(in_vocab, model, input_sentence, device='cpu')



        out_tokens = []
        for indices in output_indices:
            token = in_vocab.ind_to_str(indices.tolist())
            out_tokens.append(token)

        st.text(f"LARK calculated answer: {calculated}")
        st.text(f"Model predicted: {out_tokens}")
        st.text(f"Model answer: {lm_out}")
        html(model_view_html.data, width=50 +(2*n_heads*1400), height=1000, scrolling=True)



        

if __name__ == '__main__':
    main()