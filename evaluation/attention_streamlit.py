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

sys.path.append("../")  # Dirty hack
from vocabulary import CharVocabulary
from data_utils import build_dataset_addmult_mod10
from data_utils.ds_addmult_gen.ds_addmult_gen import evaluate_single_expression

from evaluate_transformers import (
    package_data,
    eval_single_input,
    decollate_batch,
    get_model_attn_matrix,
    get_vocab,
    LMEvaluator,
)


def main():
    # Streamlit UI elements for user inputs
    st.sidebar.title("Settings")
    model_folder = st.sidebar.text_input(
        "Enter model weight storage folder",
        "/Users/kylemcgrath/projects/structural-grokking-330/sample_runs/",
    )
    model_file = st.sidebar.selectbox("Select Model", sorted(os.listdir(model_folder)))
    vec_dim = st.sidebar.number_input("Vector Dimension", 512)
    n_heads = st.sidebar.number_input("Number of Heads", 4, step=1)
    encoder_n_layers = st.sidebar.number_input("Number of Encoder Layers", 2, step=1)
    dataset_type = st.sidebar.selectbox(
        "Select Dataset Type", ["ds-addmult-mod10", "let"]
    )

    input_sentence = st.text_input("Input Sentence", "(+5(*32))")

    if dataset_type == "let":
        st.warning("The 'let' dataset is not available yet.")
        run_button_key = None
    else:
        run_button_key = "run_button"

    if st.button("Run model", key=run_button_key):
        # if True:
        MODEL_LOAD_PATH = os.path.join(model_folder, model_file)
        tokens = list(input_sentence)

        in_vocab, out_vocab = get_vocab(dataset_type)

        evaluator = LMEvaluator(
            in_vocab,
            out_vocab,
            vec_dim,
            n_heads,
            encoder_n_layers,
            model_load_path=MODEL_LOAD_PATH,
        )

        model_out = evaluator.get_output(input_sentence)
        data_tensor = package_data([input_sentence], in_vocab)
        attn_matrices = get_model_attn_matrix(evaluator.model, data_tensor)
        model_view_html = model_view(attn_matrices, tokens, html_action="return")

        head_view_html = head_view(attn_matrices, tokens, html_action="return")

        calculated = evaluate_single_expression(input_sentence)
        print(f"LARK calculated answer: {calculated}")

        st.subheader("Answer", divider="grey")
        st.text(f"LARK calculated answer: {calculated}")
        st.text(f"Model answer: {model_out}")

        st.subheader("Head Attention", divider="grey")
        html(head_view_html.data, width=1000, height=500, scrolling=True)

        st.subheader("Model Attention", divider="grey")
        html(
            model_view_html.data,
            width=50 + (2 * n_heads * 1400),
            height=1000,
            scrolling=True,
        )


if __name__ == "__main__":
    main()
