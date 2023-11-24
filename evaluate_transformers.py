import torch
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from vocabulary import CharVocabulary

import layers

from transformers.data.data_collator import DataCollatorWithPadding

from datasets import DatasetDict


from torch.utils.data import DataLoader

import collate


def package_data(input_text, tokenizer, device="cpu"):
    enum_input = enumerate(input_text)

    data = [
        {
            "in": tokenizer(text),
            "in_len": len(tokenizer(text)),
            "labels": 0,
            "idxs": idx,
        }
        for idx, text in enum_input
    ]

    collator = collate.VarLengthCollate(None)
    train_dataloader = DataLoader(
        data,
        collate_fn=collator,
        batch_size=16
    )
    data_dict = list(train_dataloader)[0]
    # Send to device
    out = {k: v.to(device=device, non_blocking=True) for k, v in data_dict.items()}

    return out


def prepare_dataset(dataset, batch_size=16):
    collator = collate.VarLengthCollate(None)

    train_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collator,
    )
    return train_dataloader


def decollate_batch(collated_batch, lengths):
    """
    Splits the collated batch back into individual sequences based on the provided lengths.
    :param collated_batch: The collated batch tensor.
    :param lengths: The tensor with the lengths of each sequence in the batch.
    :return: A list of tensor sequences.
    """
    decollated = []

    collated_batch = collated_batch.T
    for i, length in enumerate(lengths):
        sequence = collated_batch[i][: int(length)]
        decollated.append(sequence)
    return decollated


# Function to evaluate a single input statement
def eval_single_input(model, input_data):
    with torch.no_grad():
        logits = model(input_data)

    return logits


# Function to run the model on an entire dataset
def eval_dataset(
    model, tokenizer, dataset, device, output_path="logs/incorrect_predictions.csv"
):
    num_processed = 0
    num_incorrect = 0

    in_vocabulary = CharVocabulary(chars=set("0123456789+*()="))
    preds = []
    for batch in dataset:
        batch_gpu = {}
        for key in batch:
            batch_gpu[key] = batch[key].to(device)

        batch_predictions = eval_single_input(model, batch_gpu).outputs.cpu()
        decollated_batch = decollate_batch(batch["in"], batch["in_len"])

        for idx, tokenized_in_text in enumerate(decollated_batch):
            num_processed += 1
            in_text = in_vocabulary(tokenized_in_text.tolist())
            prediction = int(np.argmax(batch_predictions[idx]).cpu())
            answer = int(batch["labels"][idx])

            if prediction != answer:
                num_incorrect += 1
            preds.append(
                {
                    "input": in_text,
                    "predicted": prediction,
                    "answer": answer,
                    "correct": prediction == answer,
                }
            )
    print(f"{num_processed = }, {num_incorrect = }")
    # Convert to DataFrame and save to CSV
    incorrect_df = pd.DataFrame(preds)
    incorrect_df.to_csv(output_path, index=False)


# Function to generate confusion matrix
def generate_confusion_matrix(model, tokenizer, dataset, device):
    y_true = []
    y_pred = []
    for example in dataset:
        true_answer = example["answer"]
        predicted_answer = eval_single_input(model, tokenizer, example["input"], device)
        y_true.append(true_answer)
        y_pred.append(predicted_answer)

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt="d")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()


def evaluate_network(args, interface, test_dataset, device="cpu", tokenizer=None):
    interface.model.eval()

    mode = "classification"

    if tokenizer is not None:
        train_data_collator = DataCollatorWithPadding(tokenizer=None)
    else:
        train_data_collator = collate.VarLengthCollate(tokenizer)

    # Evaluate a single statement if specified
    if args.eval_single_input:
        input_text = args.eval_single_input
        print(f"Evaluating single input: {input_text}")
        input_data = package_data(input_text, tokenizer)
        output = eval_single_input(interface, input_data)

        if mode == "classification":
            selected_class = int(np.argmax(output.outputs.cpu(), axis=1))
        print(f"Input: {input_text}, Output from model: {selected_class}")

    # Evaluate on the entire dataset if specified
    if args.eval_dataset:
        print(f"Evaluating on dataset: {args.dataset}")

        dataloader = prepare_dataset(test_dataset)
 
        eval_dataset(interface, tokenizer, dataloader, device)

    # Generate and save confusion matrix if specified
    if args.eval_confusion_matrix:
        print("Generating confusion matrix.")
        generate_confusion_matrix(interface.model, tokenizer, dataset, device)
