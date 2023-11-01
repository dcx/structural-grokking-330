import torch
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


# Function to evaluate a single input statement
def eval_single_input(model, tokenizer, input_text, device):
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model(inputs)
    return output

# Function to run the model on an entire dataset
def eval_dataset(model, tokenizer, dataset, device):
    incorrect_preds = []
    for example in dataset:
        prediction = eval_single_input(model, tokenizer, example['input'], device)
        if prediction != example['answer']:
            incorrect_preds.append({
                'input': example['input'],
                'predicted': prediction,
                'correct': example['answer']
            })
    # Convert to DataFrame and save to CSV
    incorrect_df = pd.DataFrame(incorrect_preds)
    incorrect_df.to_csv('/path/to/incorrect_predictions.csv', index=False)

# Function to generate confusion matrix
def generate_confusion_matrix(model, tokenizer, dataset, device):
    y_true = []
    y_pred = []
    for example in dataset:
        true_answer = example['answer']
        predicted_answer = eval_single_input(model, tokenizer, example['input'], device)
        y_true.append(true_answer)
        y_pred.append(predicted_answer)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
