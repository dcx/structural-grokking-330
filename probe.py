import os
import sys
import argparse



from tqdm import tqdm
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler
import pathlib

from datasets import load_dataset
from transformers.data.data_collator import DataCollatorWithPadding


import collate
from vocabulary import CharVocabulary
from evaluate_transformers import LMEvaluator



class LinearClassifier(nn.Module):
    def __init__(self, hidden_size, num_classes, device='cpu'):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(hidden_size, num_classes)
        self.to(device)

    def forward(self, x):
        return self.fc(x)

class TwoLayerMLP(nn.Module):
    def __init__(self, hidden_size, num_classes, device='cpu'):
        super(TwoLayerMLP, self).__init__()
        self.fc1 = nn.Linear(hidden_size, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.fc3 = nn.Linear(2048, num_classes)
        self.relu = nn.ReLU()
        
        self.to(device)

    def forward(self, x):
        x = self.relu(self.fc2(self.relu(self.fc1(x))))
        return self.fc3(x)


def get_final_hidden_state(evaluator, input_data, input_lengths, labels=None, device='cuda'):
    with torch.no_grad():
        embed = evaluator.model.input_embed(input_data)        
        max_len = torch.tensor(embed.shape[1]).to(device)

        mask = evaluator.model.generate_len_mask(max_len, input_lengths)
        hidden_states = evaluator.model.encoder_only(input_data, mask, layer_id=-1)

    
    return hidden_states.detach()
    

def get_character_index(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]

def bracket_depth(s):
    """
    Function to calculate the specific depth for each ')' bracket in a string, as per the new rule.
    The depth is the maximum depth of the smallest substring that contains the ')' and is balanced.

    Args:
    s (str): The input string containing brackets.

    Returns:
    list of int: The list of depths for each ')' bracket.
    """
    # Helper function to find the max depth of a substring
    def max_depth(subs):
        current_depth = max_depth = 0
        for char in subs:
            if char == '(':
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char == ')':
                current_depth -= 1
        return max_depth

    depths = []
    # Find all closing brackets and their respective depths
    for i in range(len(s)):
        if s[i] == ')':
            # Find the matching opening bracket
            balance = 1
            for j in range(i - 1, -1, -1):
                if s[j] == ')':
                    balance += 1
                elif s[j] == '(':
                    balance -= 1
                if balance == 0:
                    # Calculate the depth of the smallest balanced substring containing this bracket
                    depths.append(max_depth(s[j:i+1]))
                    break

    return depths

def print_percent_accurate(evaluations_dict):
    for key, value in evaluations_dict.items():
        if value['count'] > 0:
            print(f"{key}: {value['correct']/value['count']*100} -- correct: {value['correct']} -- count {value['count']}")

def get_batch_char_idxs(strings):
    char_idxs = []
    for string in strings:
        # Get batch of idxs of ')' character
        test_idxs = get_character_index(string, ')')
        char_idxs.append(test_idxs)

    return char_idxs

def replace_char_at_index(s, index, new_char):
    if index < 0 or index >= len(s):
        raise ValueError("Index out of bounds")
    return s[:index] + new_char + s[index + 1:]

def generate_output(idxs, max_len, predictions):
    output = ""
    prediction_idx = 0  # To keep track of the position in the predictions list

    for i in range(max_len):
        if i in idxs:
            output += str(predictions[prediction_idx])
            prediction_idx += 1
        else:
            output += "_"

    return output

def visualize_line(strings, predictions, labels, char_idxs_batch, depth_batch, sample_size=10):
    assert sample_size < len(strings), "Sample size larger than batch!"
    print("\nVisualizations")
    print(f"{'='*50}\n")
    for idx in range(sample_size):

        string = strings[idx]
        prediction = predictions[idx]
        label = labels[idx]
        char_idx_row = char_idxs_batch[idx]
        depths = depth_batch[idx]
    

        tensor_length = len(strings[idx])
        preds_string = generate_output(char_idx_row, tensor_length, prediction)
        depth_string = generate_output(char_idx_row, tensor_length, depths)



        string_label = in_vocab.ind_to_str(label.tolist())
        print(f"{string =       }")
        print(f"{preds_string = }")
        print(f"{string_label = }")
        print(f"{depth_string = }\n")


def train(evaluator, classifier, dataset, optimizer, criterion, device='cpu', step=None, eval_steps=1000):
    classifier.train()
    evaluator.model.eval()  # Set evaluator to evaluation mode


    train_dataloader = get_dataloader(dataset['train'], collate_fn=collator, batch_size=64)
    counts = {k: 0 for k in range(10)}
    for batch in tqdm(train_dataloader, desc="Training"):
        step += 1

        inputs = batch['in'].transpose(0,1).to(device)
        input_lengths = batch['in_len'].long().to(device)
        labels = batch['labels'].transpose(0,1).long().to(device)
        strings = batch['string']
        char_idxs = get_batch_char_idxs(strings)
        hidden_states = get_final_hidden_state(evaluator, inputs, input_lengths, labels=labels, device=device)

        for row_idx, char_batch in enumerate(char_idxs):
            if char_batch:
                label = labels[row_idx][char_batch] - 5 # Hack shift 
                outputs = classifier(hidden_states[row_idx, char_batch])
                loss = criterion(outputs, label)

                loss.backward()  # Perform a single backward pass
                optimizer.step()  # Update parameters
                optimizer.zero_grad()


        if step % eval_steps == 0:
            test_dataloader = get_dataloader(dataset['test'], collate_fn=collator, batch_size=64)
            avg_loss, accuracy, evaluations = evaluate(evaluator, classifier, test_dataloader, criterion, device=device, visualize=True)
            print(f"{step = }: {avg_loss = }, {accuracy = }")
            print_percent_accurate(evaluations)
            wandb.log({"loss": avg_loss, "accuracy": accuracy, "step" :step})
            

    return step


def evaluate(evaluator, classifier, dataloader, criterion, device='cpu', visualize=False, viz_every=150):
    classifier.eval()  # Set the classifier to evaluation mode
    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    total_iters = 0

    cfg_depth = 8

    evaluations = {key: {'count': 0, 'correct': 0} for key in range(cfg_depth)}
    with torch.no_grad():  # Ensure no gradients are calculated
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            inputs = batch['in'].transpose(0,1).to(device)
            input_lengths = batch['in_len'].long().to(device)
            labels = batch['labels'].transpose(0,1).long().to(device)
            strings = batch['string']
            char_idxs = get_batch_char_idxs(strings)


            hidden_states = get_final_hidden_state(evaluator, inputs, input_lengths)
            for row_idx, char_batch in enumerate(char_idxs):
                
                depths = bracket_depth(strings[row_idx])

                if char_batch:

                    outputs = classifier(hidden_states[row_idx, char_batch])

                    label = labels[row_idx][char_batch] - 5
                    loss = criterion(outputs, label)
                    total_loss += loss.item()

                    _, predicted = torch.max(outputs.data, 1)
                    
                    # correct_predictions += int(sum((predicted == label)))
                    total_predictions += len(label)
                    total_iters +=1

                    for pred_idx, prediction in enumerate(predicted):
                        correct_int = (prediction == label[pred_idx]).item()
                        correct_predictions += correct_int
                        depth = depths[pred_idx]
                        evaluations[depth]['count'] += 1
                        evaluations[depth]['correct'] += correct_int

                # if visualize and batch_idx % viz_every == 0:
                #     visualize_line(strings, predicted, labels, char_batch, depth_batch)

    avg_loss = total_loss / total_iters if total_iters > 0 else 0
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0


    accuracies_at_depth = {f"accuracy_depth_{key}": (value['correct'] / value['count']) if value['count'] > 0 else 0 for key, value in evaluations.items()}
    wandb.log(accuracies_at_depth)

    return avg_loss, accuracy, evaluations



def get_dataloader(dataset, batch_size = 16, collate_fn = DataCollatorWithPadding):
    train_dataloader = DataLoader(
        dataset,
        sampler=RandomSampler(dataset),
        batch_size=batch_size,
        collate_fn=collate_fn,
    )
    return train_dataloader

if __name__ == '__main__':

    learning_rate = 0.0001
    

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=pathlib.Path)
    parser.add_argument("--wandb_name", type=str)
    args = parser.parse_args()


    saved_model = args.model


    device ='cuda'


    in_vocab = CharVocabulary(chars=set('0123456789+*()[]xyzL=_'), ignore_char='_', ignore_char_idx=0)
    out_vocab = CharVocabulary(chars=set("0123456789_"))
    evaluator = LMEvaluator(in_vocab, out_vocab, 512, 8, 7, model_load_path=saved_model, device=device)

    collator = collate.VarLengthCollate(in_vocab)

    # dataset = load_dataset("cs330/minH_1_maxH_4_maxW_100_holdUniq_1000_regex_None_pSub_0.15_maxHeld_10000_balance_True_inter_True")
    data_files = {"train": "train/data-00000-of-00001.arrow", "test": "test/data-00000-of-00001.arrow"}
    dataset = load_dataset('/home/x11kjm/sweeps/dataset', data_files=data_files)

    classifier = TwoLayerMLP(512, 10, device=device)

    weights = torch.tensor([0.1, 1, 1,1,1,1,1,1,1,1]).to(device)

    criterion = torch.nn.CrossEntropyLoss(weight=weights) 
    optimizer = torch.optim.Adam(classifier.parameters(), lr = learning_rate)

    wandb.init(project="probe", name=args.wandb_name)
    wandb.config.update({"learning_rate": learning_rate})


    step = 0
    for i in range(100):
        step = train(evaluator, classifier, dataset, optimizer, criterion, eval_steps=10, step=step, device=device)

    wandb.finish()