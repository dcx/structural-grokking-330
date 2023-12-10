import os
import sys
# sys.path.append("../")  # Dirty hack


import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler

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
        self.fc1 = nn.Linear(hidden_size, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()

        self.to(device)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)


def get_final_hidden_state(evaluator, input_data):
    with torch.no_grad():
        embed = evaluator.model.input_embed(input_data)
        # Maybe I need this?? TransformerLM uses pos_embed, both work tho!
        pos_embed = evaluator.model.pos_embed(embed, 0)
        # hidden_states = evaluator.model.trafo.get_hidden_states(embed, is_lm=False, layer_id=0)

        hidden_states = evaluator.model.trafo.encoder(pos_embed, src_length_mask=None, layer_id=-1)

    return hidden_states.detach()
    

def get_character_index(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]


def get_gpu_data(batch, device='cpu'):
    curr_batch_dict_gpu = {key: batch[key].to(device) if key != 'string' else batch[key] for key in batch}
    inputs, labels, strings= curr_batch_dict_gpu['in'].T, curr_batch_dict_gpu['labels'].T, curr_batch_dict_gpu['string']
    return inputs, labels, strings


def get_batch_char_idxs(strings):
    char_idxs = []
    for string in strings:
        # Get batch of idxs of ')' character
        test_idxs = get_character_index(string, ')')
        char_idxs.append(test_idxs)

    return char_idxs



def train(evaluator, classifier, dataset, optimizer, criterion, device='cpu', eval_steps=1000):
    classifier.train()
    evaluator.model.eval()  # Set evaluator to evaluation mode

    step = 0
    train_dataloader = get_dataloader(dataset['train'], collate_fn=collator, batch_size=64)

    for batch in tqdm(train_dataloader, desc="Training"):
        step += 1
        optimizer.zero_grad()

        inputs, labels, strings = get_gpu_data(batch, device=device)
        char_batch = get_batch_char_idxs(strings)
        hidden_states = get_final_hidden_state(evaluator, inputs)

        for row_idx, char_idxs in enumerate(char_batch):
            outputs = classifier(hidden_states[row_idx][char_idxs]).float()
            label_batch = labels[row_idx][char_idxs]-5
            loss = criterion(outputs, label_batch)


            loss.backward()  # Perform a single backward pass
            optimizer.step()  # Update parameters

        if step % eval_steps == 0:
            test_dataloader = get_dataloader(dataset['test'], collate_fn=collator, batch_size=64)
            avg_loss, accuracy, evaluations = evaluate(evaluator, classifier, test_dataloader, criterion, device=device, visualize=True)
            print(f"{step = }: {avg_loss = }, {accuracy = }")
            print_percent_accurate(evaluations)

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

def evaluate(evaluator, classifier, dataloader, criterion, device='cpu', visualize=False, viz_every = 200):
    classifier.eval()  # Set the classifier to evaluation mode
    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    key_range = range(8)  # Adjust this range as needed

    # Create the dictionary using dictionary comprehension
    evaluations = {key: {'count': 0, 'correct': 0} for key in key_range}

    with torch.no_grad():  # Ensure no gradients are calculated
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            inputs, labels, strings = get_gpu_data(batch, device=device)
            char_idxs = get_batch_char_idxs(strings)

            hidden_states = get_final_hidden_state(evaluator, inputs)
            depth_batch = []
            char_batch = []
            prediction_batch = []



            for row_idx, char_row in enumerate(char_idxs):
                depths = bracket_depth(strings[row_idx])
                depth_batch.append(depths)

                char_batch.append(char_row)
                predictions = []
                for idx, char_idx in enumerate(char_row):
                    depth = depths[idx]

                    outputs = classifier(hidden_states[row_idx, char_idx]).float()

                    real_str = batch['string'][row_idx]
                    label = labels[row_idx][char_idx] - 5
                    

                    loss = criterion(outputs, label)
                    total_loss += loss.item()

                    _, predicted = torch.max(outputs.data, 0)
                    correct_predictions += (predicted == label).item()

                    total_predictions += 1
                    predictions.append(int(predicted))
                    evaluations[depth]['count'] += 1
                    evaluations[depth]['correct'] += (predicted == label).item()
                prediction_batch.append(predictions)

            if visualize and batch_idx % viz_every == 0:
                visualize_line(strings, prediction_batch, labels, char_batch, depth_batch)

    avg_loss = total_loss / total_predictions if total_predictions > 0 else 0
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

    return avg_loss, accuracy, evaluations





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


def get_dataloader(dataset, batch_size = 16, collate_fn = DataCollatorWithPadding):
    train_dataloader = DataLoader(
        dataset,
        sampler=RandomSampler(dataset),
        batch_size=batch_size,
        collate_fn=collate_fn,
    )
    return train_dataloader

def print_percent_accurate(evaluations_dict):
    for key, value in evaluations_dict.items():
        if value['count'] > 0:
            print(f"{key}: {value['correct']/value['count']*100} -- correct: {value['correct']} -- count {value['count']}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model evaluation script.')
    parser.add_argument('--model_file', type=str, required=True, help='Path to the saved model weight file (.pt)')

    args = parser.parse_args()

    checkpoint = 'state_final_model.pt'
    dataset_dir = 'primary_dataset'

    device ='cuda'

    num_epochs = 100

    saved_model = args.model_file

    in_vocab = CharVocabulary(chars=set('0123456789+*()[]xyzL=_'), ignore_char='_', ignore_char_idx=0)
    out_vocab = CharVocabulary(chars=set("0123456789_"))
    evaluator = LMEvaluator(in_vocab, out_vocab, 512, 8, 7, model_load_path=saved_model, device=device)

    collator = collate.VarLengthCollate(in_vocab)


    data_files = {"train": "train/data-00000-of-00001.arrow", "test": "test/data-00000-of-00001.arrow"}

    dataset = load_dataset('/home/x11kjm/sweeps/dataset', data_files=data_files)

    classifier = TwoLayerMLP(512, 10, device=device)

    criterion = torch.nn.CrossEntropyLoss() 
    optimizer = torch.optim.SGD(classifier.parameters(), lr = 0.001)

    test_dataloader = get_dataloader(dataset['test'], collate_fn=collator, batch_size=64)
    avg_loss, accuracy, evaluations = evaluate(evaluator, classifier, test_dataloader, criterion, device=device)
    print(f"Pre evaluate:")
    print_percent_accurate(evaluations)


    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train(evaluator, classifier, dataset, optimizer, criterion, eval_steps=1500, device=device)


    test_dataloader = get_dataloader(dataset['test'], collate_fn=collator, batch_size=64)
    avg_loss, accuracy, evaluations = evaluate(evaluator, classifier, test_dataloader, criterion, device=device)
    print(f"Post evaluate:")
    print_percent_accurate(evaluations)


    

