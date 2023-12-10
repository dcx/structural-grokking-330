import os
import sys
# sys.path.append("../")  # Dirty hack

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


def get_final_hidden_state(evaluator, input_data, input_lengths, device='cuda'):
    with torch.no_grad():
        embed = evaluator.model.input_embed(input_data)
        # pos_embed = evaluator.model.pos_embed(embed, 0)
        # hidden_states = evaluator.model.trafo.get_hidden_states(embed, is_lm=False, layer_id=0)
        
        max_len = embed.shape[1]

        input = input_data

        prefix_sos = (torch.ones((input.shape[0],1)).long() * evaluator.model.encoder_sos).to(device)
        input = torch.cat((prefix_sos, input), dim=1) # add SOS token
        input_lengths += 1
        outputs = evaluator.model(input, input_lengths)
        input_lengths = input_lengths.long()
        mask = torch.arange(max_len).expand(len(input_lengths), max_len) >= input_lengths.unsqueeze(1)
        
        mask = mask.to(device)
        #mask = mask.unsqueeze(2).dtype(torch.long)

        #mask = mask.expand((-1,-1,512))




        hidden_states = evaluator.model.trafo.encoder(embed, src_length_mask=mask, layer_id=-1)


    return hidden_states.detach()
    

def get_character_index(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]




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

def visualize_line(string, predictions, label, char_idxs):

    tensor_length = len(string)
    view_string = generate_output(char_idxs, tensor_length, predictions)

    print("\nVisualizations")
    print(f"{'='*50}\n")

    string_label = in_vocab.ind_to_str(label.tolist())
    print(f"{string = }")
    print(f"{view_string = }")
    print(f"{string_label = }")


def train(evaluator, classifier, dataset, optimizer, criterion, device='cpu', eval_steps=1000):
    classifier.train()
    evaluator.model.eval()  # Set evaluator to evaluation mode

    step = 0
    train_dataloader = get_dataloader(dataset['train'], collate_fn=collator, batch_size=64)

    for batch in tqdm(train_dataloader, desc="Training"):
        step += 1

        inputs = batch['in'].transpose(0,1).to(device)
        input_lengths = batch['in_len'].long().to(device)
        labels = batch['labels'].long().to(device)
        strings = batch['string']
        char_idxs = get_batch_char_idxs(strings)
        hidden_states = get_final_hidden_state(evaluator, inputs, input_lengths)

        for row_idx, char_batch in enumerate(char_idxs):
            if char_batch:
                for char_idx in char_batch:
                    outputs = classifier(hidden_states[row_idx, char_idx])

                    label = labels[row_idx][char_idx] - 5 # Hack shift 
                    loss = criterion(outputs, label)

                    loss.backward()  # Perform a single backward pass
                    optimizer.step()  # Update parameters
                    optimizer.zero_grad()


        if step % eval_steps == 0:
            test_dataloader = get_dataloader(dataset['test'], collate_fn=collator, batch_size=64)
            avg_loss, accuracy = evaluate(evaluator, classifier, test_dataloader, criterion, device=device)
            print(f"{step = }: {avg_loss = }, {accuracy = }")


def evaluate(evaluator, classifier, dataloader, criterion, device='cpu', visualize=False):
    classifier.eval()  # Set the classifier to evaluation mode
    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():  # Ensure no gradients are calculated
        for batch in tqdm(dataloader, desc="Evaluating"):
            inputs, labels, strings = get_gpu_data(batch, device=device)
            char_idxs = get_batch_char_idxs(strings)

            input_lengths = batch['in_len']
            hidden_states = get_final_hidden_state(evaluator, inputs, input_lengths)

            for row_idx, char_batch in enumerate(char_idxs):
                if char_batch:
                    predictions = []
                    for char_idx in char_batch:
                        outputs = classifier(hidden_states[row_idx, char_idx])

                        label = labels[row_idx][char_idx] - 5
                        loss = criterion(outputs, label)
                        total_loss += loss.item()

                        _, predicted = torch.max(outputs.data, 0)
                        correct_predictions += (predicted == label).item()
                        total_predictions += 1
                        predictions.append(int(predicted))
                    if visualize:
                        visualize_line(strings[row_idx], predictions, labels[row_idx], batch)

    avg_loss = total_loss / total_predictions if total_predictions > 0 else 0
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

    return avg_loss, accuracy



def get_dataloader(dataset, batch_size = 16, collate_fn = DataCollatorWithPadding):
    train_dataloader = DataLoader(
        dataset,
        sampler=RandomSampler(dataset),
        batch_size=batch_size,
        collate_fn=collate_fn,
    )
    return train_dataloader

if __name__ == '__main__':

    # model_dir = '/home/x11kjm/structural-grokking/saved_models/231130_high_depth/run_nhead_4_enclayer_4_vec_512_dsheight_6_dsmin_1_regularize_False_gold_False'
    # checkpoint = 'state_final_model.pt'
    # dataset_dir = 'primary_dataset'

    model_dir = '/home/x11kjm/sweeps/probe_reg/run_nhead_8_enclayer_7_vec_512_dsheight_8_dsmin_1_regularize_True_gold_True'
    checkpoint = 'state_200000.pt'

    device ='cuda'

    saved_model = os.path.join(model_dir, checkpoint)

    in_vocab = CharVocabulary(chars=set('0123456789+*()[]xyzL=_'), ignore_char='_', ignore_char_idx=0)
    out_vocab = CharVocabulary(chars=set("0123456789_"))
    evaluator = LMEvaluator(in_vocab, out_vocab, 512, 8, 7, model_load_path=saved_model, device=device)

    collator = collate.VarLengthCollate(in_vocab)



    # dataset = load_dataset("cs330/minH_1_maxH_4_maxW_100_holdUniq_1000_regex_None_pSub_0.15_maxHeld_10000_balance_True_inter_True")
    data_files = {"train": "train/data-00000-of-00001.arrow", "test": "test/data-00000-of-00001.arrow"}
    dataset = load_dataset('/home/x11kjm/sweeps/dataset', data_files=data_files)

    classifier = TwoLayerMLP(512, 10, device=device)

    criterion = torch.nn.CrossEntropyLoss() 
    optimizer = torch.optim.SGD(classifier.parameters(), lr = 0.01)

    for i in range(100):
        train(evaluator, classifier, dataset, optimizer, criterion, eval_steps=1000, device=device)

