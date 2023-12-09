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
    def __init__(self, hidden_size, num_classes):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        return self.fc(x)

class TwoLayerMLP(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(TwoLayerMLP, self).__init__()
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)



def generate_src_length_mask(sequence_length, max_length=None):
    # Convert booleans to integers
    input_data = [1 for value in sequence_length]

    # Define the required sequence length
    required_seq_length = max_length if max_length else len(sequence_length)

    # Pad the sequence
    padded_input = input_data + [0] * (required_seq_length - len(input_data))

    # Convert to tensor and reshape
    # Reshape to [batch_size, sequence_length, feature_size]
    input_tensor = torch.tensor([padded_input]).unsqueeze(-1)  # Adds a feature dimension

    return input_tensor

def get_final_hidden_state(evaluator, input_data):

    # Otherwise there will be non-deterministic output
    
    embed = evaluator.model.input_embed(input_data)
    # Maybe I need this?? TransformerLM uses pos_embed, both work tho!
    pos_embed = evaluator.model.pos_embed(embed, 0)
    # hidden_states = evaluator.model.trafo.get_hidden_states(embed, is_lm=False, layer_id=0)

    hidden_states = evaluator.model.trafo.encoder(embed, src_length_mask=None, layer_id=-1)

    return hidden_states
    

# ''.join([str(i) for i in row.tolist()])
# 

def train(evaluator, classifier, data_loader, optimizer, criterion, device='cpu', eval_steps =1000):

    classifier.train()  # Set the classifier to training mode
    step = 0

    for batch in tqdm(data_loader, desc="Training"):
        step += 1

        curr_batch_dict_gpu = {}
        for key in batch:
            if (key == 'string'):
                curr_batch_dict_gpu[key] = batch[key]
            else:
                curr_batch_dict_gpu[key] = batch[key].to(device)
        inputs, labels = curr_batch_dict_gpu['in'].T, curr_batch_dict_gpu['labels'].T

        evaluator.model.eval()

        column_length = len(inputs[0])

        with torch.no_grad():

            for idx in range(column_length):
                    idx +=1 

                    column = inputs[:, 0:idx]
                    label = labels[:, 0:idx]

                
                    hidden_states = get_final_hidden_state(evaluator, column)


                    for i in range(hidden_states.shape[1]):

                        # 0 is for the batch size select, will remove this.
                        outputs = classifier(hidden_states[:,i]).float()
                        
                        loss = criterion(outputs, label[:, i])
                        optimizer.zero_grad()
                        
                        # loss = torch.autograd.Variable(loss, requires_grad = True)
                        loss.backward()
                        optimizer.step()

        if step % eval_steps == 0:
            print(f"Step {step}: Current Loss = {loss.item()}")


def evaluate(model, classifier, data_loader):
    model.eval()
    classifier.eval()
    total_correct = 0
    total_samples = 0

    for batch in tqdm(data_loader, desc="Evaluating"):
        inputs, labels = batch
        with torch.no_grad():
            hidden_states = get_final_hidden_state(model, inputs)
            outputs = classifier(hidden_states)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()

    accuracy = (total_correct / total_samples) * 100
    return accuracy

def get_dataloader(dataset, batch_size = 16, collate_fn = DataCollatorWithPadding):
    train_dataloader = DataLoader(
        dataset,
        sampler=RandomSampler(dataset),
        batch_size=batch_size,
        collate_fn=collate_fn,
    )
    return train_dataloader

if __name__ == '__main__':

    model_dir = '/home/x11kjm/structural-grokking/saved_models/231130_high_depth/run_nhead_4_enclayer_4_vec_512_dsheight_6_dsmin_1_regularize_False_gold_False'

    checkpoint = 'state_final_model.pt'
    dataset_dir = 'primary_dataset'

    saved_model = os.path.join(model_dir, checkpoint)

    in_vocab = CharVocabulary(chars=set('0123456789+*()[]xyzL=_'), ignore_char='_', ignore_char_idx=0)
    out_vocab = CharVocabulary(chars=set("0123456789_"))
    evaluator = LMEvaluator(in_vocab, out_vocab, 512, 4, 4, model_load_path=saved_model)

    collator = collate.VarLengthCollate(in_vocab)



    dataset = load_dataset("cs330/minH_1_maxH_6_maxW_100_holdUniq_1000_regex_None_pSub_0.15_maxHeld_10000_balance_True_inter_True")
    train_dataloader = get_dataloader(dataset['train'], collate_fn=collator)

    classifier = LinearClassifier(512, 22)

    criterion = torch.nn.CrossEntropyLoss() 
    optimizer = torch.optim.SGD(classifier.parameters(), lr = 0.001)

    train(evaluator, classifier, train_dataloader, optimizer, criterion)

