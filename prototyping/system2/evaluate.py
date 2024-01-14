import torch, torch.nn as nn, torch.utils.data as data
import lightning as L
import torch.nn.functional as F
import random

import json
from tqdm import tqdm
from model import PlanTransformer
import dataset
from dataset import collate_fn, detokenize
from utils import get_difference


checkpoint_path = 'saved_checkpoint/epoch=4-step=150568.ckpt'

torch.set_float32_matmul_precision('medium')

hparams = {
    'bs': 2,
    'num_workers': 1,
    'pad_token_id': 30,

    # dataset
    'csv_file': 'data/amlif-data/amlif-50k.csv',
    'use_cur_action': True,
    'use_cur_action_result': True,
    'use_next_action': True,
    'val_check_interval': 300, # in steps
    'holdout_trees_frac': 0.15,
    'train_frac': 0.99,
    'val_frac': 0.01, # currently ignored
    'test_max_items': None,
}
# reminder: unlike main framework, here we are plugging test into val
# because Lightning (correctly) doesn't have a test step during training


hparams['model_hparams'] = {
    'd_model': 512,
    'nhead': 8,
    'num_encoder_layers': 6, 
    'dropout': 0.1,
    'ntoken': 32,
    'lr': 3e-4,
    'pad_token_id': hparams['pad_token_id'],
    'weight_decay': 0,
}

device = torch.device('cuda:0')


# setup data
_, _, _, eval_dataset = dataset.make_datasets(
    hparams['csv_file'],
    holdout_trees_frac=hparams['holdout_trees_frac'],
    train_frac=hparams['train_frac'], val_frac=hparams['val_frac'],
    test_max_items=hparams['test_max_items'],
    use_cur_action=hparams['use_cur_action'], 
    use_cur_action_result=hparams['use_cur_action_result'], 
    use_next_action=hparams['use_next_action'])

# model = PlanTransformer.load_from_checkpoint(checkpoint_path, hparams_file=hparams['model_hparams'])
model = PlanTransformer.load_from_checkpoint(checkpoint_path, **hparams['model_hparams'])

model.eval()


all_new_sequence_indices = eval_dataset.get_indices_of_new_sequences()

random.shuffle(all_new_sequence_indices)  # Shuffle the indices to randomize the order

sampled_indices = set()  # Set to keep track of sampled indices

# Determine the final step for each sequence
eval_dataset.csv_data['new_sequence'] = eval_dataset.csv_data['idx'].diff().ne(0).cumsum()
final_steps = eval_dataset.csv_data.groupby('new_sequence')['step'].max().to_dict()
# Add 'final_step' column to the DataFrame
eval_dataset.csv_data['final_step'] = eval_dataset.csv_data['new_sequence'].map(final_steps)


# Organize data by 'new_sequence'
grouped_data = eval_dataset.csv_data.groupby('new_sequence')

# Shuffle group keys (i.e., unique 'new_sequence' values)
group_keys = list(grouped_data.groups.keys())
random.shuffle(group_keys)

num_examples = 0
correct_dist = []

percentage_correct_dist = []
operation_failed_dist = {}
int_final_ans_dist = []
example_height = []
final_preds = []
prompts = []


def list_rindex(li, x):
    for i in reversed(range(len(li))):
        if li[i] == x:
            return i
    return 0 # element is not in list

with torch.no_grad():
    # Iterate over each randomly shuffled group



    for group_key in tqdm(group_keys):
        group_df = grouped_data.get_group(group_key)  # Get the DataFrame for the group
        first_row_data = group_df.iloc[1] # skip blank cur_state. steps to validate are final_step - 1
        group_final_step = first_row_data.final_step
        final_answer = int(group_df.iloc[group_final_step].next_state)
        # Logging
        predictions = []
        stepwise_prediction_correct = []
        stepwise_answer_list = []


        x, y = eval_dataset.get_curr_and_next_state(first_row_data)
        x_batch, y_batch, padding_mask = collate_fn([(x,y)])

        for idx, (_, row) in enumerate(group_df.iterrows()):
            if idx == 0:
                continue
            x_batch = x_batch.to(device)
            padding_mask = padding_mask.to(device)

            y_hat = model(x_batch, padding_mask=padding_mask) # (seq_len, bs, ntoken)
            ce_y_hat = torch.permute(y_hat, (1, 2, 0)) # (bs, ntoken, seq_len)

            # accuracy
            y_pred = torch.argmax(y_hat, dim=2).T # (bs, seq_len)
            predicted_chars = dataset.detokenize(y_pred.tolist()[0])
            isolated = predicted_chars.split('|')[0].strip(' ')

            step_correct = isolated == row.next_state
            stepwise_prediction_correct.append(step_correct)

            # Metrics
            stepwise_answer_list.append(row.next_state)
            predictions.append(isolated)

            x, y, = eval_dataset.get_curr_and_next_state(row, predicted=isolated)
            x_batch, y_batch, padding_mask = collate_fn([(x,y)])

            # break condition 1 -> len is 0
            if len(isolated) == 1:
                mod10_ans = None
                try:
                    mod10_ans = int(isolated)
                except ValueError as e:
                    print(f"Cannot convert '{isolated}' to int")
                break

        # check final answer
            
        # Logging
        prompts.append(first_row_data.cur_state)
        example_height.append(group_final_step)
            
        answer_list = []
        operations = []
        for idx, (_, row) in enumerate(group_df.iterrows()):
            if idx == 0:
                continue

            x, y = eval_dataset.get_curr_and_next_state(row)
            answer = dataset.detokenize(y)

            ans = answer.split('|')[0]
            operations.append(row.cur_action_type)
            answer_list.append(ans)

        # Comparison with predictions, note that both list lengths are variable, pick the shortest to iterate through.
        accuracy_list = [answer_list[i] == predictions[i].split('|')[0] for i in range(min(len(answer_list), len(predictions)))]

        final_pred_row = predictions[-1].split('|')[0].strip(' ')

        final_preds.append(final_pred_row)

        correct = False
        if len(final_pred_row) == 1:
            try:
                pred_answer = int(final_pred_row)
            except ValueError as e:
                pred_answer = None
            
            if pred_answer:
            
                correct = pred_answer == final_answer
                int_final_ans_dist.append(True)
        else:
            int_final_ans_dist.append(False)
            correct = False

        correct_dist.append(correct)
        num_examples += 1


        percentage_true = sum(accuracy_list) / len(accuracy_list) * 100 if accuracy_list else 0
        percentage_correct_dist.append(percentage_true)

        last_true_idx = list_rindex(accuracy_list, True)
        
        if percentage_true != 100:
            if last_true_idx == 0 or last_true_idx == group_final_step-1:
                fail_op = operations[last_true_idx]
            else:
                fail_op = operations[last_true_idx+1]

            if fail_op not in operation_failed_dist:
                operation_failed_dist[fail_op] = 1
            else:
                operation_failed_dist[fail_op] += 1

def save_data_to_json(filename, data):
    with open(filename, 'w') as f:
        json.dump(data, f, default=convert)

def convert(o):
    if isinstance(o, np.int64): 
        return int(o)  
    raise TypeError

# Data to be saved
data_to_save = {
    "num_examples": num_examples,
    "correct_dist": correct_dist,
    "percentage_correct_dist": percentage_correct_dist,
    "operation_failed_dist": operation_failed_dist,
    "int_final_ans_dist": int_final_ans_dist,
    "example_height": example_height
}

# Save to JSON
save_data_to_json("eval_data.json", data_to_save)
