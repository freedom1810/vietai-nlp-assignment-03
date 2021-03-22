import pandas as pd
from tqdm import tqdm
tqdm.pandas()
import json
import numpy as np
import pickle
import os
import torch


def convert_labels(df):
    outputs = np.zeros((len(df), 5))
    for idx, item in tqdm(df.iterrows(), total=len(df)): 
        label = [item["goal_info"],
                item["match_info"],
                item["match_result"],
                item["substitution"],
                # item["penalty"],
                item["card_info"]]
        outputs[idx,:] = np.array(label).astype("float32")
    return outputs

def convert_text_to_input_ids(df, bpe, max_sequence_length):
    outputs = np.zeros((len(df), max_sequence_length))
    
    cls_id = 0
    eos_id = 2
    pad_id = 1

    for idx, row in tqdm(df.iterrows(), total=len(df)): 
        input_ids = bpe.encode(row['text'])

        if len(input_ids) > max_sequence_length: 
            input_ids = input_ids[:max_sequence_length] 
            input_ids[-1] = eos_id
        else:
            input_ids = input_ids + [pad_id, ]*(max_sequence_length - len(input_ids))
        outputs[idx,:] = np.array(input_ids)
    return outputs

def seed_everything(SEED):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
