import pandas as pd
from tqdm import tqdm
# tqdm.pandas()
from torch import nn
import json
import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from transformers import *
import torch
import matplotlib.pyplot as plt
import torch.utils.data
import torch.nn.functional as F
import argparse
from transformers.modeling_utils import * 
from vncorenlp import VnCoreNLP

import os
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from transformers import AutoModel, AutoTokenizer

from dataloader.utils import *
from dataloader.data import *
from net.nets import *

def load_state(x, dir_):

    y = torch.load(dir_)
        
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for i, w in enumerate(y.items()):
        k, v = w
        name = x[i] # remove `module.`
        new_state_dict[name] = v
    return new_state_dict


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--train_path', type=str, default='./data/train.csv')
parser.add_argument('--weight', type=str, default='./data/train.csv')
parser.add_argument('--dict_path', type=str, default="./phobert/dict.txt")
parser.add_argument('--config_path', type=str, default="./phobert/config.json")
parser.add_argument('--rdrsegmenter_path', type=str, required=True)
parser.add_argument('--pretrained_path', type=str, default='./phobert/model.bin')
parser.add_argument('--max_sequence_length', type=int, default=256)
parser.add_argument('--batch_size', type=int, default=24)
parser.add_argument('--accumulation_steps', type=int, default=5)
parser.add_argument('--fold', type=int, default=0)
parser.add_argument('--seed', type=int, default=69)
parser.add_argument('--ckpt_path', type=str, default='./models')
parser.add_argument('--bpe-codes', default="./phobert/bpe.codes",type=str, help='path to fastBPE BPE')

args = parser.parse_args()
bpe = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
rdrsegmenter = VnCoreNLP(args.rdrsegmenter_path, annotators="wseg", max_heap_size='-Xmx500m') 

seed_everything(69)

# Load training data
print('load training data')
train_df = pd.read_csv(args.train_path)

if os.path.isfile('./dataset/X_train.npy') :
    X_train = np.load('./dataset/X_train.npy')
else:
    train_df.text = train_df['text'].progress_apply(lambda x: ' '.join([' '.join(sent) for sent in rdrsegmenter.tokenize(x)]))
    X_train = convert_text_to_input_ids(train_df, bpe, args.max_sequence_length)
    np.save('./dataset/X_train.npy', X_train)

if os.path.isfile('./dataset/X_test.npy') :
    X_test = np.load('./dataset/X_test.npy')
else:
    train_df.text = train_df['text'].progress_apply(lambda x: ' '.join([' '.join(sent) for sent in rdrsegmenter.tokenize(x)]))
    X_test = convert_text_to_input_ids(train_df, bpe, args.max_sequence_length)
    np.save('./dataset/X_test.npy', X_test)

if os.path.isfile('./dataset/y.npy') :
    y = np.load('./dataset/y.npy')
else:
    y = convert_labels(train_df)
    np.save('./dataset/y.npy', y)

y_test = np.ones_like(X_test)

# Load model
config = RobertaConfig.from_pretrained(
    args.config_path,
    output_hidden_states=True,
    num_labels=5,
)

model_bert = RobertaForAIViVN.from_pretrained(args.pretrained_path, config=config)
model_bert.cuda()
x = list(model_bert.state_dict().keys())
model_bert.load_state_dict(load_state(x, args.weight))


test = True
if test:
    test_dataset = VietAI3_Dataset(X_test, y_test)
else:
    splits = list(MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=0).split(X_train, y))
    train_idx, val_idx = splits[args.fold]
    test_dataset = VietAI3_Dataset(X_train[val_idx], y[val_idx])

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

test_preds = None
model_bert.eval()
for i,(x_batch, y_batch) in tqdm(enumerate(test_loader)):
    y_pred = model_bert(x_batch.cuda(), attention_mask=(x_batch>0).cuda())
    y_pred = y_pred.squeeze().detach().cpu().numpy()
    test_preds = np.atleast_1d(y_pred) if test_preds is None else np.concatenate([test_preds, np.atleast_1d(y_pred)])

test_preds = sigmoid(test_preds)
print(len(test_preds))

if test:
    
    res_df = {'id':[], 'goal_info':[], 'match_info':[], 'match_result':[], 'substitution':[], 'penalty':[], 'card_info': []}
    for i in range(len(train_df)):
        res_df['id'].append(i)
        res_df['goal_info'].append(test_preds[i][0])
        res_df['match_info'].append(test_preds[i][1])
        res_df['match_result'].append(test_preds[i][2])
        res_df['substitution'].append(test_preds[i][3])
        res_df['penalty'].append(0)
        res_df['card_info'].append(test_preds[i][4])
    pd.DataFrame(res_df).to_csv('res.csv', index = False)

else:
    print("\nAUC val = {}".format(roc_auc_score(y[val_idx], test_preds, average=None)))

    score = np.mean(roc_auc_score(y[val_idx], test_preds, average=None))
    print("\nAUC mean = {}".format(score))
