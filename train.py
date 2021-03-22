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

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--train_path', type=str, default='./data/train.csv')
parser.add_argument('--dict_path', type=str, default="./phobert/dict.txt")
parser.add_argument('--config_path', type=str, default="./phobert/config.json")
parser.add_argument('--rdrsegmenter_path', type=str, required=True)
parser.add_argument('--pretrained_path', type=str, default='./phobert/model.bin')
parser.add_argument('--max_sequence_length', type=int, default=256)
parser.add_argument('--batch_size', type=int, default=24)
parser.add_argument('--accumulation_steps', type=int, default=5)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--fold', type=int, default=0)
parser.add_argument('--seed', type=int, default=69)
parser.add_argument('--lr', type=float, default=3e-5)
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

if os.path.isfile('./dataset/y.npy') :
    y = np.load('./dataset/y.npy')
else:
    y = convert_labels(train_df)
    np.save('./dataset/y.npy', y)

# Load model
config = RobertaConfig.from_pretrained(
    args.config_path,
    output_hidden_states=True,
    num_labels=5,
)

model_bert = RobertaForAIViVN.from_pretrained(args.pretrained_path, config=config)
model_bert.cuda()

if torch.cuda.device_count():
    print(f"Training using {torch.cuda.device_count()} gpus")
    model_bert = nn.DataParallel(model_bert)
    tsfm = model_bert.module.roberta
else:
    tsfm = model_bert.roberta

# Creating optimizer and lr schedulers
param_optimizer = list(model_bert.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
num_train_optimization_steps = int(args.epochs*len(train_df)/args.batch_size/args.accumulation_steps)
optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=num_train_optimization_steps)  # PyTorch scheduler
scheduler0 = get_constant_schedule(optimizer)  # PyTorch scheduler

if not os.path.exists(args.ckpt_path):
    os.mkdir(args.ckpt_path)

# splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=123).split(X_train, y))
splits = list(MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=0).split(X_train, y))
for fold, (train_idx, val_idx) in enumerate(splits):
    if fold != args.fold:
        continue

    print("\nTraining for fold {}".format(fold))
    best_score = 0

    train_dataset = VietAI3_Dataset(X_train[train_idx], y[train_idx])
    valid_dataset = VietAI3_Dataset(X_train[val_idx], y[val_idx])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

    # tq = tqdm(range(args.epochs + 1))
    for child in tsfm.children():
        for param in child.parameters():
            if not param.requires_grad:
                print("whoopsies")
            param.requires_grad = False
    frozen = True
    for epoch in range(args.epochs):
        print('\n------------Epoch-----------', epoch)

        if epoch > 0 and frozen:
            for child in tsfm.children():
                for param in child.parameters():
                    param.requires_grad = True
            frozen = False
            del scheduler0
            torch.cuda.empty_cache()

        val_preds = None
        avg_loss_train = 0.
        avg_loss_val = 0.
        avg_accuracy = 0.

        optimizer.zero_grad()

        for i,(x_batch, y_batch) in tqdm(enumerate(train_loader)):
            model_bert.train()
            y_pred = model_bert(x_batch.cuda(), attention_mask=(x_batch>0).cuda())

            loss =  F.binary_cross_entropy_with_logits(y_pred.view(-1).cuda(), y_batch.view(-1).float().cuda())
            loss = loss.mean()
            loss.backward()

            if i % args.accumulation_steps == 0 or i == len(train_loader) - 1:
                optimizer.step()
                optimizer.zero_grad()
                
                if not frozen:
                    scheduler.step()
                else:
                    scheduler0.step()

            lossf = loss.item()
            avg_loss_train += loss.item() * x_batch.shape[0]
        
        model_bert.eval()
        for i,(x_batch, y_batch) in tqdm(enumerate(valid_loader)):
            y_pred = model_bert(x_batch.cuda(), attention_mask=(x_batch>0).cuda())
            loss =  F.binary_cross_entropy_with_logits(y_pred.view(-1).cuda(), y_batch.view(-1).float().cuda())
            loss = loss.mean()
            avg_loss_val += loss.item() * x_batch.shape[0]

            y_pred = y_pred.squeeze().detach().cpu().numpy()
            val_preds = np.atleast_1d(y_pred) if val_preds is None else np.concatenate([val_preds, np.atleast_1d(y_pred)])

            
        val_preds = sigmoid(val_preds)

        best_th = 0
        print('Train loss {}'.format(avg_loss_train/len(train_idx)))
        print('Val loss   {}'.format(avg_loss_val/len(val_idx)))
        print("AUC val    {}".format(roc_auc_score(y[val_idx], val_preds, average=None)))
        
        score = np.mean(roc_auc_score(y[val_idx], val_preds, average=None))
        print("AUC mean   {}".format(score))

        # if score >= best_score:
        torch.save(model_bert.state_dict(),os.path.join(args.ckpt_path, f"model_{fold}_{epoch}.bin"))
        best_score = score
