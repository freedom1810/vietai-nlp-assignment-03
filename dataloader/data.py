import cv2
import torch
import numpy as np
import random
import sys


class VietAI3_Dataset(torch.utils.data.Dataset):
    def __init__(self, list_input_ids, list_label):
        self.list_input_ids = list_input_ids
        self.list_label = list_label

    def __len__(self):
        return len(self.list_input_ids)

    def __getitem__(self, idx):

        label = torch.tensor(self.list_label[idx], dtype=torch.long)
        input_ids = torch.tensor(self.list_input_ids[idx], dtype=torch.long)

        return input_ids, label