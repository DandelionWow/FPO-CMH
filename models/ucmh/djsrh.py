import torch
import torch.nn as nn
import torch.nn.functional as F

import os.path as osp

class ImgNet(nn.Module):
    def __init__(self, code_len):
        super(ImgNet, self).__init__()
        self.fc_encode = nn.Linear(512, code_len)
        self.alpha = 1.0
    
    def forward(self, x):
        feat = x
        hid = self.fc_encode(feat)
        code = torch.tanh(self.alpha * hid)
        return code
class TxtNet(nn.Module):
    def __init__(self, code_len):
        super(TxtNet, self).__init__()
        self.fc1 = nn.Linear(512, 4096)
        self.fc2 = nn.Linear(4096, code_len)
        self.alpha = 1.0

    def forward(self, x):
        feat = F.relu(self.fc1(x))
        hid = self.fc2(feat)
        code = torch.tanh(self.alpha * hid)
        return code
