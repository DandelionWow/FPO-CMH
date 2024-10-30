import math
import torch
import torch.nn as nn


class ImgNet(nn.Module):
    def __init__(self, code_len, ima_feat_len=512):
        super(ImgNet, self).__init__()
        self.fc1 = nn.Linear(512, 4096)
        self.fc_encode = nn.Linear(4096, code_len)


        self.alpha = 1.0
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        x = x.view(x.size(0), -1)

        feat1 = self.relu(self.fc1(x))
        hid = self.fc_encode(self.dropout(feat1))
        code = torch.tanh(self.alpha * hid)

        return code

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)


class TxtNet(nn.Module):
    def __init__(self, code_len, txt_feat_len=512):
        super(TxtNet, self).__init__()
        self.fc1 = nn.Linear(txt_feat_len, 4096)
        self.fc_encode = nn.Linear(4096, code_len)

        self.alpha = 1.0
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(inplace=True)

        torch.nn.init.normal_(self.fc_encode.weight, mean=0.0, std=0.3)

    def forward(self, x):

        feat = self.relu(self.fc1(x))
        hid = self.fc_encode(self.dropout(feat))
        code = torch.tanh(self.alpha * hid)

        return code

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)