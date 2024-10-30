import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class ImgNet(nn.Module):
    def __init__(self, code_len):
        super(ImgNet, self).__init__()
        self.hash_layer = nn.Linear(512, code_len)
        self.alpha = 1.0

    def forward(self, x):
        feat = x

        hid = self.hash_layer(feat)
        code = torch.tanh(self.alpha * hid)

        return code

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)


class TxtNet(nn.Module):
    def __init__(self, code_len, txt_feat_len=512):
        super(TxtNet, self).__init__()

        self.net = nn.Sequential(nn.Linear(txt_feat_len, 4096),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(4096, code_len),
                                 )

        self.alpha = 1.0

    def forward(self, x):
        hid = self.net(x)
        code = torch.tanh(self.alpha * hid)
        return code

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)
