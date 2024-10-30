import torch
import torch.nn as nn
from torch.nn import functional as F


class ImgNet(nn.Module):
    def __init__(self, code_len):
        super(ImgNet, self).__init__()
        self.classifier = nn.Linear(in_features=512, out_features=code_len)
        self.classifier.weight.data = torch.randn(code_len, 512) * 0.01
        self.classifier.bias.data = torch.randn(code_len) * 0.01

    def forward(self, x):
        x = self.classifier(x)
        return x


class TxtNet(nn.Module):
    def __init__(self, code_len):
        super(TxtNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 8192, kernel_size=(512, 1), stride=(1, 1))
        self.conv2 = nn.Conv2d(8192, code_len, kernel_size=1, stride=(1, 1))
        self.apply(self.weights_init)

    def weights_init(self, m):
        if type(m) == nn.Conv2d:
            nn.init.normal_(m.weight.data, 0.0, 0.01)
            nn.init.normal_(m.bias.data, 0.0, 0.01)

    def forward(self, x):
        x = x.unsqueeze(1).unsqueeze(-1)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = x.squeeze(-1).squeeze(-1)
        return x