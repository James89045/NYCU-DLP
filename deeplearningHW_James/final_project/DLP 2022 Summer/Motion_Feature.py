import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

def activation_func(activation):
    return  nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ['selu', nn.SELU(inplace=True)],
        ['none', nn.Identity()]
    ])[activation]


class ResidualBlock(nn.Module):
    def __init__(self, in_size, use_conv=False, out_size=256, hidden_size1=512, hidden_size2=1024):
        super(ResidualBlock, self).__init__()
        self.use_conv = use_conv
        self.conv1 = nn.Conv2d(in_size, hidden_size1, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_size1, hidden_size2, 3, padding=1)
        self.conv3 = nn.Conv2d(hidden_size2, out_size, 3, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(hidden_size1)
        self.batchnorm2 = nn.BatchNorm2d(hidden_size2)
        self.batchnorm3 = nn.BatchNorm2d(out_size)
        self.conv = nn.Conv2d(in_size, out_size, 1)
        self.linear = nn.Linear(64, out_size)

    def convblock(self, x):
        x = F.relu(self.batchnorm1(self.conv1(x)))
        x = F.relu(self.batchnorm2(self.conv2(x)))
        x = F.relu(self.batchnorm3(self.conv3(x)))
        return x
   
    """
    Combine output with the original input
    """
    def forward(self, x):
        if self.use_conv:
            return self.conv(x) + self.convblock(x) # skip connection
            # return self.linear(x) + self.convblock(x)
        return x + self.convblock(x)

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels

class motion_encoder(nn.Module):
    def __init__(self, in_size):
        super(motion_encoder, self).__init__()
        self.ResBlock = ResidualBlock(in_size)
    
    def forward(self, x):
        x = self.ResBlock(x)
        x = self.ResBlock(x)
        x = self.ResBlock(x)
        x = self.ResBlock(x)
        x = self.ResBlock(x)
        x = self.ResBlock(x)
        return x

class Motion_Feature(nn.Module):
    def __init__(self, h, w, batch_size, out_size=256):
        super(Motion_Feature, self).__init__()
        self.h = h
        self.w = w
        self.batch_size = batch_size
        self.ResBlock = nn.Sequential(
            ResidualBlock(h*w, use_conv=True),
            ResidualBlock(out_size),
            ResidualBlock(out_size),
            ResidualBlock(out_size),
            ResidualBlock(out_size),
            ResidualBlock(out_size),
        )
        # self.Embedding = nn.Embedding(256, out_size)
        self.embedding = nn.Linear(w, w)

    def forward(self, CV):
        matching_cost = CV.view(self.batch_size, -1, self.h, self.w)
        motion_feature = self.ResBlock(matching_cost)
        return self.embedding(motion_feature)


if __name__ == "__main__":

    h, w = 48, 64
    batch_size = 12
    input = torch.randn(batch_size, h, w, h, w)

    Net = Motion_Feature(h, w, batch_size)

    result = Net(input)

    print(result.shape)

