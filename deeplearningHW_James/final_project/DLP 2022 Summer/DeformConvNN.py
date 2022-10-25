import torch
import torch.nn as nn
import torchvision

class DeformConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size = 3, stride = 1, padding = 1 ):
        super(DeformConv, self).__init__()
        
        self.conv   = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding) 
        self.offset = nn.Conv2d(in_channel, 2*kernel_size*kernel_size, kernel_size, stride, padding)
        self.mask   = nn.Sequential(
            nn.Conv2d(in_channel, kernel_size*kernel_size, kernel_size, stride, padding),
            nn.Sigmoid()
        )

    def forward(self, x):
        offset = self.offset(x)
        mask   = self.mask(x)
        output = torchvision.ops.deform_conv2d(
            input   = x, 
            offset  = offset, 
            weight  = self.conv.weight, 
            mask    = mask, 
            padding = (1,1)
        )

        return output

class ResidualBlock(nn.Module):
    def __init__(self, in_channel):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Sequential(
            DeformConv(in_channel, 64),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            DeformConv(64, in_channel),
            nn.ReLU()
        )

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        output = c2 + x

        return output

class NN(nn.Module):
    def __init__(self, in_channel):
        super(NN, self).__init__()

        self.res1 = ResidualBlock(in_channel)

        self.trans = nn.Conv2d(in_channel, 256, 3, padding = 1)

        self.res2 = ResidualBlock(256)

    def forward(self, x):
        
        x = self.res1(x)
        x = self.trans(x)
        x = self.res2(x)

        return x

if __name__ == '__main__':
    test_net = NN(128)

    input = torch.rand(12, 128, 384, 512)

    print('input:',input.shape)
    test  = test_net(input)
    print('test:',test.shape)