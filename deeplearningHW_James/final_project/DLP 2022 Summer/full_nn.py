import torch
import torch.nn as nn
import torch.nn.functional as F
from feature import feature as extractor
from DeformConvNN import NN as deform_comv
from Motion_Feature import ResidualBlock
from Motion_Feature import Motion_Feature as motion

def upflow8(flow, mode='bilinear'):
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    return  8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)




class KPA(nn.Module):
    def __init__(self, args):
        super(KPA, self).__init__()

        self.extractor = extractor(args)
        self.motion = motion( args.H, args.W, args.batch)
        self.deform_comv = deform_comv(256)

        self.decoder = nn.Sequential(
            ResidualBlock(in_size = 256),
            ResidualBlock(in_size = 256, out_size = 2, use_conv = True)
        )
        

    def forward(self, x):

        img1 = x[:,0,:,:]
        img2 = x[:,1,:,:]

        corr, context = self.extractor(img1, img2)
        # print('corr:',corr.shape)
        # print('context:',context.shape)

        feature = self.motion(corr)
        attention = self.deform_comv(context)
        # print('feature:',feature.shape)
        # print('attention:',attention.shape)

        updated_feature = feature + torch.multiply(feature, attention)
        # print('updated_feature:',updated_feature.shape)

        flow = self.decoder(updated_feature)
        # print('flow:',flow.shape)

        flow = upflow8(flow)
        # print('flow:',flow.shape)

        return flow
        

if __name__ == '__main__':

    import argparse
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=12)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--shuffle', action='store_false')
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--H', type=int, default=48)
    parser.add_argument('--W', type=int, default=64)
    args = parser.parse_args()

    test_net = KPA(args)

    input = torch.rand(args.batch, 2, 3, 384, 512)

    print('input:',input.shape)
    test_net(input)


        