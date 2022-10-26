import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

def read_flo_file(filename):

    f = open(filename, 'rb')
    magic = np.fromfile(f, np.float32, count = 1)[0]
    w = np.fromfile(f, np.int32, count = 1)[0]
    h = np.fromfile(f, np.int32, count = 1)[0]
    
    data2d = np.fromfile(f, np.float32, count = 2 * w * h)
    data2d = np.resize(data2d, (h, w, 2))
    f.close()

    return data2d

class FlyingChairs():
    def __init__(self, data_root = './'):
        self.root = os.path.join(data_root, 'FlyingChairs_release','data/')
        self.len  = len([name for name in os.listdir(self.root) if os.path.isfile(os.path.join(self.root, name))]) / 3

    def __len__(self):
        return int(self.len)

    def __getitem__(self, index):

        ToTensor = transforms.ToTensor()

        index += 1
        img1 = Image.open(self.root + (str(index)).zfill(5) + '_img1.ppm')
        img2 = Image.open(self.root + (str(index)).zfill(5) + '_img2.ppm')
        flow = read_flo_file(self.root + (str(index)).zfill(5) + '_flow.flo')

        img1 = ToTensor(img1)
        img2 = ToTensor(img2)
        flow = ToTensor(flow)

        return torch.stack((img1, img2)), flow

if __name__ == '__main__':

    # Test DataLoader
    test = FlyingChairs('FlyingChairs')
    testset = torch.utils.data.DataLoader(test, batch_size = 12, shuffle=False, num_workers = 8)

    epochs = 5
    for epoch in range(epochs):
        for idx, (data, target) in enumerate(testset):

            print('data:',data.shape)
            print('target:',target.shape)

            img1 = data[:,0,:,:]
            img2 = data[:,1,:,:]

            print(img1)

            print('img1:',img1.shape)
            print('img2:',img2.shape)

            break
