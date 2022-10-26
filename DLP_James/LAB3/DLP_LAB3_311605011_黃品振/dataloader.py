from pickle import NONE
from cv2 import waitKey
from matplotlib import transforms
import pandas as pd
from torch.utils import data
import numpy as np
import os
import torch
import PIL 
from torchvision import transforms 



def getData(mode):
    if mode == 'train':
        img = pd.read_csv('train_img.csv')
        label = pd.read_csv('train_label.csv')
        print(np.squeeze(img.values))
        return np.squeeze(img.values), np.squeeze(label.values)
    else:
        img = pd.read_csv('test_img.csv')
        label = pd.read_csv('test_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)


class RetinopathyLoader(data.Dataset):
    def __init__(self, root, mode, augmentation = NONE):
        """
        Args:
            root (string): Root path of the dataset.
            mode : Indicate procedure status(training or testing)

            self.img_name (string list): String list that store all image names.
            self.label (int or float list): Numerical list that store all ground truth label values.
        """
        self.root = root
        self.img_name, self.label = getData(mode)
        self.mode = mode
        trans = []
        #augmentation 是要擴增資料種類的list
        if augmentation == True:
            trans += augmentation
        trans.append(transforms.ToTensor())
        self.transform = transforms.Compose(trans)
        print("> Found %d images..." % (len(self.img_name)))

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)

    def __getitem__(self, index):        
        path = os.path.join(self.root, self.img_name[index] + '.jpeg')    
        label = self.label[index]
        img = PIL.Image.open(path)
        img = self.transform(img)
        return img, label
########
a = RetinopathyLoader(root = 'data', mode = 'train')
loader = data.DataLoader(a, batch_size=8)

for i in range(5):
    for idx, (img, label) in enumerate(loader):
        print('img: ', img.shape)
        print("label: ", label)

                     