from torch.utils.data import Dataset
import os
import json
from PIL import Image
from torchvision import transforms
import torch


class CLEVRDataset(Dataset):
    def __init__(self, img_path, json_path):
        """
        :param img_path: file of training images
        :param json_path: train.json
        """
        self.img_path=img_path
        with open(('dataset/objects.json'), 'r') as file:
            self.classes = json.load(file)
        self.numclasses = len(self.classes)
        self.img_names = []
        self.img_cond = []
        with open(json_path,'r') as file:
            dict = json.load(file)
            for img_name, img_cond in dict.items():
                self.img_names.append(img_name)
                #將每個data裡面有的情形存成一個list，例如[1,5,6]，即為有1、5、6種情形的圖片。再根據大data有的圖片數量存成一個大list
                self.img_cond.append([self.classes[cond] for cond in img_cond])
        self.transform = transforms.Compose([transforms.Resize((64, 64)),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_path, self.img_names[index])).convert('RGB')
        img = self.transform(img)
        cond = self.get_onehot(self.img_cond[index])
        return img,cond

    def get_onehot(self, int_list):
        onehot = torch.zeros(self.numclasses)
        for i in int_list:
            onehot[i] = 1.
        return onehot