from pickle import NONE
from os import read
from pickletools import optimize
import dataloader
from dataloader import read_bci_data
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch.utils.data as Data
from torchsummary import summary


activations = {"ELU": nn.ELU(), "LeakyReLU": nn.LeakyReLU(), "ReLu": nn.ReLU()}


#data processing
train_data, train_label, test_data, test_label = read_bci_data()
trainingdataset = TensorDataset(torch.from_numpy(train_data), torch.from_numpy(train_label))
loader_train = DataLoader(dataset = trainingdataset, batch_size = 128, shuffle = True)
testdataset =  TensorDataset(torch.from_numpy(test_data), torch.from_numpy(test_label))
loader_test = DataLoader(dataset = testdataset, batch_size = 128, shuffle = True)


#def network
class EEGNET(nn.Module):
    def __init__(self, activation = nn.ELU(), dropout = 0.25):
        super(EEGNET, self).__init__()


        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size = (1, 51), stride = (1, 1), padding = (0, 25), bias = True),
            nn.BatchNorm2d(16)
        )


        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size = (2, 1), stride = (1, 1), groups = 16, bias = True),
            nn.BatchNorm2d(32),
            activation,
            nn.AvgPool2d(kernel_size = (1, 4), stride = (1, 4), padding = 0),
            nn.Dropout(p = dropout)
        )


        self.separableConv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size = (1, 15), stride = (1, 1), padding = (0, 7), bias = True),
            nn.BatchNorm2d(32),
            activation,
            nn.AvgPool2d(kernel_size = (1, 8), stride = (1, 8), padding = 0),
            nn.Dropout(p = dropout)
        )


        self.classify = nn.Linear(in_features = 736, out_features = 2, bias = True)

    

    def forward(self, x):
        output = self.firstconv(x)
        output = self.depthwiseConv(output)
        output = self.separableConv(output)
        #攤平:
        output = output.view(output.shape[0], -1)
        output = self.classify(output)

        return output


class DeepConvNet(nn.Module):
    def __init__(self, activation = nn.ELU(), filters = [25, 50, 100, 200], dropout = 0.5):
        super(DeepConvNet, self).__init__()


        filters = filters
        self.conv0 = nn.Sequential(
            nn.Conv2d(1, filters[0], kernel_size = (1, 5), stride = (1, 1),padding = (0, 0), bias = True),
            nn.Conv2d(filters[0], filters[0], kernel_size = (2, 1), padding = (0, 0), bias = True),
            nn.BatchNorm2d(filters[0]),
            activation,
            nn.MaxPool2d(kernel_size = (1, 2)),
            nn.Dropout(p = dropout)
        )


        for i in range(1, len(filters)):
            setattr(self, 'conv'+str(i), nn.Sequential(
                nn.Conv2d(filters[i-1], filters[i], kernel_size = (1, 5), stride = (1, 1), padding = (0, 0), bias = True),
                nn.BatchNorm2d(filters[i]),
                activation,
                nn.MaxPool2d(kernel_size = (1, 2)),
                nn.Dropout(p = dropout)
            ))


        self.classify = nn.Linear(8600,2)

          
    def forward(self, x):
        out = self.conv0(x)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out.view(out.shape[0], -1)
        out = self.classify(out)

        return out


def train(loader_train, loader_test, activation, lr, epochs, model =EEGNET):
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    print("running on ", device)
    loss = nn.CrossEntropyLoss()
    model = model(activation)
    model = model.to(device)
    train_acc = []
    test_acc = []
    for epoch in range(1, 1+epochs):
        model.train()
        acc_train = 0
        acc_test = 0
        for idx, (trains, labels) in enumerate(loader_train):
            trains = trains.to(device, dtype = torch.float)
            labels = labels.to(device, dtype = torch.long)
            optimizer = torch.optim.Adagrad(model.parameters(), lr = lr)
            predict = model(trains)
            label = torch.reshape(labels, (predict.shape[0], 1))
            Loss_train = loss(predict, labels)
            predict = predict.argmax(dim=1,keepdim=True)
            acc_train += len(predict[predict == label])

                #update weight
            optimizer.zero_grad()
            Loss_train.backward()
            optimizer.step()
        model.eval()
        x = (acc_train/1080)*100
        #存取training正確率
        train_acc.append(x)

        
        for idx, (tests, labels) in enumerate(loader_test):
            tests = tests.to(device, dtype = torch.float)
            labels = labels.to(device, dtype = torch.long)
            pred = model(tests)
            label = torch.reshape(labels, (pred.shape[0], 1))
            Loss_test = loss(pred, labels)
            pred = pred.argmax(dim=1,keepdim=True)
            acc_test += len(pred[pred == label])
        y = (acc_test/1080)*100
        test_acc.append(y)


        if epoch % 10 == 0:
            print(f'train_accuracy : {x:.2f}%  test_accuracy : {y:.2f}%')
   
    #print("Highest accuracy : ", max(test_acc))
    return train_acc, test_acc, max(test_acc), epochs
             

def EEGNET_plot():            
    ELU_train_acc, ELU_test_acc , max_ELU_acc, epochs= train(loader_train, loader_test, nn.ELU(), 0.01, 150, EEGNET)
    ReLU_train_acc, ReLU_test_acc ,max_ReLU_acc, epochs= train(loader_train, loader_test, nn.ReLU(), 0.01, 150, EEGNET)
    LeakyReLU_train_acc, LeakyReLU_test_acc , max_LeakyReLU_acc, epochs= train(loader_train, loader_test, nn.LeakyReLU(), 0.01, 150, EEGNET)
    plt.plot(np.arange(1, epochs+1), ELU_train_acc, label = "ELU_train")
    plt.plot(np.arange(1, epochs+1), ReLU_train_acc, label = "ReLU_train")
    plt.plot(np.arange(1 ,epochs+1), LeakyReLU_train_acc, label = "LeakyReLU_train")
    plt.plot(np.arange(1 ,epochs+1), ELU_test_acc, label = "ELU_test")
    plt.plot(np.arange(1 ,epochs+1), ReLU_test_acc, label = "ReLU_test")
    plt.plot(np.arange(1 ,epochs+1), LeakyReLU_test_acc, label = "LeakyReLU_test")
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel("Acuuracy")
    plt.title("Activation function comparison(EEGNET)")
    plt.show()
    print("EEGNET:")
    print(f"MaxELUaccuracy: {max_ELU_acc:.2f}")
    print(f"MaxReLUaccuracy: {max_ReLU_acc:.2f}")
    print(f"MaxLeakyReLUaccuracy: {max_LeakyReLU_acc:.2f}")
    print(f"Highest accuracy:{max(max_ELU_acc, max_ReLU_acc, max_LeakyReLU_acc):.2f}%")


def DeepConvNet_plot():    
    ELU_train_acc, ELU_test_acc , max_ELU_acc, epochs= train(loader_train, loader_test, nn.ELU(), 0.01, 150, DeepConvNet)
    ReLU_train_acc, ReLU_test_acc ,max_ReLU_acc, epochs= train(loader_train, loader_test, nn.ReLU(), 0.01, 150, DeepConvNet)
    LeakyReLU_train_acc, LeakyReLU_test_acc , max_LeakyReLU_acc, epochs= train(loader_train, loader_test, nn.LeakyReLU(), 0.01, 150, DeepConvNet)
    plt.plot(np.arange(1, epochs+1), ELU_train_acc, label = "ELU_train")
    plt.plot(np.arange(1, epochs+1), ReLU_train_acc, label = "ReLU_train")
    plt.plot(np.arange(1 ,epochs+1), LeakyReLU_train_acc, label = "LeakyReLU_train")
    plt.plot(np.arange(1 ,epochs+1), ELU_test_acc, label = "ELU_test")
    plt.plot(np.arange(1 ,epochs+1), ReLU_test_acc, label = "ReLU_test")
    plt.plot(np.arange(1 ,epochs+1), LeakyReLU_test_acc, label = "LeakyReLU_test")
    plt.hlines(y = 87.0, xmin = 1, xmax = epochs)
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel("Acuuracy")
    plt.title("Activation function comparison(DeepConvNet)")
    plt.show()
    print("DeepConvNet:")
    print(f"MaxELUaccuracy: {max_ELU_acc:.2f}")
    print(f"MaxReLUaccuracy: {max_ReLU_acc:.2f}")
    print(f"MaxLeakyReLUaccuracy: {max_LeakyReLU_acc:.2f}")
    print(f"Highest accuracy: {max(max_ELU_acc, max_ReLU_acc, max_LeakyReLU_acc):.2f}%")


DeepConvNet_plot()








