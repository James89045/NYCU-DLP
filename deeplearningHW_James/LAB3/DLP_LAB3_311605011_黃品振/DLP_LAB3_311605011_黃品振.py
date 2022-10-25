from cProfile import label
from pickle import NONE
from dataloader import RetinopathyLoader
import numpy as np 

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.utils.data as Data
from torchsummary import summary
from torchvision import models 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd




def Resnet(type = 18, class_num = 5, pretrained = False):
    if type ==18:
        model = models.resnet18(pretrained)
        model.fc = nn.Linear(model.fc.in_features, class_num)
    
    if type ==50:
        model = models.resnet50(pretrained)
        model.fc = nn.Linear(model.fc.in_features, class_num)
    
    return model


def train(model, model_name, loader_train, loader_test, epochs, lr):
    model = model 
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    print("running on ", device)
    model = model.to(device)
    Loss = nn.CrossEntropyLoss()
    train_acc = []
    test_acc = []
    loss_list = []
    """
    start training

    """
    for epoch in range(1, epochs+1):
        model.train()
        correct_train = 0
        total_loss = 0
        with torch.set_grad_enabled(True):
            for idx, (trains, labels) in enumerate(loader_train):
                torch.cuda.empty_cache()
                trains = trains.to(device, dtype = torch.float)
                labels = labels.to(device, dtype = torch.long)
                optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum = 0.9, weight_decay = 5e-4)
                pred = model(trains)
                #labels = torch.reshape(labels, (pred.shape[0], 1))
                #labels = labels.squeeze(1)
                #print(labels)
                loss = Loss(pred, labels)
                total_loss += loss
                pred = pred.argmax(dim=1,keepdim=True)
                pred = pred.squeeze(1)
                print(pred)
                correct_train += len(pred[pred == labels])
                print("correct training number: ", correct_train)
                print(f"each batch acc: {(100*len(pred[pred == labels])/len(labels))}%")

                #update weights

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        torch.save(model.state_dict(), model_name + 'save.pt')
        acc_element = (100*(correct_train/len(loader_train.dataset)))
        total_loss = (total_loss/len(loader_train))
        train_acc.append(acc_element)
        loss_list.append(total_loss)
        
        #Run test data

        test_acc_element = (100* test_evaluate(model, loader_test)/len(loader_test.dataset))
        test_acc.append(test_acc_element)
        print(f"accuracy: {acc_element : .2f}% loss: {total_loss: .5f}")
        print(train_acc)
        print(f"test accuracy: {test_acc_element : .2f}%")
        print(test_acc)
    return train_acc, test_acc, loss_list


def test_evaluate(model, loader_test):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    correct_test = 0
    for idx, (tests, test_labels) in enumerate(loader_test):
        torch.cuda.empty_cache()
        tests = tests.to(device, dtype = torch.float)
        test_labels = test_labels.to(device, dtype = torch.float)
        pred_test = model(tests)
        pred_test = pred_test.argmax(dim=1,keepdim=True)
        pred_test = pred_test.squeeze(1)
        correct_test += len(pred_test[pred_test == test_labels])
        print("correct testing number: ", correct_test)
        print(f"each batch test acc:{100*len(pred_test[pred_test == test_labels])/len(test_labels)}")

        
    return correct_test


def plot_acc(epochs, train_acc, test_acc, pre_train_acc, pre_test_acc, model_name):
    plt.plot(np.arange(1, epochs+1), train_acc, label = "Train(without pretraining)")
    plt.plot(np.arange(1, epochs+1), test_acc, label = "Test(without pretraining)")
    plt.plot(np.arange(1, epochs+1), pre_train_acc, label = "Train(with pretraining)")
    plt.plot(np.arange(1, epochs+1), pre_test_acc, label = "Test(with pretraining)")
    plt.legend(loc = "lower right")
    plt.xlabel('Epochs')
    plt.ylabel("Acuuracy(%)")
    plt.title("Result comparison" + model_name)
    plt.show()
    print(f"Highest testing accuracy(without pretraining): {max(test_acc) : .2f}%")
    print(f"Highest testing accuracy(with pretraining): {max(pre_test_acc) : .2f}%")



def plot_confusion_matrix(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    cf_matrix = confusion_matrix(y_true, y_pred, normalize = 'true')
    class_names = ['No DR', 'Mild', 'Moderate', 'severe', 'Proliferative DR']
    df_cm = pd.DataFrame(cf_matrix, class_names, class_names)    
    plt.figure(figsize = (9,6))
    sns.heatmap(df_cm, annot=True, fmt=".3g", cmap=plt.cm.Blues)
    plt.xlabel("predicted label")
    plt.ylabel("True lable")
    plt.title('Normolized confusion matrix')
    plt.show()


def run_test_cfmatrix(model, model_name, loader_test):
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    model.to(device)
    model.eval()
    correct_test = 0
    y_true = []
    y_pred = []
    for idx, (tests, test_labels) in enumerate(loader_test):
        tests = tests.to(device, dtype = torch.float)
        test_labels = test_labels.to(device, dtype = torch.float)
        model.load_state_dict(torch.load(model_name + 'save.pt'))
        pred = model(tests)
        pred = pred.argmax(dim=1,keepdim=True)
        pred = pred.squeeze(1)
        correct_test += len(pred[pred == test_labels])
        pred_list = pred.tolist()
        test_labels_list = test_labels.tolist()
        y_pred += pred_list
        y_true += test_labels_list
    
    plot_confusion_matrix(y_true, y_pred)
    print("test accuracy" + '('+model_name+')' + f': {100*correct_test/len(loader_test.dataset)}%')

#data for resnet18
train_data18 = RetinopathyLoader('data', 'train',)
loader_train18 = DataLoader(train_data18, batch_size = 32, shuffle = True)
test_data18 = RetinopathyLoader('data', 'test')
loader_test18 = DataLoader(test_data18, batch_size = 32, shuffle = True)

#Resnet18
#without pretraining

#train_acc, test_acc, loss_list = train(Resnet(18, 5, False), "Resnet18_nopretrain", loader_train18, loader_test18, 10, 0.001)
#plot matrix
#run_test_cfmatrix(Resnet(18, 5, False), "Resnet18_nopretrain", loader_test18)
#pretraining
#pre_train_acc, pre_test_acc, pre_loss_list = train(Resnet(18, 5, pretrained=True), "Resnet18_pretrain", loader_train18, loader_test18, 10, 0.001)
#plot matrix
#run_test_cfmatrix(Resnet(18, 5, True), "Resnet18_pretrain", loader_test18)
#plot ResNet18
#plot_acc(10, train_acc, test_acc, pre_train_acc, pre_test_acc, "ResNet18")


#data for resnet50
train_data50 = RetinopathyLoader('data', 'train',)
loader_train50 = DataLoader(train_data50, batch_size = 4, shuffle = True)
test_data50 = RetinopathyLoader('data', 'test')
loader_test50 = DataLoader(test_data50, batch_size = 4, shuffle = True)


#Resnet50    
#without pretraining
#train_acc, test_acc, loss_list = train(Resnet(50, 5, False), "Resnet50_nopretrain", loader_train50, loader_test50, 5, 0.001)
#pretraining
#pre_train_acc, pre_test_acc, pre_loss_list = train(Resnet(50, 5, pretrained=True), "Resnet50_pretrain", loader_train50, loader_test50, 5, 0.001)
#plot nopretrain matrix
#run_test_cfmatrix(Resnet(50, 5, False), "Resnet50_nopretrain", loader_test50)
#plot pretrain matrix
run_test_cfmatrix(Resnet(50, 5, True), "Resnet50_pretrain", loader_test50)
#plot Resnet50
#plot_acc(5, train_acc, test_acc, pre_train_acc, pre_test_acc, "ResNet50")
