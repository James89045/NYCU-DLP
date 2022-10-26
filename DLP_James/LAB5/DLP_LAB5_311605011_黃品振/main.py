from cProfile import label
import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import copy
import json
import matplotlib.pyplot as plt

from evaluator import evaluation_model
from torchvision.utils import save_image

from dataloader import CLEVRDataset
from model import Generator, Discriminator
#----#
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.0002, type=float, help='learning rate')
    parser.add_argument('--z_dim', default=100, type=int, help='')
    parser.add_argument('--c_dim', default=300, type=int, help='')
    parser.add_argument('--epochs', default=300, type=int, help='')
    parser.add_argument('--batch_size', default=512, type=int, help='batch size')

    args = parser.parse_args()
    return args


def get_test_cond(path):
    
    
    with open(('dataset/objects.json'), 'r') as file:
        classes = json.load(file)
    with open(path,'r') as file:
        test_cond_list = json.load(file)

    labels = torch.zeros(len(test_cond_list), len(classes))
    for i in range(len(test_cond_list)):
        for cond in test_cond_list[i]:
            labels[i,int(classes[cond])]=1.

    return labels

def train(dataloader, g_model, d_model,args):
    
    criterion = nn.BCELoss()
    optimizer_G = torch.optim.Adam(g_model.parameters(), args.lr, betas=(0.5, 0.99))
    optimizer_D = torch.optim.Adam(d_model.parameters(), args.lr, betas=(0.5, 0.99))
    evaluator = evaluation_model()

    test_cond = get_test_cond(os.path.join('dataset', 'test.json')).to(device)
    fixed_z = torch.randn(len(test_cond), args.z_dim).to(device)
    best_score = 0
    loss_g_list = []
    loss_d_list = []
    score_list = []
    for epoch in range(1,1 + args.epochs):
        total_loss_g=0
        total_loss_d=0
        for i,(images,cond) in enumerate(dataloader):
            g_model.train()
            d_model.train()
            batch_size=len(images)
            images = images.to(device)
            cond = cond.to(device)

            real = torch.ones(batch_size).to(device)
            fake = torch.zeros(batch_size).to(device)
            """
            train discriminator
            """
            optimizer_D.zero_grad()

            # for fake images
            z = torch.randn(batch_size, args.z_dim).to(device)
            gen_imgs = g_model(z, cond)
            pred = d_model(gen_imgs.detach(), cond)
            loss_fake = criterion(pred, fake)


            # for real images
            pred = d_model(images, cond)
            loss_real = criterion(pred, real)
            
            loss_d = loss_fake + loss_real
            loss_d.backward()
            optimizer_D.step()

            """
            train generator
            """
            for _ in range(4):
                optimizer_G.zero_grad()

                z = torch.randn(batch_size,args.z_dim).to(device)
                gen_imgs = g_model(z, cond)
                predicts = d_model(gen_imgs,cond)
                loss_g = criterion(predicts,real)

                loss_g.backward()
                optimizer_G.step()
            
            
            print(f'epoch{epoch} {i}/{len(dataloader)}  loss_g: {loss_g.item():.3f}  loss_d: {loss_d.item():.3f}')
            total_loss_g += loss_g.item()
            total_loss_d += loss_d.item()

        # add element    
        loss_d_list.append(loss_d/ len(dataloader))
        loss_g_list.append(loss_g/ len(dataloader))


        # evaluate
        g_model.eval()
        d_model.eval()
        with torch.no_grad():
            gen_imgs = g_model(fixed_z, test_cond)
        score = evaluator.eval(gen_imgs, test_cond)
        if score>best_score:
            best_score=score
            best_g_model_wts = copy.deepcopy(g_model.state_dict())
            best_d_model_wts = copy.deepcopy(d_model.state_dict())
            torch.save(best_g_model_wts, os.path.join('models/generator', f'epoch{epoch}_score{score:.2f}.pt'))
            torch.save(best_d_model_wts, os.path.join('models/discriminator', f'epoch{epoch}_score{score:.2f}.pt'))
        score_list.append(score)
        print(f'avg loss_g: {total_loss_g/ len(dataloader):.3f}  avg_loss_d: {total_loss_d/ len(dataloader):.3f}')
        print(f'testing score: {score:.2f}')
        print('---------------------------------------------')
        save_image(gen_imgs, os.path.join('results', f'epoch{epoch}.png'), nrow=8, normalize=True)
    print(f"max score: {max(score_list):.3f}")
    
    
    # plot
    score_list = torch.Tensor(score_list)
    loss_d_list = torch.Tensor(loss_d_list)
    loss_g_list = torch.Tensor(loss_g_list)

    plt.plot(np.arange(1, args.epochs+1), score_list.cpu().numpy(), label = "Accuracy")
    plt.plot(np.arange(1, args.epochs+1), loss_g_list.cpu().numpy(), label = "Generator loss")
    plt.plot(np.arange(1, args.epochs+1), loss_d_list.cpu().numpy(), label = "Discriminator loss")
    plt.legend(loc = "lower right")
    plt.xlabel('Epochs')
    plt.ylabel("Loss")
    plt.savefig("Loss per epoch")
    plt.show()



if __name__=='__main__':
    args = parse_args()
    image_shape=(3, 64, 64)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #training data
    dataset_train = CLEVRDataset(img_path='iclevr', json_path = 'dataset/train.json')
    loader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=2)

    generator=Generator(args.z_dim, args.c_dim).to(device)
    discrimiator=Discriminator(image_shape, args.c_dim).to(device)
    generator.weight_init(mean=0, std=0.02)
    discrimiator.weight_init(mean=0, std=0.02)

    # train
    train(loader_train, generator, discrimiator, args)
