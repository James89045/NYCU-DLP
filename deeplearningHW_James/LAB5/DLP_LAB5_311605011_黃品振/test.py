import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import copy
import json

from evaluator import evaluation_model
from torchvision.utils import save_image

from model import Generator



def get_test_cond(path):
   

    with open(os.path.join('dataset', 'objects.json'), 'r') as file:
        classes = json.load(file)
    with open(path,'r') as file:
        test_cond_list=json.load(file)

    labels=torch.zeros(len(test_cond_list),len(classes))
    #讀取到n個圖片的情況，得知要生成n張圖片，回傳tensor為(n,24)
    for i in range(len(test_cond_list)):
        for cond in test_cond_list[i]:
            labels[i,int(classes[cond])]=1.

    return labels

def test(g_model,z_dim,epochs):
    """
    :param z_dim: 100
    """
    model_evaluator=evaluation_model()

    new_test_cond=get_test_cond(os.path.join('dataset','test.json')).to(device)
    
    for epoch in range(epochs):

	    g_model.eval()
	    fixed_z = torch.randn(len(new_test_cond), z_dim).to(device)
	    with torch.no_grad():
	        gen_imgs=g_model(fixed_z, new_test_cond)
	    score=model_evaluator.eval(gen_imgs, new_test_cond)
	    print(f'testing score: {score:.2f}')
	    # savefig
	    save_image(gen_imgs, os.path.join('test_results', f'epoch{epoch}.png'), nrow=8, normalize=True)

if __name__=='__main__':
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    generator=Generator(100, 300).to(device)
    generator.load_state_dict(torch.load('models/generator/epoch168_score0.65.pt'))

    # test
    test(generator, 100, 20)
