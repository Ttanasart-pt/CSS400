from cProfile import label
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchsummary import summary

from tqdm import tqdm
from util.checkpoint import save_checkpoint, load_checkpoint
import modelLoader

import os
import yaml
import argparse
import math
import json
import time

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--config', help='Path to config file.')

args = parser.parse_args()
with open(args.config, 'r') as f:
    CONFIG = yaml.safe_load(f.read())
    
def main():
    num_epoch = CONFIG['epoch']
    learning_rate = CONFIG['learning_rate']
    batch_size = CONFIG['batch_size']

    checkpoint_path = CONFIG['checkpoint']
    save_model = CONFIG['save_model']
    load_model = CONFIG['load_model']
    
    report_path = CONFIG['report']
    if not os.path.isdir(report_path):
        os.mkdir(report_path)

    imageSize = CONFIG['input_size']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if(CONFIG['output_type'] == "keypoints"):
        from dataloader.CMUpanoptic import CMUDataset
        dataset = CMUDataset(CONFIG['dataset'], imageSize)
    elif(CONFIG['output_type'] == "heatmap"):
        from dataloader.CMUpanoptic import CMUHeatmapDataset
        dataset = CMUHeatmapDataset(CONFIG['dataset'], imageSize)
    elif(CONFIG['output_type'] == "bbox"):
        from dataloader.CMUpanoptic import CMUBBoxDataset
        dataset = CMUBBoxDataset(CONFIG['dataset'], imageSize)
    elif(CONFIG['output_type'] == "segment"):
        from dataloader.CMUpanoptic import CMUHandSegment
        dataset = CMUHandSegment(CONFIG['dataset'], imageSize)
    elif(CONFIG['output_type'] == "segment_RHD"):
        from dataloader.RHD import RHDSegment
        dataset = RHDSegment(CONFIG['dataset'], imageSize)
        
    model = modelLoader.loadFromConfig(CONFIG).to(device)
        
    if(load_model):
        load_checkpoint(checkpoint_path, model)
        
    summary(model, (3, imageSize, imageSize))
    dataloader = DataLoader(dataset, 
                        batch_size = batch_size,
                        shuffle = True, 
                        num_workers = 2, 
                        pin_memory = True)
    
    if("segment" in CONFIG['output_type']):
        loss = nn.CrossEntropyLoss()
    else:
        loss = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    
    loss_rec = 100
    losses = []
    min_loss = None
    train_time_start = time.time()
    
    for e in range(num_epoch):
        print(f'Epoch {e}')
        acc_loss = 0
        avd_n = 0
        loader = tqdm(dataloader)
        for imgs, labels in loader:
            imgs = imgs.type(torch.FloatTensor).to(device)
            labels = labels.type(torch.FloatTensor).to(device)
            predicted = model(imgs).to(device)
            avd_n += 1
            
            if("segment" in CONFIG['output_type']):
                labels = labels.type(torch.LongTensor).to(device)
                
                loss = loss(predicted, labels)
                acc_loss += loss.item()
            else:
                loss = loss(predicted, labels)
                acc_loss += loss.item()
            
            loader.set_description(f'Loss: {(acc_loss / avd_n):.4e}')
            
            if(avd_n % loss_rec == 0 and acc_loss > 0):
                losses.append(math.log10(acc_loss / avd_n))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        avg_loss = (acc_loss / avd_n)
        if(avg_loss > 0):
            losses.append(math.log10(avg_loss))
        print(f"Average loss = {avg_loss:.5e}")
        if save_model and (min_loss is None or avg_loss < min_loss):
            min_loss = avg_loss
            save_checkpoint(checkpoint_path, model)
    
    plt.plot(losses)
    plt.savefig(report_path + "losses.png")
    
    report = {
        'min_loss': min_loss,
        'duration': time.time() - train_time_start
    }
    
    with open(report_path + "report.json", 'w') as f:
        json.dump(report, f)
    
if __name__ == "__main__":
    main()
    
# example uses: python .\train.py --config "homebrew/train.yaml"