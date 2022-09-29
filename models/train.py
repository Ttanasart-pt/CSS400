#from homebrew.CNN import Model
#from mobilenet.model import Model
#from hourglass.model import Model
from Zimmerman.handSegNet import Model

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchsummary import summary

from tqdm import tqdm
from dataloader.CMUpanoptic import CMUDataset, CMUHeatmapDataset, CMUBBoxDataset, CMUHandSegment
from util.checkpoint import save_checkpoint, load_checkpoint

import yaml
import argparse

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

    imageSize = CONFIG['input_size']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if(CONFIG['output_type'] == "keypoints"):
        dataset = CMUDataset(CONFIG['dataset'], imageSize)
    elif(CONFIG['output_type'] == "heatmap"):
        dataset = CMUHeatmapDataset(CONFIG['dataset'], imageSize)
    elif(CONFIG['output_type'] == "bbox"):
        dataset = CMUBBoxDataset(CONFIG['dataset'], imageSize)
    elif(CONFIG['output_type'] == "segment"):
        dataset = CMUHandSegment(CONFIG['dataset'], imageSize)
        
    model = Model().to(device)
        
    if(load_model):
        load_checkpoint(checkpoint_path, model)
        
    summary(model, (3, imageSize, imageSize))
    dataloader = DataLoader(dataset, 
                        batch_size = batch_size,
                        shuffle = True, 
                        num_workers = 2, 
                        pin_memory = True)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    min_loss = None
    
    for e in range(num_epoch):
        print(f'Epoch {e}')
        acc_loss = 0
        for imgs, labels in tqdm(dataloader):
            imgs = imgs.type(torch.FloatTensor).to(device)
            labels = labels.type(torch.FloatTensor).to(device)
            
            predicted = model(imgs)
            loss = criterion(predicted, labels)
            acc_loss += loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        avg_loss = (acc_loss / len(dataloader))
        print(f"Average loss = {avg_loss:.5f}")
        if save_model and (min_loss is None or avg_loss < min_loss):
            min_loss = avg_loss
            save_checkpoint(checkpoint_path, model)
    
if __name__ == "__main__":
    main()
    
# example uses: python .\train.py --config "homebrew/train.yaml"