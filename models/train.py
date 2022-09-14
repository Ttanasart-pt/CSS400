from homebrew.CNN import Model

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchsummary import summary

from tqdm import tqdm
from dataloader.CMUpanoptic import CMUDataset
from util.checkpoint import save_checkpoint, load_checkpoint
    
def main():
    num_epoch = 10
    learning_rate = 4e-2
    batch_size = 1

    checkpoint_path = "homebrew/checkpoints/test.chk"
    save_model = True
    load_model = True

    imageSize = 128
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = CMUDataset("E:\\HandPose\\CMU\\hand_syn\\synth1", imageSize)

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
    
    for e in range(num_epoch):
        print(f'Epoch {e}')
        acc_loss = 0
        for imgs, labels in tqdm(dataloader):
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            predicted = model(imgs)
            
            loss = criterion(predicted, labels)
            acc_loss += loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Average loss = {(acc_loss / len(dataloader)):.2f}")
        
    if(save_model):
        save_checkpoint(checkpoint_path, model)
    
if __name__ == "__main__":
    main()