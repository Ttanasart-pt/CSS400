import os
import torch
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import segmentation_models_pytorch as smp

from pprint import pprint
from torch.utils.data import DataLoader

from dataloader.RHD import RHDSegment
from handSegment.model import PetModel

def main():
    train_dataset = RHDSegment("E:/Dataset/RHD_published_v2/training/", 256)

    print(f"Train size: {len(train_dataset)}")

    n_cpu = 1
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=n_cpu)

    model = PetModel("FPN", "resnet34", in_channels=3, out_classes=1)

    trainer = pl.Trainer(
        accelerator = 'gpu', 
        devices = 1,
        max_epochs = 10,
        default_root_dir = "handSegment"
    )

    trainer.fit(
        model, 
        train_dataloaders = train_dataloader,
    )    

if __name__ == "__main__":
    main()