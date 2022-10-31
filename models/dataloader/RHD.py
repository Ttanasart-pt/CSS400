from PIL import Image
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from util.transforms import preprocess

import albumentations as A
from albumentations.pytorch import ToTensorV2
import warnings
warnings.filterwarnings("ignore", category = UserWarning) 

class RHDSegment(Dataset):
    def __init__(self, root, size = 256):
        super(RHDSegment, self).__init__()
        
        self.root = root
        self.dir_clr = self.root + 'color'
        self.dir_msk = self.root + 'mask'
        
        self.files = [f[:-4] for f in os.listdir(self.dir_clr)]
        self.size = size
        
        self.transform = A.Compose(
            [
                A.HorizontalFlip(p = 0.5),
                A.ShiftScaleRotate(p = 0.5),
                A.Resize(size, size),
                ToTensorV2()
            ],
            additional_targets = { 'mask': 'mask' }
        )

    def __getitem__(self, index):
        name = self.files[index]
        pth_img = self.dir_clr + name + '.png'
        pth_msk = self.dir_msk + name + '.png'
        
        img = np.array(Image.open(pth_img))
        msk = np.array(Image.open(pth_msk))
        
        transformed = self.transform(image = img, mask = msk)
        imgT = transformed['image']
        mskT = transformed['mask']
        
        return imgT, mskT

    def __len__(self):
        return len(self.files)