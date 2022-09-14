from PIL import Image
import numpy as np
import os
import json
import torch
from torch.utils.data import Dataset
from util.transforms import preprocess
from util.math import clamp

class CMUDataset(Dataset):
    def __init__(self, root, size) -> None:
        super().__init__()
        self.root = root
        self.size = size
        self.proprocessor = preprocess(size)
        
        self.list_files = [f[:-4] for f in os.listdir(self.root) if f.endswith(".jpg")]
    
    def __len__(self):
        return len(self.list_files)
    
    def __getitem__(self, index):
        img_name = self.list_files[index]
        img_path = os.path.join(self.root, img_name + ".jpg")
        image = Image.open(img_path)
        
        w, h = image.size
        image = self.proprocessor(image)
        
        label_path = os.path.join(self.root, img_name + ".json")
        with open(label_path, 'r') as f:
            label_json = json.load(f)
        
        hand_label = [lm[:2] for lm in label_json['hand_pts']]
        labels = []
        sf = self.size / h
        for hand in hand_label:
            x = hand[0] * sf
            y = hand[1] * sf
            
            x -= (w * sf - self.size) / 2
            
            labels.append(clamp(x, 0, self.size))
            labels.append(clamp(y, 0, self.size))
            
        
        hand_label = torch.tensor(labels)
        
        return image, hand_label