from PIL import Image
import numpy as np
import os
import json
import torch
from torch.utils.data import Dataset
from util.transforms import preprocess
from util.math import clamp

import albumentations as A
from albumentations.pytorch import ToTensorV2

class CMUDataset(Dataset):
    def __init__(self, root, size) -> None:
        super().__init__()
        self.root = root
        self.size = size
        self.preprocessor = preprocess(size)
        
        self.list_files = [f[:-4] for f in os.listdir(self.root) if f.endswith(".jpg")]
    
    def __len__(self):
        return len(self.list_files)
    
    def __getitem__(self, index):
        img_name = self.list_files[index]
        img_path = os.path.join(self.root, img_name + ".jpg")
        image = Image.open(img_path)
        
        w, h = image.size
        image = self.preprocessor(image)
        
        label_path = os.path.join(self.root, img_name + ".json")
        with open(label_path, 'r') as f:
            label_json = json.load(f)
        
        hand_label = [lm[:2] for lm in label_json['hand_pts']]
        labels = []
        sw, sh = self.size / w, self.size / h
        for hand in hand_label:
            x = hand[0] * sw
            y = hand[1] * sh
            
            labels.append(clamp(x, 0, self.size))
            labels.append(clamp(y, 0, self.size))
        
        hand_label = torch.tensor(labels)
        
        return image, hand_label

class CMUHeatmapDataset(CMUDataset):
    def gaussian(self, pos, sigma=5):
        x, y = pos
        hm = [ np.exp(-((c - x) ** 2 + (r - y) ** 2) / (2 * sigma ** 2)) for r in range(self.size) for c in range(self.size) ]
        hm = np.array(hm, dtype = np.float32)
        hm = np.reshape(hm, newshape = (self.size, self.size))
        return hm

    def __getitem__(self, index):
        image, hand_label = super().__getitem__(index)
        hand_xy = []
        
        for i in range(round(len(hand_label) / 2)):
            hand_xy.append((hand_label[i * 2], hand_label[i * 2 + 1]))
        
        hand_xy = np.array(hand_xy)
        hms = torch.tensor([self.gaussian(lm) for lm in hand_xy])
        return image, hms

class CMUBBoxDataset(CMUDataset):
    def __getitem__(self, index):
        image, hand_label = super().__getitem__(index)
        #dim = image.size()
        hand_min = [ self.size, self.size ]
        hand_max = [ 0, 0 ]
        
        for i in range(round(len(hand_label) / 2)):
            if(hand_label[i * 2] == 0 or hand_label[i * 2 + 1] == 0):
                continue
            
            hand_min[0] = min(hand_min[0], hand_label[i * 2])
            hand_min[1] = min(hand_min[1], hand_label[i * 2 + 1])
            hand_max[0] = max(hand_max[0], hand_label[i * 2])
            hand_max[1] = max(hand_max[1], hand_label[i * 2 + 1])
            
        bbox = [ hand_min[0], hand_min[1], hand_max[0], hand_max[1] ]
        bbox = torch.tensor(bbox, dtype=torch.float32)
        return image, bbox

class CMUHandSegment(CMUBBoxDataset):
    def __init__(self, root, size) -> None:
        super().__init__(root, size)
        
        self.transform = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.RandomSizedBBoxSafeCrop(size, size),
                ToTensorV2()
            ],
            additional_targets = {
                'segment': 'image',
            },
            bbox_params = A.BboxParams(format = "pascal_voc")
        )
        
    def __getitem__(self, index):
        img_name = self.list_files[index]
        img_path = os.path.join(self.root, img_name + ".jpg")
        image = np.array(Image.open(img_path))
        
        label_path = os.path.join(self.root, img_name + ".json")
        with open(label_path, 'r') as f:
            label_json = json.load(f)
        
        hand_label = [lm[:2] for lm in label_json['hand_pts']]
        hand_min = [ image.shape[1], image.shape[0] ]
        hand_max = [ 0, 0 ]
        
        for hand in hand_label:
            x = hand[0]
            y = hand[1]
            
            hand_min[0] = min(hand_min[0], x)
            hand_min[1] = min(hand_min[1], y)
            hand_max[0] = max(hand_max[0], x)
            hand_max[1] = max(hand_max[1], y)
        
        padding = 8
        #dim = image.size()
        handSeg = np.zeros((image.shape[0], image.shape[1]))
        x0 = round(max(0, hand_min[0] - padding))
        y0 = round(max(0, hand_min[1] - padding))
        x1 = round(min(image.shape[1], hand_max[0] + padding))
        y1 = round(min(image.shape[0], hand_max[1] + padding))
        handSeg[y0:y1, x0:x1] = 1
        bbox = [[x0, y0, x1, y1, "hand"]]
        
        transformed = self.transform(image = image, segment = handSeg, bboxes = bbox)
        
        return transformed['image'], transformed['segment'].squeeze()