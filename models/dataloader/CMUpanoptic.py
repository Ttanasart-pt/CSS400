import random
from PIL import Image
import numpy as np
import os
import json
import torch
import cv2
from torch.utils.data import Dataset
from util.transforms import preprocess

import albumentations as A
from albumentations.pytorch import ToTensorV2
import warnings
warnings.filterwarnings("ignore", category = UserWarning) 

class CMUDataset(Dataset):
    def __init__(self, root, size) -> None:
        super().__init__()
        self.root = root
        self.size = size
        self.preprocessor = preprocess(size)
        
        self.list_files = [f[:-4] for f in os.listdir(self.root) if f.endswith(".jpg")]
        
        self.transform = A.Compose(
            [
                A.Resize(size, size),
                ToTensorV2()
            ],
            keypoint_params = A.KeypointParams("xy")
        )
        
        self.padding = 32
    
    def __len__(self):
        return len(self.list_files)
    
    def __getitem__(self, index):
        img_name = self.list_files[index]
        img_path = os.path.join(self.root, img_name + ".jpg")
        img = np.array(Image.open(img_path))
        
        label_path = os.path.join(self.root, img_name + ".json")
        with open(label_path, 'r') as f:
            label_json = json.load(f)
        
        hand_label = [lm[:2] for lm in label_json['hand_pts']]
        kp = []
        for h in hand_label:
            x = h[0]
            y = h[1]
            if x > 0 and x < img.shape[1] and y > 0 and y < img.shape[0]:
                kp.append((x, y))
            else:
                kp.append((0, 0))
        
        x_min = img.shape[1]
        y_min = img.shape[0]
        x_max = 0
        y_max = 0
        
        px0 = random.randrange(0, self.padding)
        py0 = random.randrange(0, self.padding)
        px1 = random.randrange(0, self.padding)
        py1 = random.randrange(0, self.padding)
        
        for k in kp:
            x_min = min(x_min, k[0])
            y_min = min(y_min, k[1])
            x_max = max(x_max, k[0])
            y_max = max(y_max, k[1])
        x_min = round(max(0, x_min - px0))
        y_min = round(max(0, y_min - py0))
        x_max = round(min(img.shape[1], x_max + px1))
        y_max = round(min(img.shape[0], y_max + py1))
        x_min, x_max = min(x_min, x_max), max(x_min, x_max)
        y_min, y_max = min(y_min, y_max), max(y_min, y_max)
        if x_max > x_min and y_max > y_min:
            img = A.crop(img, x_min, y_min, x_max, y_max)
        _kp = []
        for k in kp:
            x = k[0] - x_min
            y = k[1] - y_min
            if x > 0 and x < img.shape[1] and y > 0 and y < img.shape[0]:
                _kp.append((x, y))
            else:
                _kp.append((0, 0))
        
        transformed = self.transform(image = img, keypoints = _kp)
        kp = torch.tensor(transformed['keypoints']).reshape((42))
        return transformed['image'], kp

class CMUHeatmapDataset(CMUDataset):
    def __init__(self, root, size) -> None:
        super().__init__(root, size)
        self.padding = 16
    
    def gaussian(self, pos, sigma = 6, scale = 1):
        scale = 1 / 4
        x, y = pos
        
        x *= scale
        y *= scale
        hmSize = round(self.size * scale)
        
        if(pos == (0, 0)):
            hm = np.zeros((hmSize, hmSize))
        else:
            hm = [ np.exp(-((c - x) ** 2 + (r - y) ** 2) / (2 * sigma ** 2)) for r in range(hmSize) for c in range(hmSize) ]
        hm = np.array(hm, dtype = np.float32)
        hm = np.reshape(hm, newshape = (hmSize, hmSize))
        return hm

    def __getitem__(self, index):
        image, hand_label = super().__getitem__(index)
        hl = []
        for i in range(0, len(hand_label), 2):
            hl.append((hand_label[i], hand_label[i + 1]))
        hms = np.array([self.gaussian(lm) for lm in hl])
        hms = torch.tensor(hms)
        return image, hms

class CMUBBoxDataset(CMUDataset):
    def __init__(self, root, size) -> None:
        super().__init__(root, size)
        self.padding = 96
        
    def __getitem__(self, index):
        image, kp = super().__getitem__(index)
        #dim = image.size()
        
        x_min = image.shape[2]
        y_min = image.shape[1]
        x_max = 0
        y_max = 0
        pad = 16
        
        for k in kp:
            x_min = min(x_min, k[0])
            y_min = min(y_min, k[1])
            x_max = max(x_max, k[0])
            y_max = max(y_max, k[1])
        x_min = round(max(0, x_min - pad))
        y_min = round(max(0, y_min - pad))
        x_max = round(min(image.shape[2], x_max + pad))
        y_max = round(min(image.shape[1], y_max + pad))
        
        bbox = [ x_min, y_min, x_max, y_max ]
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
        
        #dim = image.size()
        handSeg = np.zeros((image.shape[0], image.shape[1]))
        x0 = round(max(0, hand_min[0] - self.padding))
        y0 = round(max(0, hand_min[1] - self.padding))
        x1 = round(min(image.shape[1], hand_max[0] + self.padding))
        y1 = round(min(image.shape[0], hand_max[1] + self.padding))
        handSeg[y0:y1, x0:x1] = 1
        bbox = [[x0, y0, x1, y1, "hand"]]
        
        transformed = self.transform(image = image, segment = handSeg, bboxes = bbox)
        
        return transformed['image'], transformed['segment'].squeeze()