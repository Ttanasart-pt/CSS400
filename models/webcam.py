import torch
import numpy as np
import cv2
import torch
from util.tester import runModel, runModelKeypoint
from util.checkpoint import load_checkpoint
from util.segment import calc_center_bb
from modelLoader import loadFromConfig
from util.transforms import preprocess
from torch.utils.data import DataLoader
from PIL import Image

from dataloader.RHD import RHDSegment
train_dataset = RHDSegment("E:/Dataset/RHD_published_v2/training/", 256)
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
batch = next(iter(train_dataloader))

import yaml
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--segment', help='Path to segment config.')
parser.add_argument('--keypoints', help='Path to keypoint config.')

args = parser.parse_args()
with open(args.segment, 'r') as f:
    SEGMENT = yaml.safe_load(f.read())
with open(args.keypoints, 'r') as f:
    KEYPOINTS = yaml.safe_load(f.read())
    
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    segment_model = loadFromConfig(SEGMENT).to(device)
    keypoint_model = loadFromConfig(KEYPOINTS).to(device)
    
    if SEGMENT['checkpoint']:
        load_checkpoint(SEGMENT['checkpoint'], segment_model)
    segment_model.eval()
    
    if KEYPOINTS['checkpoint']:
        load_checkpoint(KEYPOINTS['checkpoint'], keypoint_model)
    keypoint_model.eval()
    
    imageSize = SEGMENT['input_size']
    detectSize = KEYPOINTS['input_size']
    preprocessor = preprocess(imageSize)
    cap = cv2.VideoCapture(0)
    padding = 64
    
    erode = np.ones((7, 7), np.uint8)
    blur = np.ones((32, 32), np.uint8)
    lastSeg = None
    segFade = 0.9

    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    while True:
        _, frame = cap.read()
        size = frame.shape[:2]
        
        frame = cv2.flip(frame, 1)
        segment = np.zeros(frame.shape).astype(np.uint8)
        handBBox = []
        
        topLeft = frame
        topRight = np.zeros((size[0], size[1], 3)).astype(np.uint8)
        bottomLeft = np.zeros((size[0], size[1], 3)).astype(np.uint8)
        bottomRight = np.zeros((size[0], size[1], 3)).astype(np.uint8)
        
        if SEGMENT['output_type'] == "light_segment" : 
            img = Image.fromarray(frame)
            img = preprocessor(img) * 255
            img = torch.unsqueeze(img.to(device), 0)
            
            with torch.no_grad():
                logits = segment_model(img)
            segment = logits.sigmoid().detach().cpu().numpy().squeeze()
            segment = cv2.resize(segment, (size[1], size[0]))
            segment = cv2.blur(segment, (5, 5))
            
            if lastSeg is not None:
                segment = np.clip(segment + lastSeg * segFade, 0, 1)
            lastSeg = segment
            
            segment = (segment * 255).astype(np.uint8)
            
            _, segment = cv2.threshold(segment, 200, 255, cv2.THRESH_BINARY)
            segment = cv2.dilate(segment, erode)
            cnt, _ = cv2.findContours(segment, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            segment = cv2.cvtColor(segment, cv2.COLOR_GRAY2BGR)
            
            for c in cnt:
                x, y, w, h = cv2.boundingRect(c)
                if w * h > 2500:
                    handBBox.append([max(0, x - padding), max(0, y - padding), min(size[1], x + w + padding), min(size[0], y + h + padding)])
        
        elif SEGMENT['output_type'] == "bbox" :  
            bbox = runModel(segment_model, frame, imageSize, device)
            p0 = (round(bbox[0][0] * size[1] / imageSize), round(bbox[0][1] * size[0] / imageSize))
            p1 = (round(p0[0] + bbox[1][0] * size[1] / imageSize), round(p0[1] + bbox[1][1] * size[0] / imageSize))
            handBBox = [p0[0], p0[1], p1[0], p1[1]]
            
        else : 
            segment = runModel(segment_model, frame, imageSize, device)
            segment = segment.argmax(0).astype(np.float32) * 255
            segment = cv2.resize(segment, (size[1], size[0]))
            
            non_zero = segment.nonzero()
            try:
                y_min = non_zero[0].min() - padding
                x_min = non_zero[1].min() - padding
                y_max = non_zero[0].max() + padding
                x_max = non_zero[1].max() + padding
                handBBox = [x_min, y_min, x_max, y_max]
            except Exception:
                pass
            
            segment = cv2.cvtColor(segment, cv2.COLOR_GRAY2BGR)
            segment = (segment * 255).astype(np.uint8)
        topRight = segment
        
        handIso = np.zeros((size[0], size[1], 3)).astype(np.uint8)
        resSurface = np.zeros(size).astype(np.float32)
        
        for hand in handBBox:
            x_min, y_min, x_max, y_max = hand
            handImg = frame[y_min : y_max, x_min : x_max, :]
            frame = cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
            
            width = handImg.shape[1]
            height = handImg.shape[0]
            
            handIso[y_min : y_max, x_min : x_max, :] = handImg
            
            if width > 0 and height > 0:
                if KEYPOINTS['output_type'] == "heatmap" :
                    hms = runModel(segment_model, frame, detectSize, device)[0]
                    for hm in hms:
                        _hm = cv2.resize(hm, (width, height), cv2.INTER_LINEAR)
                        am = np.argmax(_hm)
                        x = am % width + x_min
                        y = am // width + y_min
                        
                        frame = cv2.circle(frame, (x, y), 4, (255, 0, 0))
                        resSurface[y_min : y_max, x_min : x_max] += _hm
                    
                elif KEYPOINTS['output_type'] == "keypoints" :
                    lms = runModelKeypoint(segment_model, frame, detectSize, device)
                    for lm in lms:
                        c = (round(lm[0] * width / detectSize), round(lm[1] * height / detectSize))
                        frame = cv2.circle(frame, c, 2, (255, 0, 0), -1)    
                    frame = np.concatenate((frame, segment), 1)
        
        resSurface = (resSurface - resSurface.min()) / (resSurface.max() - resSurface.min())
        resSurface = cv2.cvtColor(resSurface, cv2.COLOR_GRAY2BGR)
        resSurface = (resSurface * 255).astype(np.uint8)
        
        bottomLeft = handIso
        bottomRight = resSurface
        
        top = np.concatenate((topLeft, topRight), 1)
        bot = np.concatenate((bottomLeft, bottomRight), 1)
        
        con = np.concatenate((top, bot), 0)
        cv2.imshow('Input', con)
        
        c = cv2.waitKey(1)
        if c == 27:
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

# example uses: python .\webcam.py --segment .\handSegment\segment_cf01_RHD.yaml --keypoints .\hourglass\train.yaml