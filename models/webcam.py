#from homebrew.CNN import Model
#from mobilenet.model import Model
#from hourglass.model import Model
#from colorHandPose.handSegNet import HandSegNet as Model
from colorHandPose.posenet import PoseNet as Model

import torch
import numpy as np
import cv2
import torch
from util.tester import runModel, runModelKeypoint
from util.checkpoint import load_checkpoint
from util.segment import calc_center_bb

import yaml
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config', help='Path to config file.')

args = parser.parse_args()
with open(args.config, 'r') as f:
    CONFIG = yaml.safe_load(f.read())
    
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Model().to(device)
    
    if CONFIG['checkpoint']:
        checkpoint_path = CONFIG['checkpoint']
        load_checkpoint(checkpoint_path, model)
    imageSize = CONFIG['input_size']
    
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    while True:
        _, frame = cap.read()
        
        size = frame.shape[:2]
        sw = size[0] / imageSize
        sh = size[1] / imageSize
        
        if CONFIG['output_type'] == "heatmap" :
            hms = runModel(model, frame, imageSize, device)[0]
            srf = np.zeros(size).astype(np.float32)
            for hm in hms:
                _hm = cv2.resize(hm, (size[1], size[0]), cv2.INTER_LINEAR)
                am = np.argmax(_hm)
                x = am % size[0]
                y = am // size[0]
                
                frame = cv2.circle(frame, (x, y), 4, (255, 0, 0))
                srf += _hm
            srf = (srf - srf.min()) / (srf.max() - srf.min())
            srf = cv2.cvtColor(srf, cv2.COLOR_GRAY2BGR)
            srf = (srf * 255).astype(np.uint8)
            
            frame = np.concatenate((frame, srf), 1)
        elif CONFIG['output_type'] == "keypoints" :
            lms = runModelKeypoint(model, frame, imageSize, device)
            for lm in lms:
                c = (round(lm[0] * sw), round(lm[1] * sh))
                frame = cv2.circle(frame, c, 2, (255, 0, 0), -1)
        elif CONFIG['output_type'] == "bbox" :  
            bbox = runModel(model, frame, imageSize, device)
            p0 = (round(bbox[0][0] * sw), round(bbox[0][1] * sh))
            p1 = (round(p0[0] + bbox[1][0] * sw), round(p0[1] + bbox[1][1] * sh))
            frame = cv2.rectangle(frame, p0, p1, (255, 0, 0), 2)
        elif CONFIG['output_type'] == "segment" : 
            segment = runModel(model, frame, imageSize, device)
            #print(segment.shape)
            segment = segment.argmax(0).astype(np.float32)
            segment = cv2.resize(segment, (size[1], size[0]))
            
            non_zero = segment.nonzero()
            padding = 16
            try:
                y_min = non_zero[0].min() - padding
                x_min = non_zero[1].min() - padding
                y_max = non_zero[0].max() + padding
                x_max = non_zero[1].max() + padding
                frame = cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
            except Exception:
                pass
            
            segment = cv2.cvtColor(segment, cv2.COLOR_GRAY2BGR)
            segment = (segment * 255).astype(np.uint8)
            
            frame = np.concatenate((frame, segment), 1)
            #cv2.addWeighted(frame, 1, segment, 1, 1)
            
        cv2.imshow('Input', frame)
        
        c = cv2.waitKey(1)
        if c == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    
# example uses: python .\webcam.py --config "homebrew/train.yaml"