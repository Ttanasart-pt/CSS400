#from homebrew.CNN import Model
#from mobilenet.model import Model
#from hourglass.model import Model
from Zimmerman.handSegNet import Model

import torch
import numpy as np
import cv2
import torch
from util.tester import runModel, runModelKeypoint
from util.checkpoint import load_checkpoint

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
            hms = runModelKeypoint(model, frame, imageSize, device)
            srf = np.zeros_like(hms[0])
            for hm in hms:
                srf += hm
            cv2.addWeighted(frame, 1, srf, 1, 1)
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
            frame = segment * 255
            
        cv2.imshow('Input', frame)
        
        c = cv2.waitKey(1)
        if c == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    
# example uses: python .\webcam.py --config "homebrew/train.yaml"