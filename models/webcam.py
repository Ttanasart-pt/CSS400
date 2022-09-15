#from homebrew.CNN import Model
from mobilenet.model import Model

import cv2
import torch
from util.tester import runModel
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
        
        lms = runModel(model, frame, imageSize, device)
        size = frame.shape[:2]
        sw = size[0] / imageSize
        sh = size[1] / imageSize
        
        for lm in lms:
            c = (round(lm[0] * sw), round(lm[1] * sh))
            frame = cv2.circle(frame, c, 2, (255, 0, 0), -1)
        
        cv2.imshow('Input', frame)
        
        c = cv2.waitKey(1)
        if c == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()