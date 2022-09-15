import cv2
from homebrew.CNN import Model
import torch
from util.tester import runModel
from util.checkpoint import load_checkpoint

def main():
    checkpoint_path = "homebrew/checkpoints/test.chk"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    imageSize = 128
    
    model = Model().to(device)
    load_checkpoint(checkpoint_path, model)
    
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