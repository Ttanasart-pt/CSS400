from homebrew.CNN import Model

import torch
from util.tester import testModel
from util.checkpoint import load_checkpoint

def inference():
    checkpoint_path = "homebrew/checkpoints/test.chk"
    imageSize = 128
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = Model().to(device)

    load_checkpoint(checkpoint_path, model)
    testModel(model, "E:\\HandPose\\CMU\\hand_syn\\synth1\\0001.jpg", imageSize, device, "homebrew/sample/testImg")
    
if __name__ == "__main__":
    inference()