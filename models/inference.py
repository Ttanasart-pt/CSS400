#from homebrew.CNN import Model
from mobilenet.model import Model

import torch
from util.tester import testModel
from util.checkpoint import load_checkpoint

import yaml
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config', help='Path to config file.')

args = parser.parse_args()
with open(args.config, 'r') as f:
    CONFIG = yaml.safe_load(f.read())
    
def inference():
    checkpoint_path = CONFIG['checkpoint']
    imageSize = 128
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = Model().to(device)

    load_checkpoint(checkpoint_path, model)
    testModel(model, "E:\\HandPose\\CMU\\hand_syn\\synth1\\0001.jpg", imageSize, device, "mobilenet/sample/testImg")
    
if __name__ == "__main__":
    inference()