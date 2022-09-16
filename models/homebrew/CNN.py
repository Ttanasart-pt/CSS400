import torchvision

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        features = [8, 16, 32]
        num_channel = 3
        
        layers = []
        r = num_channel
        for f in features:
            layers.append(self.downSampleBlock( r,  f))
            r = f
        features.reverse()
        for f in features[1:]:
            layers.append(self.downSampleBlock( r,  f))
            r = f

        layers.append(nn.Flatten())
        layers.append(nn.Linear(72, 42))
        
        self.layers = nn.Sequential(*layers)
        
    def downSampleBlock(self, in_channel, out_channel):
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channels = out_channel, kernel_size = 3, stride = 2),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self, x):
        return self.layers(x)