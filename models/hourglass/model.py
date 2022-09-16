import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchsummary import summary

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Hourglass(nn.Module):
    def __init__(self):
        super(Hourglass, self).__init__()
        
        self.down = []
        self.down.append(self.downSample(3,  8).to(device))
        self.down.append(self.downSample(8, 16).to(device))
        
        self.up = []
        self.up.append(self.upSample(16, 8).to(device))
        self.up.append(self.upSample( 8, 8).to(device))
        
        self.bottleneck = nn.Conv2d(16, 16, 1)
        
        self.interSup1  = nn.Conv2d(8, 8, 1)
        self.heatmap    = nn.Conv2d(8, 3, 1)
        self.heatSup    = nn.Conv2d(3, 8, 1)
        self.interSup2  = nn.Conv2d(8, 3, 1)
        
    def downSample(self, in_channel, out_channel, stride = 1):
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            
            nn.Conv2d(out_channel, out_channel, 3, stride, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            
            nn.MaxPool2d(2),
        )

    
    def upSample(self, in_channel, out_channel):
        return nn.Sequential(
            nn.Upsample(scale_factor = 2),
            nn.Conv2d(in_channel, out_channel, 1),
            nn.ReLU(),
        )
        
    def forward(self, x):
        #print(f"{x.shape = }")
        ds0 = self.down[0](x)
        #print(f"{ds0.shape = }")
        ds1 = self.down[1](ds0)
        #print(f"{ds1.shape = }")
        
        bottle = self.bottleneck(ds1)
        #print(f"{bottle.shape = }")
        
        du1 = self.up[0](bottle + ds1)
        #print(f"{du1.shape = }")
        
        du0 = self.up[1](du1 + ds0)
        #print(f"{du0.shape = }")
        
        sup  = self.interSup1(du0)
        heat = self.heatmap(sup)
        heas = self.heatSup(heat)
        out  = self.interSup2(sup + heas)
        
        return out
    
if __name__ == "__main__":
    model = Hourglass().to(device)
    
    summary(model, (3, 128, 128))