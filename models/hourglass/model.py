from email import header
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchsummary import summary

class Hourglass(nn.Module):
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
        self.heatmap    = nn.Conv2d(8, 1, 1)
        self.heatSup    = nn.Conv2d(1, 8, 1)
        self.interSup2  = nn.Conv2d(8, 3, 1)
        
    def forward(self, x):
        ds0 = self.down[0](x)
        ds1 = self.down[1](ds0)
        
        bottle = self.bottleneck(ds1)
        
        du1 = self.up[0](bottle + ds1)
        du0 = self.up[1](du1 + ds0)
        
        sup  = self.interSup1(du0)
        heat = self.heatmap(sup)
        
        heas = self.heatSup(heat)
        out  = self.interSup2(sup + heas)
        
        return out, heat
    
class Model(nn.Module):
    def __init__(self, num_class = 21, device = "cuda"):
        super(Model, self).__init__()
        
        self.num_class = num_class
        self.num_layers = 2
        
        self.hourglasses = nn.ModuleList()
        for _ in range(num_class):
            self.hourglasses.append(Hourglass().to(device))

    def forward(self, x):
        pHeat = torch.zeros((x.shape[0], self.num_class, x.shape[2], x.shape[3]))
        
        curr = x
        for _ in range(self.num_layers):
            for i, hr in enumerate(self.hourglasses):
                out, heat = hr(curr)
                pHeat[:, i, :, :] = heat.squeeze()
            curr = curr + out
            
        return pHeat
    
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Model().to(device)
    
    summary(model, (3, 256, 256))