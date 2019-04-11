import torch
from torch import nn
import torch.nn.functional as F

"""
Questions?

Should the convolutional layers have biases?
Paper reads: "biases are omitted for simplifying notation" (p.3) which is ambiguous.

Clarification from Kaiming He here:
>"Biases are in the BN layers that follow." (https://github.com/KaimingHe/deep-residual-networks/issues/10)

This can be accomplished in PyTorch BatchNorm2d by setting affine=True (default). 
"""

class block(nn.Module):
    def __init__(self, filters, subsample=False):
        super().__init__()
        """
        A 2-layer residual learning building block as illustrated by Fig.2
        in "Deep Residual Learning for Image Recognition"
        
        Parameters:
        
        - filters:   int
                     the number of filters for all layers in this block
                   
        - subsample: boolean
                     whether to subsample the input feature maps with stride 2
                     and doubling in number of filters
                     
        Attributes:
        
        - shortcuts: boolean
                     When false the residual shortcut is removed
                     resulting in a 'plain' convolutional block.
        """
        # Determine subsampling
        s = 0.5 if subsample else 1.0
        
        # Setup layers
        self.conv1 = nn.Conv2d(int(filters*s), filters, kernel_size=3, 
                               stride=int(1/s), padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(filters, track_running_stats=True)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(filters, track_running_stats=True)
        self.relu2 = nn.ReLU()

        # Shortcut downsampling
        self.downsample = nn.AvgPool2d(kernel_size=1, stride=2)

        # Initialise weights according to the method described in 
        # “Delving deep into rectifiers: Surpassing human-level performance on ImageNet 
        # classification” - He, K. et al. (2015)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)   
        
    def shortcut(self, z, x):
        """ 
        Implements parameter free shortcut connection by identity mapping.
        If dimensions of input x are greater than activations then this
        is rectified by downsampling and then zero padding dimension 1
        as described by option A in paper.
        
        Parameters:
        - x: tensor
             the input to the block
        - z: tensor
             activations of block prior to final non-linearity
        """
        if x.shape != z.shape:
            d = self.downsample(x)
            p = torch.mul(d, 0)
            return z + torch.cat((d, p), dim=1)
        else:
            return z + x        
    
    def forward(self, x, shortcuts=False):
        z = self.conv1(x)
        z = self.bn1(z)
        z = self.relu1(z)
        
        z = self.conv2(z)
        z = self.bn2(z)
        
        # Shortcut connection
        # This if statement is the only difference between
        # a convolutional net and a resnet!
        if shortcuts:
            z = self.shortcut(z, x)

        z = self.relu2(z)
        
        return z
    


class ResNet(nn.Module):
    def __init__(self, n, shortcuts=True):
        super().__init__()
        self.shortcuts = shortcuts
        
        # Input
        self.convIn = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bnIn   = nn.BatchNorm2d(16, track_running_stats=True)
        self.relu   = nn.ReLU()
        
        # Stack1
        self.stack1 = nn.ModuleList([block(16, subsample=False) for _ in range(n)])

        # Stack2
        self.stack2a = block(32, subsample=True)
        self.stack2b = nn.ModuleList([block(32, subsample=False) for _ in range(n-1)])

        # Stack3
        self.stack3a = block(64, subsample=True)
        self.stack3b = nn.ModuleList([block(64, subsample=False) for _ in range(n-1)])
        
        # Output
        # The parameters of this average pool are not specified in paper.
        # Initially I tried kernel_size=2 stride=2 resulting in 
        # 64*4*4= 1024 inputs to the fully connected layer. More aggresive
        # pooling as implemented below results in better results and also
        # better matches the total model parameter count cited by authors.
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fcOut   = nn.Linear(64, 10, bias=True)
        self.softmax = nn.LogSoftmax(dim=-1)
        
        # Initilise weights in fully connected layer 
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal(m.weight)
                m.bias.data.zero_()      
        
        
    def forward(self, x):     
        z = self.convIn(x)
        z = self.bnIn(z)
        z = self.relu(z)
        
        for l in self.stack1: z = l(z, shortcuts=self.shortcuts)
        
        z = self.stack2a(z, shortcuts=self.shortcuts)
        for l in self.stack2b: 
            z = l(z, shortcuts=self.shortcuts)
        
        z = self.stack3a(z, shortcuts=self.shortcuts)
        for l in self.stack3b: 
            z = l(z, shortcuts=self.shortcuts)

        z = self.avgpool(z)
        z = z.view(z.size(0), -1)
        z = self.fcOut(z)
        return self.softmax(z)
