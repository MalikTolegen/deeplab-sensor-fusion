import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from ..quantization import QuantizableMixin, fuse_conv_bn_relu


def conv_block(in_channels, out_channels, quantize: bool = False):
    if quantize:
        return QuantizableConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            batch_norm=True,
            activation=True
        )
    else:
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class SAM(nn.Module):
    def __init__(self, bias=False, quantize: bool = False):
        super(SAM, self).__init__()
        self.bias = bias
        self.quantize = quantize
        self.conv = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=7,
            stride=1,
            padding=3,
            dilation=1,
            bias=self.bias
        )
        
        if self.quantize:
            self.quant = torch.quantization.QuantStub()
            self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        if self.quantize:
            x = self.quant(x)
            
        max_pool = torch.max(x, 1, keepdim=True)[0]
        avg_pool = torch.mean(x, 1, keepdim=True)
        concat = torch.cat((max_pool, avg_pool), dim=1)
        output = self.conv(concat)
        output = output * x
        
        if self.quantize:
            output = self.dequant(output)
            
        return output
    
    def fuse_model(self):
        # No layers to fuse in SAM
        pass

class CAM(nn.Module):
    def __init__(self, channels, r, quantize: bool = False):
        super(CAM, self).__init__()
        self.channels = channels
        self.r = r
        self.quantize = quantize
        
        self.fc1 = nn.Linear(
            in_features=self.channels,
            out_features=self.channels//self.r,
            bias=True
        )
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(
            in_features=self.channels//self.r,
            out_features=self.channels,
            bias=True
        )
        
        if self.quantize:
            self.quant = torch.quantization.QuantStub()
            self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        if self.quantize:
            x = self.quant(x)
            
        b, c, h, w = x.size()
        y = F.adaptive_avg_pool2d(x, (1, 1)).view(b, c)
        
        if self.quantize:
            y = self.fc1(y)
            y = self.relu(y)
            y = self.fc2(y)
        else:
            y = self.fc1(y)
            y = self.relu(y)
            y = self.fc2(y)
            
        y = y.view(b, c, 1, 1)
        y = torch.sigmoid(y).expand_as(x)
        output = x * y
        
        if self.quantize:
            output = self.dequant(output)
            
        return output
    
    def fuse_model(self):
        # No layers to fuse in CAM
        pass

class Cbam(nn.Module, QuantizableMixin):
    def __init__(self, channels, r, quantize: bool = False):
        super(Cbam, self).__init__()
        self.channels = channels
        self.r = r
        self.quantize = quantize
        
        # Initialize quantization stubs if needed
        if self.quantize:
            QuantizableMixin.__init__(self)
        
        self.cam = CAM(channels=self.channels, r=self.r, quantize=quantize)
        self.sam = SAM(bias=False, quantize=quantize)

    def forward(self, x):
        if self.quantize:
            x = self.quant(x)
            
        x = self.cam(x)
        x = self.sam(x)
        
        if self.quantize:
            x = self.dequant(x)
            
        return x
    
    def fuse_model(self):
        if hasattr(self, 'quant'):
            # Fuse CAM and SAM if they have fuse_model
            if hasattr(self.cam, 'fuse_model'):
                self.cam.fuse_model()
            if hasattr(self.sam, 'fuse_model'):
                self.sam.fuse_model()