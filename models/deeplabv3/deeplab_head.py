from typing import Sequence, Optional

import torch
from torch import nn
from torch.nn import functional as F
from ..cbam import Cbam, conv_block
from ..quantization import QuantizableMixin, fuse_conv_bn_relu


class DeepLabHeadWithCbam(nn.Module, QuantizableMixin):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        atrous_rates: Sequence[int] = (12, 24, 36),
        quantize: bool = False
    ) -> None:
        super().__init__()
        self.quantize = quantize
        self.in_channels = in_channels
        
        # Initialize quantization stubs if needed
        if self.quantize:
            QuantizableMixin.__init__(self)
        
        # ASPP expects 2048 input channels by default
        self.aspp_in_channels = 2048
        
        # Add a channel adjustment layer to handle different input dimensions
        self.channel_adjust = None
        if in_channels != self.aspp_in_channels:
            self.channel_adjust = nn.Sequential(
                nn.Conv2d(in_channels, self.aspp_in_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(self.aspp_in_channels),
                nn.ReLU(inplace=True)
            )
            
        self.aspp = ASPPwithCbam(self.aspp_in_channels, atrous_rates, quantize=quantize)
        
        # Additional layers after ASPP
        self.conv1 = nn.Conv2d(256, 256, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(256, num_classes, 1)
        
    def forward(self, x):
        if self.quantize:
            x = self.quant(x)
        
        # Adjust input channels if needed
        if self.channel_adjust is not None:
            x = self.channel_adjust(x)
            
        x = self.aspp(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        
        if self.quantize:
            x = self.dequant(x)
            
        return x
    
    def fuse_model(self):
        # Fuse Conv + BN + ReLU
        torch.quantization.fuse_modules(
            self,
            ['conv1', 'bn1', 'relu'],
            inplace=True
        )
        # Fuse ASPP if it has fuse_model
        if hasattr(self.aspp, 'fuse_model'):
            self.aspp.fuse_model()


class ASPPConvWithCbam(nn.Module, QuantizableMixin):
    def __init__(self, in_channels: int, out_channels: int, dilation: int, quantize: bool = False) -> None:
        super().__init__()
        self.quantize = quantize
        
        # Initialize quantization stubs if needed
        if self.quantize:
            QuantizableMixin.__init__(self)
        
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.cbam = Cbam(out_channels, r=2, quantize=quantize)
        
    def forward(self, x):
        if self.quantize:
            x = self.quant(x)
            
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.cbam(x)
        
        if self.quantize:
            x = self.dequant(x)
            
        return x
    
    def fuse_model(self):
        # Fuse Conv + BN + ReLU
        torch.quantization.fuse_modules(
            self,
            ['conv', 'bn', 'relu'],
            inplace=True
        )
        # Fuse CBAM if it has fuse_model
        if hasattr(self.cbam, 'fuse_model'):
            self.cbam.fuse_model()


class ASPPPooling(nn.Module, QuantizableMixin):
    def __init__(self, in_channels: int, out_channels: int, quantize: bool = False) -> None:
        super().__init__()
        self.quantize = quantize
        
        # Initialize quantization stubs if needed
        if self.quantize:
            QuantizableMixin.__init__(self)
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        if self.quantize:
            x = self.quant(x)
            
        size = x.shape[-2:]
        x = self.pool(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        
        if self.quantize:
            x = self.dequant(x)
            
        return x
    
    def fuse_model(self):
        # Fuse Conv + BN + ReLU
        torch.quantization.fuse_modules(
            self,
            ['conv', 'bn', 'relu'],
            inplace=True
        )
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPPwithCbam(nn.Module, QuantizableMixin):
    def __init__(
        self,
        in_channels: int,
        atrous_rates: Sequence[int],
        out_channels: int = 256,
        quantize: bool = False
    ) -> None:
        super().__init__()
        self.quantize = quantize
        self.out_channels = out_channels
        
        # Initialize quantization stubs if needed
        if self.quantize:
            QuantizableMixin.__init__(self)
        
        # 1x1 convolution branch
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Atrous convolution branches
        rates = tuple(atrous_rates)
        self.aspp_conv1 = ASPPConvWithCbam(in_channels, out_channels, rates[0], quantize=quantize) if len(rates) > 0 else None
        self.aspp_conv2 = ASPPConvWithCbam(in_channels, out_channels, rates[1], quantize=quantize) if len(rates) > 1 else None
        self.aspp_conv3 = ASPPConvWithCbam(in_channels, out_channels, rates[2], quantize=quantize) if len(rates) > 2 else None
        
        # Image pooling branch
        self.image_pool = ASPPPooling(in_channels, out_channels, quantize=quantize)
        
        # Calculate the number of branches (1x1 + atrous convs + image pool)
        num_branches = 1 + len(rates) + 1  # 1x1 + atrous convs + image pool
        
        # Projection
        self.project = nn.Sequential(
            nn.Conv2d(
                num_branches * out_channels,
                out_channels,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.quantize:
            x = self.quant(x)
        
        # Forward through all branches
        x1 = self.conv1x1(x)
        
        # Forward through atrous convolutions if they exist
        x2 = self.aspp_conv1(x) if self.aspp_conv1 is not None else None
        x3 = self.aspp_conv2(x) if self.aspp_conv2 is not None else None
        x4 = self.aspp_conv3(x) if self.aspp_conv3 is not None else None
        
        # Image pooling branch
        x5 = self.image_pool(x)
        
        # Shapes are now aligned, no debug prints needed
        
        # Concatenate all branches
        branches = [x1]
        if x2 is not None:
            branches.append(x2)
        if x3 is not None:
            branches.append(x3)
        if x4 is not None:
            branches.append(x4)
        branches.append(x5)
        
        # Ensure all branches have the same spatial dimensions
        h, w = x1.size(2), x1.size(3)
        for i in range(len(branches)):
            if branches[i].size(2) != h or branches[i].size(3) != w:
                branches[i] = F.interpolate(
                    branches[i], 
                    size=(h, w), 
                    mode='bilinear', 
                    align_corners=False
                )
        
        # Project and return
        out = torch.cat(branches, dim=1)
        out = self.project(out)
        
        if self.quantize:
            out = self.dequant(out)
            
        return out
