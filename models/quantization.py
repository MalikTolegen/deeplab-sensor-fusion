import torch
import torch.nn as nn
from torch.quantization import QuantStub, DeQuantStub
from typing import Optional, List, Union, Tuple

class QuantizableMixin:
    """Mixin class to add quantization support to any module."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # type: ignore
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        
    def fuse_model(self):
        """Fuse Conv/BN/ReLU layers for quantization.
        Should be implemented by subclasses if they contain fusable layers.
        """
        pass

def fuse_conv_bn_relu(module, conv_name='conv', bn_name='bn', relu_name='relu'):
    """Helper function to fuse Conv + BN + ReLU layers."""
    if hasattr(module, conv_name) and hasattr(module, bn_name):
        if hasattr(module, relu_name):
            # Conv + BN + ReLU
            torch.quantization.fuse_modules(
                module,
                [conv_name, bn_name, relu_name],
                inplace=True
            )
        else:
            # Conv + BN
            torch.quantization.fuse_modules(
                module,
                [conv_name, bn_name],
                inplace=True
            )
    elif hasattr(module, conv_name) and hasattr(module, relu_name):
        # Conv + ReLU
        torch.quantization.fuse_modules(
            module,
            [conv_name, relu_name],
            inplace=True
        )

class QuantizableConv2d(nn.Conv2d):
    """Quantizable Conv2d layer with optional batch norm and ReLU."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, dilation: int = 1, 
                 groups: int = 1, bias: bool = True, 
                 batch_norm: bool = False, activation: bool = False):
        super().__init__(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias
        )
        self.batch_norm = nn.BatchNorm2d(out_channels) if batch_norm else None
        self.activation = nn.ReLU(inplace=True) if activation else None
    
    def forward(self, x):
        x = self.quant(x)
        x = super().forward(x)
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return self.dequant(x)
    
    def fuse_model(self):
        if self.batch_norm is not None:
            if self.activation is not None:
                # Conv + BN + ReLU
                torch.quantization.fuse_modules(
                    self,
                    ['conv', 'batch_norm', 'activation'],
                    inplace=True
                )
            else:
                # Conv + BN
                torch.quantization.fuse_modules(
                    self,
                    ['conv', 'batch_norm'],
                    inplace=True
                )
        elif self.activation is not None:
            # Conv + ReLU
            torch.quantization.fuse_modules(
                self,
                ['conv', 'activation'],
                inplace=True
            )
