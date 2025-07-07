from torch import nn
import torch.nn.functional as F

from config import SENSOR_RATIO
from ..sensor import BaseSensor


class SensorVisionFusion(nn.Module):
    def __init__(self, channels=2048):
        super(SensorVisionFusion, self).__init__()
        self.s_model = BaseSensor(channels)
        self.channels = channels
        
        # Channel adjustment for sensor features
        self.sensor_channel_adjust = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        # Create visual channel adjustment dynamically in forward pass
        self.visual_channel_adjust = None

    def forward(self, v_features, sensors):
        # Get sensor features
        s_features = self.s_model(sensors)
        
        # Create visual channel adjustment if needed
        if v_features.size(1) != self.channels:
            if self.visual_channel_adjust is None or self.visual_channel_adjust[0].in_channels != v_features.size(1):
                self.visual_channel_adjust = nn.Sequential(
                    nn.Conv2d(v_features.size(1), self.channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(self.channels),
                    nn.ReLU(inplace=True)
                ).to(v_features.device)
            v_features = self.visual_channel_adjust(v_features)
        
        # Ensure sensor features have the same spatial dimensions as visual features
        if s_features.size()[-2:] != v_features.size()[-2:]:
            s_features = F.interpolate(
                s_features,
                size=v_features.size()[-2:],
                mode='bilinear',
                align_corners=False
            )
        
        # Adjust sensor channels to match visual features
        s_features = self.sensor_channel_adjust(s_features)
        
        # Fuse features
        fused_feature = v_features + (s_features * SENSOR_RATIO)

        return fused_feature
