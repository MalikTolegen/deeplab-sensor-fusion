from torch import nn

from config import SENSOR_RATIO

from ..sensor import BaseSensor


class SensorVisionFusion(nn.Module):
    def __init__(self, channels=2048):
        super(SensorVisionFusion, self).__init__()
        self.s_model = BaseSensor(channels)

        pass

    def forward(self, v_features, sensors):
        s_features = self.s_model(sensors)

        fused_feature = v_features + (s_features * SENSOR_RATIO)

        return fused_feature
