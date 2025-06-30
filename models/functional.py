from typing import (
    Dict,
    List,
)
from collections import OrderedDict

from torch.nn import functional as F
from torchvision.models.segmentation import (
    deeplabv3_mobilenet_v3_large,
    deeplabv3_resnet50,
    deeplabv3_resnet101,
)
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.detection import (
    maskrcnn_resnet50_fpn,
    maskrcnn_resnet50_fpn_v2,
)
from torch import Tensor

from type import MODELTYPE
from type import MASKRCNNTYPE
from type import DEEPLABTYPE
from .deeplabv3 import DeepLabHeadWithCbam
from entity import (
    MaskRcnnEntity,
    Deeplabv3Entity
)

from .fusion import SensorVisionFusion


def _new_forward(self, images: Tensor, sensors: List) -> Dict[str, Tensor]:
    input_shape = images.shape[-2:]
    # contract: features is a dict of tensors
    vi_features = self.backbone(images)

    fused_features = self.fusion(vi_features["out"], sensors)
    result = OrderedDict()
    # x = vi_features["out"]
    x = self.classifier(fused_features) # CBAM
    x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
    result["out"] = x

    if self.aux_classifier is not None:
        x = vi_features["aux"]
        fused_features = self.aux_fusion(vi_features["aux"], sensors)
        x = self.aux_classifier(fused_features)
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
        result["aux"] = x

    return result


def __get_deeplab_model(config: Deeplabv3Entity):
    if config.type == DEEPLABTYPE.RESNET50:
        model = deeplabv3_resnet50(**config.params)
        if config.use_cbam:
            model.classifier = DeepLabHeadWithCbam(2048, config.num_classes)
            model.fusion = SensorVisionFusion()
            model.aux_fusion = SensorVisionFusion(1024)
        else:
            model.classifier = DeepLabHead(2048, config.num_classes)
        
        model.forward = _new_forward
        return model
    elif config.type == DEEPLABTYPE.RESNET101:
        model = deeplabv3_resnet101(**config.params)
        if config.use_cbam:
            model.classifier = DeepLabHeadWithCbam(2048, config.num_classes)
        else:
            model.classifier = DeepLabHead(2048, config.num_classes)
        return model
    elif config.type == DEEPLABTYPE.MOBILENET:
        model = deeplabv3_mobilenet_v3_large(**config.params)
        if config.use_cbam:
            model.classifier = DeepLabHeadWithCbam(2048, config.num_classes)
        else:
            model.classifier = DeepLabHead(2048, config.num_classes)
        return model
    else:
        raise ValueError(f"DeepLabV3 {type} is not supported.")


def __get_maskrcnn_model(config: MaskRcnnEntity):
    if config.type == MASKRCNNTYPE.RESNET50V1:
        return maskrcnn_resnet50_fpn(**config.params)
    elif config.type == maskrcnn_resnet50_fpn_v2:
        return maskrcnn_resnet50_fpn_v2(**config.params)
    else:
        raise ValueError(f"MaskRCNN {type} type is not supported.")


def get_model(model_type, config):
    if model_type == MODELTYPE.DEEPLABV3:
        return __get_deeplab_model(config)
    elif model_type == MODELTYPE.MASKRCNN:
        return __get_maskrcnn_model(config)
    else:
        raise NotImplementedError(f"{model_type} is not supported.")