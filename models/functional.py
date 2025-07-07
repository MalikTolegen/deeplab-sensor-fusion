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


def _new_forward(self, images: Tensor, sensors: List = None) -> Dict[str, Tensor]:
    input_shape = images.shape[-2:]
    # contract: features is a dict of tensors
    vi_features = self.backbone(images)

    # Handle case where sensors might be None
    if sensors is None:
        x = vi_features["out"]
    else:
        fused_features = self.fusion(vi_features["out"], sensors)
        x = self.classifier(fused_features)  # CBAM
    
    x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
    result = {"out": x}

    if hasattr(self, 'aux_classifier') and self.aux_classifier is not None:
        aux = vi_features.get("aux")
        if aux is not None:
            if sensors is not None and hasattr(self, 'aux_fusion'):
                aux = self.aux_fusion(aux, sensors)
            aux = self.aux_classifier(aux)
            aux = F.interpolate(aux, size=input_shape, mode="bilinear", align_corners=False)
            result["aux"] = aux

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
        
        # Use MethodType to properly bind the method to the instance
        from types import MethodType
        model.forward = MethodType(_new_forward, model)
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
    print(f"[DEBUG] model_type: {model_type}, type: {type(model_type)}")
    print(f"[DEBUG] MODELTYPE.DEEPLABV3: {MODELTYPE.DEEPLABV3}, type: {type(MODELTYPE.DEEPLABV3)}")
    print(f"[DEBUG] model_type == MODELTYPE.DEEPLABV3: {model_type == MODELTYPE.DEEPLABV3}")
    print(f"[DEBUG] model_type.value: {model_type.value if hasattr(model_type, 'value') else 'N/A'}")
    print(f"[DEBUG] MODELTYPE.DEEPLABV3.value: {MODELTYPE.DEEPLABV3.value}")
    
    # Try comparing by value as well
    if model_type == MODELTYPE.DEEPLABV3 or model_type.value == MODELTYPE.DEEPLABV3.value:
        return __get_deeplab_model(config)
    elif model_type == MODELTYPE.MASKRCNN or model_type.value == MODELTYPE.MASKRCNN.value:
        return __get_maskrcnn_model(config)
    else:
        raise NotImplementedError(f"{model_type} (value: {model_type.value if hasattr(model_type, 'value') else 'N/A'}) is not supported. Available types: {list(MODELTYPE.__members__.keys())}")