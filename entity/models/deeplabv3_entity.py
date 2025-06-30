from typing import Optional

from .model_base_entity import BaseModelEntity

from type import DEEPLABTYPE


class Deeplabv3Entity(BaseModelEntity):
    type: DEEPLABTYPE = DEEPLABTYPE.RESNET50
    use_cbam: Optional[bool] = False