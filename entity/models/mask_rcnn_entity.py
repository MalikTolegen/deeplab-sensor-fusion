from .model_base_entity import BaseModelEntity

from type import MASKRCNNTYPE


class MaskRcnnEntity(BaseModelEntity):
    type: MASKRCNNTYPE = MASKRCNNTYPE.RESNET50V1
