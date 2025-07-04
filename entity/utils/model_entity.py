from enum import Enum
from typing import Optional
from ..models.model_base_entity import BaseModelEntity
from ..models.deeplabv3_entity import Deeplabv3Entity
from ..models.mask_rcnn_entity import MaskRcnnEntity

class MODELTYPE(Enum):
    DEEPLABV3 = "deeplabv3"
    MASKRCNN = "mask_rcnn"

class ModelEntity(BaseModelEntity):
    """Model configuration entity."""
    type: MODELTYPE = MODELTYPE.DEEPLABV3
    deeplabv3: Optional[Deeplabv3Entity] = None
    mask_rcnn: Optional[MaskRcnnEntity] = None
    
    class Config:
        use_enum_values = True

# For backward compatibility
MODELTYPE = MODELTYPE
