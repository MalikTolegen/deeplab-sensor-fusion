from enum import Enum


class MODELTYPE(Enum):
    DEEPLABV3 = 1
    MASKRCNN = 2
    DEEPLABV3_SBAM = 3


class DEEPLABTYPE(Enum):
    RESNET50 = 1
    RESNET101 = 2
    MOBILENET = 3


class MASKRCNNTYPE(Enum):
    RESNET50V1 = 1
    RESNET50V2 = 2
