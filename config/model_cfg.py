import torch
from type import (
    MODELTYPE,
    DEEPLABTYPE,
    MASKRCNNTYPE,
)

from entity import (
    Deeplabv3Entity,
    MaskRcnnEntity,
    DataEntity,
    DataloaderEntity,
    TrainEntity,
)

from entity.utils.dataloader_entity import DataloaderEntity
from entity.utils.train_entity import TrainEntity
from entity.utils.data_entity import DataEntity
from entity.utils.model_entity import ModelEntity
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

classes = 5
DEVICE = "cuda:0"
MODEL_TYPE = MODELTYPE.DEEPLABV3
SENSOR_RATIO = 0.3
RESIZE_SCALE = 65 # backbone feature map width, height

DATA_CFG = DataEntity(
    root_path="/home/allbigdat/data",
    image_base_path="images/",
    train_anno_path="COCO/train_without_street.json",
    valid_anno_path="COCO/valid_without_street.json",
    test_anno_path="COCO/test_without_street.json",
)


TRAIN_DATALOADER_CFG = DataloaderEntity(
    #batch_size=16,
    batch_size=8,
    num_worker=0,
    shuffle=True
)

VALID_DATALOADER_CFG = DataloaderEntity(
    #batch_size=16,
    batch_size=8,
    num_worker=0,   
    shuffle=False
)

TEST_DATALOADER_CFG = DataloaderEntity(
    #batch_size=6,
    batch_size=4,
    num_worker=0,
    shuffle=False
)


class PruningConfig(BaseModel):
    """
    Configuration for model pruning.
    
    Structured pruning is recommended for GPUs as it creates regular sparse patterns
    that can be efficiently utilized by hardware for speed-ups.
    """
    enabled: bool = True
    # Initial sparsity (0 = no pruning, 1 = all weights pruned)
    initial_sparsity: float = 0.1
    # Target sparsity to reach by the end of training (0.0 - 1.0)
    target_sparsity: float = 0.7
    # Which epochs to apply pruning (empty = auto-calculate based on num_epochs)
    prune_epochs: List[int] = []
    
    # Pruning method:
    # - For structured pruning: 'l2_structured', 'l1_structured', 'ln_structured'
    # - For unstructured pruning: 'l1_unstructured', 'random_unstructured'
    pruning_method: str = 'l2_structured'
    
    # Dimension(s) to prune for structured pruning (0: filter pruning, 1: channel pruning)
    # For Conv2d: 0=output channels, 1=input channels
    # For Linear: 0=output features, 1=input features
    dim: int = 0
    
    # Whether to use global pruning (across all layers) or layer-wise
    global_pruning: bool = True
    
    # Layers to specifically target for pruning (empty = all eligible layers)
    layers_to_prune: List[str] = []
    
    # Layers to exclude from pruning (e.g., final classification layers)
    exclude_layers: List[str] = ['classifier', 'fusion', 'aux_classifier']
    
    # Whether to perform iterative pruning (recommended for better accuracy)
    # If True, will gradually increase sparsity over training
    iterative_pruning: bool = True
    
    # Number of warmup epochs before starting pruning
    warmup_epochs: int = 1
    
    # Whether to fine-tune after pruning (recommended)
    fine_tune_after_pruning: bool = True
    # Number of fine-tuning epochs after reaching target sparsity
    fine_tune_epochs: int = 2


# Pruning configuration
PRUNING_CFG = PruningConfig()


TRAIN_CFG = TrainEntity(
    accum_step=1,
    num_epoch=10,
    log_step=20,
)


# TODO: Augmentation 고려
TRAIN_PIPE = [
    dict(type="ToPILImage", params=dict()),
    dict(type="Resize", params=dict(size=(520, 520))),
    dict(type="ToTensor", params=dict()),
]

VALID_PIPE = [
    dict(type="ToPILImage", params=dict()),
    dict(type="Resize", params=dict(size=(520, 520))),
    dict(type="ToTensor", params=dict()),
]


TEST_PIPE = [
    dict(type="ToPILImage", params=dict()),
    dict(type="Resize", params=dict(size=(520, 520))),
    dict(type="ToTensor", params=dict()),
]

MODEL_CFG = {
    MODELTYPE.DEEPLABV3: Deeplabv3Entity(
        type=DEEPLABTYPE.RESNET50,
        num_classes=1,
        use_cbam=True,
        params=dict(
            pretrained=True,
            # Ensure batch norm uses running stats for small batch sizes
            norm_layer=lambda *args, **kwargs: torch.nn.BatchNorm2d(
                *args, **kwargs, momentum=0.1, eps=1e-5
            )
        ),
        load_from="/home/allbigdat/workspace/cls_load/resource/cls_weights.pth"
    ),
    MODELTYPE.MASKRCNN: MaskRcnnEntity(
        type=MASKRCNNTYPE.RESNET50V2,
        params=dict(
            num_classes=classes
        ),
    )
}[MODEL_TYPE]
