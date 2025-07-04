"""Configuration for model optimization techniques."""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PruningConfig:
    """Configuration for model pruning."""
    enabled: bool = False
    initial_sparsity: float = 0.1
    target_sparsity: float = 0.5
    start_epoch: int = 0
    end_epoch: int = 30
    frequency: int = 1
    prune_type: str = 'l1_unstructured'  # 'l1_unstructured', 'l2_structured', 'ln_structured'
    prune_dim: int = 0  # For structured pruning: 0 for filter pruning, 1 for channel pruning
    global_pruning: bool = True
    exclude_layers: List[str] = field(default_factory=list)  # Layers to exclude from pruning


@dataclass
class QuantizationConfig:
    """Configuration for model quantization."""
    enabled: bool = False
    qat: bool = True  # Quantization-Aware Training
    ptq: bool = False  # Post-Training Quantization
    dtype: str = 'int8'  # 'int8', 'float16'
    exclude_layers: List[str] = field(default_factory=list)  # Layers to exclude from quantization
    backend: str = 'fbgemm'  # 'fbgemm' for server, 'qnnpack' for mobile
    num_calibration_batches: int = 32  # For PTQ


@dataclass
class OptimizationConfig:
    """Main optimization configuration container."""
    pruning: PruningConfig = field(default_factory=PruningConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    
    # Gradient clipping
    grad_clip: Optional[float] = 1.0
    
    # Mixed precision training
    mixed_precision: bool = True
    
    # Batch size for optimization
    batch_size: int = 8
    
    def to_dict(self):
        """Convert config to dictionary."""
        return {
            'pruning': {k: v for k, v in self.pruning.__dict__.items() if not k.startswith('_')},
            'quantization': {k: v for k, v in self.quantization.__dict__.items() if not k.startswith('_')},
            'grad_clip': self.grad_clip,
            'mixed_precision': self.mixed_precision,
            'batch_size': self.batch_size
        }
