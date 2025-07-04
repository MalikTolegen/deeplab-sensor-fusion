"""Pruning utilities for model optimization."""
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from typing import List, Dict, Any, Optional, Callable, Union, Tuple
import math
from dataclasses import dataclass, field
from tqdm import tqdm
from collections import defaultdict

@dataclass
class PruningStats:
    """Class to track pruning statistics."""
    total_params: int = 0
    pruned_params: int = 0
    layer_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    @property
    def sparsity(self) -> float:
        """Calculate overall sparsity."""
        return self.pruned_params / self.total_params if self.total_params > 0 else 0.0
    
    def add_layer(self, name: str, total: int, pruned: int) -> None:
        """Add layer statistics."""
        self.total_params += total
        self.pruned_params += pruned
        self.layer_stats[name] = {
            'total': total,
            'pruned': pruned,
            'sparsity': pruned / total if total > 0 else 0.0
        }
    
    def format_stats(self) -> str:
        """Format statistics as a string."""
        lines = [
            f"Total Parameters: {self.total_params:,}",
            f"Pruned Parameters: {self.pruned_params:,}",
            f"Sparsity: {self.sparsity:.2%}",
            "\nLayer-wise Statistics:"
        ]
        
        for name, stats in self.layer_stats.items():
            lines.append(
                f"  {name}: {stats['pruned']:,}/{stats['total']:,} "
                f"({stats['sparsity']:.2%}) pruned"
            )
            
        return "\n".join(lines)

class ModelPruner:
    """A class to handle model pruning with various strategies."""
    
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        """Initialize the ModelPruner.
        
        Args:
            model: The PyTorch model to prune
            config: Configuration dictionary containing pruning parameters
        """
        self.model = model
        self.config = config
        self.pruning_parameters = []
        self.pruning_stats = PruningStats()
        self._initialize_pruning_parameters()
    
    def _initialize_pruning_parameters(self) -> None:
        """Initialize parameters that will be pruned."""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                # Skip output layers
                if any(x in name for x in ['classifier', 'fc', 'head', 'aux']):
                    continue
                # Skip excluded layers
                if any(excluded in name for excluded in self.config.get('exclude_layers', [])):
                    continue
                self.pruning_parameters.append((name, module, 'weight'))
    
    def get_pruning_hook(self) -> Callable:
        """Get a hook function that can be called during training."""
        def pruning_hook(epoch: int, **kwargs) -> None:
            """Prune the model based on the current epoch.
            
            Args:
                epoch: Current training epoch
            """
            if not self.config.get('enabled', False):
                return
                
            start_epoch = self.config.get('start_epoch', 0)
            end_epoch = self.config.get('end_epoch', 30)
            frequency = self.config.get('frequency', 1)
            
            if (epoch < start_epoch) or (epoch > end_epoch) or ((epoch - start_epoch) % frequency != 0):
                return
                
            # Calculate current target sparsity
            progress = min(1.0, (epoch - start_epoch) / (end_epoch - start_epoch))
            current_sparsity = self.config.get('initial_sparsity', 0.1) + \
                             (self.config.get('target_sparsity', 0.5) - 
                              self.config.get('initial_sparsity', 0.1)) * progress
            
            # Apply pruning
            self.prune_model(current_sparsity)
            
            # Print pruning info
            print(f"\nPruning at epoch {epoch}: Target sparsity = {current_sparsity:.2f}")
            print(self.pruning_stats.format_stats())
        
        return pruning_hook
    
    def prune_model(self, target_sparsity: float) -> None:
        """Prune the model to the target sparsity.
        
        Args:
            target_sparsity: Target sparsity ratio (0-1)
        """
        prune_type = self.config.get('prune_type', 'l1_unstructured')
        global_pruning = self.config.get('global_pruning', True)
        
        # Reset stats
        self.pruning_stats = PruningStats()
        
        if global_pruning:
            self._global_pruning(prune_type, target_sparsity)
        else:
            self._local_pruning(prune_type, target_sparsity)
    
    def _global_pruning(self, prune_type: str, target_sparsity: float) -> None:
        """Apply global pruning across all parameters."""
        parameters_to_prune = [
            (module, 'weight') for _, module, _ in self.pruning_parameters
        ]
        
        if prune_type == 'l1_unstructured':
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=target_sparsity
            )
        elif prune_type == 'l2_structured':
            prune.global_structured(
                parameters_to_prune,
                pruning_method=prune.LnStructured,
                amount=target_sparsity,
                dim=self.config.get('prune_dim', 0),
                n=2
            )
        elif prune_type == 'ln_structured':
            prune.global_structured(
                parameters_to_prune,
                pruning_method=prune.LnStructured,
                amount=target_sparsity,
                dim=self.config.get('prune_dim', 0),
                n=1
            )
        
        # Update statistics
        self._update_pruning_stats()
    
    def _local_pruning(self, prune_type: str, target_sparsity: float) -> None:
        """Apply local pruning to each parameter individually."""
        for name, module, _ in self.pruning_parameters:
            if prune_type == 'l1_unstructured':
                prune.l1_unstructured(module, 'weight', amount=target_sparsity)
            elif prune_type == 'l2_structured':
                prune.ln_structured(
                    module, 'weight', 
                    amount=target_sparsity,
                    dim=self.config.get('prune_dim', 0),
                    n=2
                )
            elif prune_type == 'ln_structured':
                prune.ln_structured(
                    module, 'weight',
                    amount=target_sparsity,
                    dim=self.config.get('prune_dim', 0),
                    n=1
                )
        
        # Update statistics
        self._update_pruning_stats()
    
    def _update_pruning_stats(self) -> None:
        """Update pruning statistics."""
        for name, module, param_name in self.pruning_parameters:
            param = getattr(module, param_name)
            mask = getattr(module, f"{param_name}_mask", None)
            
            if mask is not None:
                total = param.numel()
                pruned = int(torch.sum(mask == 0).item())
                self.pruning_stats.add_layer(name, total, pruned)
    
    def apply_pruning(self) -> None:
        """Apply pruning masks permanently by removing the reparameterization."""
        for _, module, param_name in self.pruning_parameters:
            prune.remove(module, param_name)
    
    def get_pruning_summary(self) -> Dict[str, Any]:
        """Get a summary of pruning statistics."""
        return {
            'total_parameters': self.pruning_stats.total_params,
            'pruned_parameters': self.pruning_stats.pruned_params,
            'sparsity': self.pruning_stats.sparsity,
            'layer_stats': self.pruning_stats.layer_stats
        }
    def _structured_prune_single_layer(self, module: nn.Module, amount: float, 
                                     pruning_method: str, dim: int) -> None:
        """
        Apply structured pruning to a single layer.
        
        Args:
            module: The module to prune
            amount: Fraction of connections to prune (0.0 - 1.0)
            pruning_method: Pruning method ('l2_structured', 'l1_structured', etc.)
            dim: Dimension to prune (0 for output channels/features, 1 for input)
        """
        weight = module.weight.detach()
        
        # Compute norm along the pruning dimension
        if dim == 0:  # Output channels (filters)
            norm = torch.norm(weight.view(weight.size(0), -1), p=2, dim=1)
        else:  # Input channels
            norm = torch.norm(weight.view(weight.size(0), -1), p=2, dim=0)
        
        # Determine number of channels to keep
        num_prune = int(len(norm) * amount)
        if num_prune == 0:
            return
            
        # Get threshold
        threshold = torch.kthvalue(norm, num_prune).values
        
        # Create mask
        mask = torch.ones_like(norm, device=weight.device)
        mask[norm <= threshold] = 0
        
        # Apply mask
        if dim == 0:  # Output channels
            module.weight.data = module.weight.data * mask.view(-1, 1, 1, 1)
        else:  # Input channels
            module.weight.data = module.weight.data * mask.view(1, -1, 1, 1)
    
    def remove_pruning(self) -> None:
        """Remove pruning reparameterization from all modules."""
        for module, param_name, _ in self.pruning_parameters:
            try:
                prune.remove(module, param_name)
            except ValueError:
                # Layer wasn't pruned, skip
                continue
    
    def get_sparsity(self, verbose: bool = False) -> Dict[str, float]:
        """
        Calculate sparsity of prunable layers.
        
        Args:
            verbose: Whether to print detailed sparsity information
            
        Returns:
            Dictionary mapping layer names to their sparsity (0.0 - 1.0)
        """
        sparsity = {}
        total_params = 0
        total_zeros = 0
        
        for param in self.pruning_parameters:
            module = param['module']
            name = param['name']
            weight = module.weight
            
            # Calculate sparsity for this layer
            zeros = torch.sum(weight == 0).item()
            total = weight.numel()
            layer_sparsity = zeros / total
            
            sparsity[name] = layer_sparsity
            total_zeros += zeros
            total_params += total
            
            if verbose:
                print(f"{name}: {layer_sparsity:.1%} sparsity ({zeros}/{total} zeros)")
        
        if verbose and total_params > 0:
            print(f"\nOverall model sparsity: {total_zeros/total_params:.1%} "
                  f"({total_zeros:,} / {total_params:,} parameters)")
        
        return sparsity
    
    def print_pruning_stats(self) -> None:
        """Print detailed statistics about pruned layers."""
        print("\n" + "="*70)
        print("PRUNING STATISTICS")
        print("="*70)
        
        # Get layer-wise sparsity
        sparsity = self.get_sparsity(verbose=False)
        
        # Print layer statistics
        print("\nLayer-wise Sparsity:")
        print("-" * 70)
        print(f"{'Layer':<50} | {'Sparsity':>10} | {'Params':>12} | {'Zeros':>12}")
        print("-" * 70)
        
        total_params = 0
        total_zeros = 0
        
        for name, layer_sparsity in sparsity.items():
            # Get parameter count for this layer
            for param in self.pruning_parameters:
                if param['name'] == name:
                    param_count = param['module'].weight.numel()
                    zero_count = int(param_count * layer_sparsity)
                    total_params += param_count
                    total_zeros += zero_count
                    
                    print(f"{name[:48]:<50} | {layer_sparsity:>9.1%} | "
                          f"{param_count:>12,} | {zero_count:>12,}")
                    break
        
        # Print summary
        print("-" * 70)
        print(f"{'TOTAL':<50} | {'':>10} | {total_params:>12,} | {total_zeros:>12,}")
        print(f"{'Overall Sparsity':<50} | {total_zeros/total_params:>9.1%} | "
              f"{'':>12} | {'':>12}")
        print("="*70 + "\n")

def apply_pruning_schedule(
    model: nn.Module,
    epochs: int,
    initial_sparsity: float = None,
    target_sparsity: float = None,
    prune_epochs: List[int] = None,
    config: Any = None
) -> Tuple[ModelPruner, Callable]:
    """
    Apply a gradual pruning schedule over training.
    
    Args:
        model: Model to prune
        epochs: Total number of training epochs
        initial_sparsity: Initial sparsity (0.0 - 1.0). If None, uses config value.
        target_sparsity: Target sparsity (0.0 - 1.0). If None, uses config value.
        prune_epochs: List of epochs when to apply pruning. If None, auto-calculates.
        config: Pruning configuration. If None, uses PRUNING_CFG.
        
    Returns:
        Tuple of (ModelPruner instance, pruning_hook function)
    """
    if config is None:
        from config.model_cfg import PRUNING_CFG
        config = PRUNING_CFG
    
    if initial_sparsity is None:
        initial_sparsity = config.initial_sparsity
    if target_sparsity is None:
        target_sparsity = config.target_sparsity
    
    # Initialize pruner
    pruner = ModelPruner(model, config)
    
    # Set up pruning schedule
    if prune_epochs is None:
        # Default: start after warmup and prune every 2 epochs
        start_epoch = config.warmup_epochs
        end_epoch = epochs - (config.fine_tune_epochs if config.fine_tune_after_pruning else 0)
        prune_epochs = list(range(start_epoch, end_epoch, 2))
    
    # Create pruning hook function
    def pruning_hook(epoch: int, **kwargs) -> None:
        if epoch in prune_epochs:
            # Calculate current target sparsity
            progress = (prune_epochs.index(epoch) + 1) / len(prune_epochs)
            # Use cubic schedule for smoother pruning
            target = initial_sparsity + (target_sparsity - initial_sparsity) * (progress ** 3)
            
            print(f"\n{'='*50}")
            print(f"PRUNING EPOCH {epoch}: Target sparsity = {target:.1%}")
            print(f"{'='*50}")
            
            # Apply pruning
            pruner.global_prune(amount=target, verbose=True)
            
            # Print model size and FLOPs if available
            try:
                from thop import profile
                input_size = (1, 3, 512, 512)  # Adjust based on your input size
                input_tensor = torch.randn(input_size).to(next(model.parameters()).device)
                macs, params = profile(model, inputs=(input_tensor,), verbose=False)
                print(f"\nModel size: {params/1e6:.2f}M parameters")
                print(f"FLOPs: {macs/1e9:.2f} GFLOPs")
            except ImportError:
                pass
    """
    Apply gradual pruning schedule over training.
    
    Args:
        model: Model to prune
        epochs: Total number of training epochs
        initial_sparsity: Initial sparsity (0.0 - 1.0)
        target_sparsity: Target sparsity (0.0 - 1.0)
        prune_epochs: List of epochs when to apply pruning (default: every 2 epochs)
        
    Returns:
        ModelPruner instance for further pruning operations
    """
    if prune_epochs is None:
        # Default: prune every 2 epochs
        prune_epochs = list(range(0, epochs, 2))
    
    pruner = ModelPruner(model)
    pruner.prepare_for_pruning()
    
    def pruning_hook(epoch: int, **kwargs) -> None:
        if epoch in prune_epochs:
            # Calculate current target sparsity
            progress = (prune_epochs.index(epoch) + 1) / len(prune_epochs)
            target = initial_sparsity + (target_sparsity - initial_sparsity) * (progress ** 3)
            
            print(f"\nPruning at epoch {epoch} to {target:.1%} sparsity")
            pruner.global_prune(amount=target)
    
    return pruner, pruning_hook
