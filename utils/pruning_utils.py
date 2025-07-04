import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
from tqdm import tqdm
import numpy as np
from collections import defaultdict

# Import configuration
import sys
import os
from typing import Any, List, Dict, Callable, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from tqdm import tqdm

# Try to import PRUNING_CFG, but don't fail if it's not available yet
try:
    from config.model_cfg import PRUNING_CFG
except ImportError:
    # Define a default config if not available
    class PRUNING_CFG:
        enabled = True
        initial_sparsity = 0.1
        target_sparsity = 0.7
        prune_epochs = []
        pruning_method = 'l2_structured'
        dim = 0
        global_pruning = True
        exclude_layers = ['classifier', 'fusion', 'aux_classifier']
        skip_1x1_convs = True

class ModelPruner:
    """
    A utility class for pruning PyTorch models with support for both
    structured and unstructured pruning strategies.
    
    Structured pruning is recommended for GPUs as it creates regular sparse patterns
    that can be efficiently utilized by hardware for speed-ups.
    """
    
    def __init__(self, model: nn.Module, config: Any = PRUNING_CFG):
        """
        Initialize the ModelPruner.
        
        Args:
            model: The PyTorch model to be pruned
            config: Pruning configuration (defaults to PRUNING_CFG from config)
        """
        self.model = model
        self.config = config
        self.pruning_parameters = []
        self.layer_sparsity = {}
        self._initialize_pruning_parameters()
        
    def _initialize_pruning_parameters(self) -> None:
        """Initialize pruning parameters based on model architecture."""
        self.pruning_parameters = []
        
        # Default layer types to prune if not specified
        layer_types = [nn.Conv2d, nn.Linear]
        
        # Find all prunable layers
        for name, module in self.model.named_modules():
            # Skip excluded layers
            if any(excluded in name for excluded in self.config.exclude_layers):
                continue
                
            # Check if layer type should be pruned
            if any(isinstance(module, layer_type) for layer_type in layer_types):
                # Skip 1x1 convolutions in residual connections if needed
                if isinstance(module, nn.Conv2d) and all(k == 1 for k in module.kernel_size):
                    if self.config.skip_1x1_convs:
                        continue
                
                self.pruning_parameters.append({
                    'module': module,
                    'name': name,
                    'type': type(module).__name__,
                    'shape': module.weight.shape,
                    'dim': self._get_pruning_dim(module)
                })
                
        print(f"Found {len(self.pruning_parameters)} prunable layers")
    
    def _get_pruning_dim(self, module: nn.Module) -> int:
        """Get the pruning dimension based on layer type and config."""
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            # For Conv2d, dimension 0 is output channels, 1 is input channels
            return 0  # Default to filter pruning (output channels)
        elif isinstance(module, nn.Linear):
            # For Linear, dimension 0 is output features, 1 is input features
            return 0  # Default to output features pruning
        return 0  # Default fallback
    
    def global_prune(self, amount: float = None, 
                    pruning_method: str = None,
                    verbose: bool = True) -> None:
        """
        Apply global pruning across all specified layers.
        
        Args:
            amount: Fraction of connections to prune (0.0 - 1.0). If None, uses config value.
            pruning_method: Pruning method ('l1_unstructured', 'l2_structured', etc.)
                           If None, uses config value.
            verbose: Whether to print pruning details
        """
        if amount is None:
            amount = self.config.target_sparsity
            
        if pruning_method is None:
            pruning_method = self.config.pruning_method
            
        if not self.pruning_parameters:
            self._initialize_pruning_parameters()
        
        # For structured pruning, we need to handle it differently
        if 'structured' in pruning_method:
            self._structured_prune(amount=amount, pruning_method=pruning_method, verbose=verbose)
        else:
            # Original unstructured pruning logic
            parameters_to_prune = [
                (param['module'], 'weight') for param in self.pruning_parameters
            ]
            
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=pruning_method,
                amount=amount,
            )
            
            if verbose:
                self.print_pruning_stats()
    
    def _structured_prune(self, amount: float, pruning_method: str, verbose: bool = True) -> None:
        """
        Apply structured pruning to the model.
        
        Args:
            amount: Fraction of connections to prune (0.0 - 1.0)
            pruning_method: Pruning method ('l2_structured', 'l1_structured', etc.)
            verbose: Whether to print pruning details
        """
        if verbose:
            print(f"\nApplying {pruning_method} structured pruning to {len(self.pruning_parameters)} layers")
            print(f"Target sparsity: {amount:.1%}")
        
        # For global structured pruning, we need to collect all parameters first
        if self.config.global_pruning:
            all_weights = []
            param_groups = []
            
            # Collect all weights for global pruning
            for param in self.pruning_parameters:
                module = param['module']
                weight = module.weight.detach().cpu().numpy()
                dim = param['dim']
                
                # For structured pruning, we need to compute the norm along the pruning dimension
                if dim == 0:  # Output channels (filters)
                    norm = np.linalg.norm(weight.reshape(weight.shape[0], -1), ord=2, axis=1)
                else:  # Input channels
                    norm = np.linalg.norm(weight.reshape(weight.shape[0], -1), ord=2, axis=0)
                
                all_weights.append(norm)
                param_groups.append((module, dim, weight.shape[dim]))
            
            # Concatenate all norms and find global threshold
            all_weights = np.concatenate(all_weights)
            threshold = np.percentile(all_weights, amount * 100)
            
            # Apply pruning based on global threshold
            for (module, dim, size), norms in zip(param_groups, all_weights):
                # Create binary mask (1 = keep, 0 = prune)
                mask = torch.ones(size, dtype=torch.float32, device=module.weight.device)
                mask[norms < threshold] = 0
                
                # Apply the mask
                if dim == 0:  # Output channels
                    module.weight.data = module.weight.data * mask.view(-1, 1, 1, 1)
                else:  # Input channels
                    module.weight.data = module.weight.data * mask.view(1, -1, 1, 1)
                
                # Update layer sparsity
                self.layer_sparsity[module] = 1.0 - float(mask.sum().item()) / mask.numel()
        else:
            # Layer-wise structured pruning
            for param in self.pruning_parameters:
                module = param['module']
                weight = module.weight.detach()
                dim = param['dim']
                
                # Compute L2 norm along the pruning dimension
                if dim == 0:  # Output channels (filters)
                    norm = torch.norm(weight.view(weight.size(0), -1), p=2, dim=1)
                else:  # Input channels
                    norm = torch.norm(weight.view(weight.size(0), -1), p=2, dim=0)
                
                # Determine number of channels to keep
                num_prune = int(len(norm) * amount)
                if num_prune == 0:
                    continue
                    
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
                
                # Update layer sparsity
                self.layer_sparsity[module] = 1.0 - float(mask.sum().item()) / mask.numel()
        
        if verbose:
            self.print_pruning_stats()
    
    def layerwise_prune(self, layer_amounts: Dict[str, float], 
                       pruning_method: str = None,
                       verbose: bool = True) -> None:
        """
        Apply different pruning amounts to specific layers.
        
        Args:
            layer_amounts: Dictionary mapping layer names to pruning amounts (0.0 - 1.0)
            pruning_method: Pruning method to use (defaults to config value)
            verbose: Whether to print pruning details
        """
        if pruning_method is None:
            pruning_method = self.config.pruning_method
            
        if not self.pruning_parameters:
            self._initialize_pruning_parameters()
        
        # Convert layer names to modules
        name_to_module = {param['name']: param['module'] for param in self.pruning_parameters}
        
        for name, amount in layer_amounts.items():
            if name in name_to_module:
                module = name_to_module[name]
                
                if 'structured' in pruning_method:
                    # Handle structured pruning
                    param = next(p for p in self.pruning_parameters if p['name'] == name)
                    self._structured_prune_single_layer(
                        module=module,
                        amount=amount,
                        pruning_method=pruning_method,
                        dim=param['dim']
                    )
                else:
                    # Handle unstructured pruning
                    prune.l1_unstructured(
                        module, 
                        name='weight', 
                        amount=amount
                    )
                
                # Update layer sparsity
                self.layer_sparsity[module] = amount
        
        if verbose:
            self.print_pruning_stats()
    
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
