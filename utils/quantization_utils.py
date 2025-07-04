"""Utility functions for quantization-aware training and model optimization."""
import copy
import torch
import torch.nn as nn
import torch.quantization as quant
from typing import Optional, Dict, Any, List, Union, Tuple, Type
from dataclasses import dataclass, field
from tqdm import tqdm


def prepare_model_for_quantization(
    model: nn.Module,
    qconfig: Optional[torch.quantization.QConfig] = None,
    inplace: bool = True,
    exclude_layers: Optional[List[str]] = None,
) -> nn.Module:
    """
    Prepare a model for quantization-aware training.
    
    Args:
        model: The model to prepare for QAT
        qconfig: Quantization configuration
        inplace: Whether to modify the model in place
        exclude_layers: List of layer names to exclude from quantization
        
    Returns:
        Prepared model for QAT
    """
    if not inplace:
        model = copy.deepcopy(model)
    
    # Set default qconfig if not provided
    if qconfig is None:
        qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    
    # Set qconfig for the model and all submodules
    model.qconfig = qconfig
    
    # Prepare the model for QAT
    torch.quantization.prepare_qat(model, inplace=True)
    
    # Fuse model if it has a fuse_model method
    if hasattr(model, 'fuse_model'):
        model.fuse_model()
    
    return model


def quantize_model(
    model: nn.Module,
    calib_data_loader: torch.utils.data.DataLoader,
    num_calib_batches: int = 32,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    inplace: bool = True,
) -> nn.Module:
    """
    Quantize a trained model using post-training quantization.
    
    Args:
        model: The trained model to quantize
        calib_data_loader: DataLoader for calibration data
        num_calib_batches: Number of batches to use for calibration
        device: Device to run calibration on
        inplace: Whether to modify the model in place
        
    Returns:
        Quantized model
    """
    if not inplace:
        model = copy.deepcopy(model)
    
    # Set the model to evaluation mode
    model.eval()
    model.to(device)
    
    # Fuse model if it has a fuse_model method
    if hasattr(model, 'fuse_model'):
        model.fuse_model()
    
    # Set qconfig for quantization
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    
    # Prepare the model for calibration
    model = torch.quantization.prepare(model, inplace=inplace)
    
    # Calibrate the model
    print("Calibrating model...")
    with torch.no_grad():
        for i, (images, _) in enumerate(calib_data_loader):
            if i >= num_calib_batches:
                break
            model(images.to(device))
    
    # Convert to quantized model
    model = torch.quantization.convert(model, inplace=inplace)
    
    return model


def save_quantized_model(
    model: nn.Module,
    save_path: str,
    example_input: torch.Tensor,
    quantize: bool = True,
) -> None:
    """
    Save a quantized model using TorchScript.
    
    Args:
        model: The model to save
        save_path: Path to save the model
        example_input: Example input for tracing
        quantize: Whether to quantize the model before saving
    """
    model.eval()
    
    # Trace the model
    with torch.no_grad():
        if quantize:
            # Use torch.jit.trace for quantized models
            model = torch.quantization.convert(model, inplace=False)
            
        # Trace the model
        traced_model = torch.jit.trace(model, example_input)
        
        # Save the traced model
        torch.jit.save(traced_model, save_path)


def load_quantized_model(
    model_path: str,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
) -> torch.jit.ScriptModule:
    """
    Load a quantized model.
    
    Args:
        model_path: Path to the quantized model
        device: Device to load the model on
        
    Returns:
        Loaded quantized model
    """
    # Load the model
    model = torch.jit.load(model_path, map_location=device)
    
    # Set to eval mode
    model.eval()
    
    return model


def print_quantization_summary(model: nn.Module) -> None:
    """Print a summary of the model's quantization status."""
    print("Quantization Summary:")
    print("=" * 80)
    
    # Check if model is quantized
    is_quantized = any(
        hasattr(m, 'qconfig') and m.qconfig is not None
        for m in model.modules()
    )
    
    print(f"Model is {'quantized' if is_quantized else 'not quantized'}")
    
    # Print quantization info for each module
    for name, module in model.named_modules():
        if hasattr(module, 'qconfig') and module.qconfig is not None:
            print(f"\nModule: {name}")
            print(f"  qconfig: {module.qconfig}")
            
            # Print weight and activation observers if they exist
            if hasattr(module, 'weight_fake_quant'):
                print(f"  Weight fake quant: {module.weight_fake_quant}")
            if hasattr(module, 'activation_post_process'):
                print(f"  Activation post-process: {module.activation_post_process}")
