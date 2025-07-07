import torch
from sklearn.metrics import jaccard_score, f1_score, precision_score, recall_score
import numpy as np

def calculate_metrics(preds, masks, threshold=0.5):
    """Calculate segmentation metrics.
    
    Args:
        preds: Model predictions (logits)
        masks: Ground truth masks
        threshold: Threshold for binary classification
        
    Returns:
        Dictionary containing metrics
    """
    # Convert to numpy if they're tensors
    if torch.is_tensor(preds):
        preds = preds.detach().cpu().numpy()
    if torch.is_tensor(masks):
        masks = masks.detach().cpu().numpy()
    
    # Flatten for sklearn metrics
    preds_flat = (preds > threshold).astype(np.int32).reshape(-1)
    masks_flat = masks.astype(np.int32).reshape(-1)
    
    # Calculate metrics
    metrics = {
        'iou': jaccard_score(masks_flat, preds_flat, zero_division=0),
        'f1': f1_score(masks_flat, preds_flat, zero_division=0),
        'precision': precision_score(masks_flat, preds_flat, zero_division=0),
        'recall': recall_score(masks_flat, preds_flat, zero_division=0)
    }
    return metrics

def analyze_pruning(model):
    """Analyze model sparsity after pruning."""
    total_params = 0
    zero_params = 0
    
    for name, param in model.named_parameters():
        if 'weight' in name and len(param.shape) > 1:  # Only weight matrices
            total_params += param.numel()
            zero_params += (param == 0).sum().item()
    
    sparsity = 100 * zero_params / total_params if total_params > 0 else 0
    return {
        'total_params': total_params,
        'zero_params': zero_params,
        'sparsity': sparsity
    }
