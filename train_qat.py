"""
Quantization-Aware Training (QAT) script for DeepLabV3 with CBAM.
"""
import os
import copy
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from config.model_cfg import (
    MODEL_CFG, TRAIN_CFG, DATASET_CFG, 
    PRUNING_CFG, QUANTIZATION_CFG
)
from models import get_model
from utils.data_loader import get_dataloaders
from utils.metrics import meanIoU
from utils.quantization_utils import (
    prepare_model_for_quantization,
    save_quantized_model,
    print_quantization_summary
)


def train_one_epoch(
    model, 
    train_loader, 
    criterion, 
    optimizer, 
    device, 
    scaler=None,
    grad_clip=None,
    print_freq=10
):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0.0
    total_iou = 0.0
    num_batches = len(train_loader)
    
    for i, (images, masks, _) in enumerate(tqdm(train_loader, desc="Training")):
        images = images.to(device)
        masks = masks.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        with autocast(enabled=scaler is not None):
            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            
            # Calculate mIoU
            preds = torch.argmax(outputs, dim=1)
            iou = meanIoU(preds, masks, num_classes=MODEL_CFG.num_classes)
        
        # Backward pass and optimize
        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        
        # Update statistics
        total_loss += loss.item()
        total_iou += iou.item()
        
        # Print statistics
        if (i + 1) % print_freq == 0:
            avg_loss = total_loss / (i + 1)
            avg_iou = total_iou / (i + 1)
            print(f"Batch [{i+1}/{num_batches}], "
                  f"Loss: {avg_loss:.4f}, mIoU: {avg_iou:.4f}")
    
    # Calculate epoch statistics
    avg_loss = total_loss / num_batches
    avg_iou = total_iou / num_batches
    
    return {
        'loss': avg_loss,
        'iou': avg_iou
    }


def validate(
    model, 
    val_loader, 
    criterion, 
    device,
    print_freq=10
):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    num_batches = len(val_loader)
    
    with torch.no_grad():
        for i, (images, masks, _) in enumerate(tqdm(val_loader, desc="Validation")):
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            
            # Calculate mIoU
            preds = torch.argmax(outputs, dim=1)
            iou = meanIoU(preds, masks, num_classes=MODEL_CFG.num_classes)
            
            # Update statistics
            total_loss += loss.item()
            total_iou += iou.item()
            
            # Print statistics
            if (i + 1) % print_freq == 0:
                avg_loss = total_loss / (i + 1)
                avg_iou = total_iou / (i + 1)
                print(f"Batch [{i+1}/{num_batches}], "
                      f"Val Loss: {avg_loss:.4f}, Val mIoU: {avg_iou:.4f}")
    
    # Calculate epoch statistics
    avg_loss = total_loss / num_batches
    avg_iou = total_iou / num_batches
    
    return {
        'loss': avg_loss,
        'iou': avg_iou
    }


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Quantization-Aware Training')
    parser.add_argument('--resume', type=str, default='',
                        help='path to checkpoint to resume from')
    parser.add_argument('--eval', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--export', action='store_true',
                        help='export quantized model')
    parser.add_argument('--output-dir', default='./output',
                        help='directory to save outputs')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get data loaders
    train_loader, val_loader, _ = get_dataloaders(
        root=DATASET_CFG.root_dir,
        batch_size=TRAIN_CFG.batch_size,
        num_workers=TRAIN_CFG.num_workers,
        pin_memory=True,
        resize_size=DATASET_CFG.resize_size,
        crop_size=DATASET_CFG.crop_size
    )
    
    # Create model
    model = get_model(MODEL_CFG.type, MODEL_CFG)
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.Adam(
        model.parameters(),
        lr=TRAIN_CFG.lr,
        weight_decay=TRAIN_CFG.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.1, patience=5, verbose=True
    )
    
    # Gradient scaler for mixed precision training
    scaler = GradScaler() if TRAIN_CFG.mixed_precision else None
    
    # Load checkpoint if resuming
    start_epoch = 0
    best_iou = 0.0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint['epoch']
            best_iou = checkpoint.get('best_iou', 0.0)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if scaler and 'scaler' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler'])
            print(f"Loaded checkpoint '{args.resume}' (epoch {start_epoch})")
        else:
            print(f"No checkpoint found at '{args.resume}'")
    
    # Prepare model for quantization-aware training
    if QUANTIZATION_CFG.enabled and not args.eval:
        print("Preparing model for quantization-aware training...")
        model = prepare_model_for_quantization(
            model,
            exclude_layers=QUANTIZATION_CFG.exclude_layers
        )
        print_quantization_summary(model)
    
    # Evaluate model if requested
    if args.eval:
        print("Evaluating model...")
        val_metrics = validate(model, val_loader, criterion, device)
        print(f"Validation Loss: {val_metrics['loss']:.4f}, "
              f"Validation mIoU: {val_metrics['iou']:.4f}")
        
        # Export quantized model if requested
        if args.export:
            print("Exporting quantized model...")
            example_input = next(iter(val_loader))[0][:1].to(device)
            save_path = os.path.join(args.output_dir, 'quantized_model.pt')
            save_quantized_model(model, save_path, example_input)
            print(f"Quantized model saved to {save_path}")
        
        return
    
    # Train the model
    print("Starting training...")
    for epoch in range(start_epoch, TRAIN_CFG.epochs):
        print(f"\nEpoch {epoch + 1}/{TRAIN_CFG.epochs}")
        
        # Train for one epoch
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, 
            device, scaler, TRAIN_CFG.grad_clip
        )
        
        # Evaluate on validation set
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_metrics['iou'])
        
        # Print epoch statistics
        print(f"Epoch {epoch + 1}:")
        print(f"  Train Loss: {train_metrics['loss']:.4f}, "
              f"Train mIoU: {train_metrics['iou']:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}, "
              f"Val mIoU: {val_metrics['iou']:.4f}")
        
        # Save checkpoint
        is_best = val_metrics['iou'] > best_iou
        best_iou = max(val_metrics['iou'], best_iou)
        
        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_iou': best_iou,
            'train_loss': train_metrics['loss'],
            'val_loss': val_metrics['loss'],
            'val_iou': val_metrics['iou']
        }
        
        if scaler is not None:
            checkpoint['scaler'] = scaler.state_dict()
        
        # Save latest checkpoint
        torch.save(
            checkpoint,
            os.path.join(args.output_dir, 'checkpoint_latest.pth.tar')
        )
        
        # Save best checkpoint
        if is_best:
            torch.save(
                checkpoint,
                os.path.join(args.output_dir, 'model_best.pth.tar')
            )
            
            # Export quantized model if this is the best model so far
            if QUANTIZATION_CFG.enabled and args.export:
                print("Exporting best quantized model...")
                example_input = next(iter(val_loader))[0][:1].to(device)
                save_path = os.path.join(args.output_dir, 'quantized_model_best.pt')
                save_quantized_model(model, save_path, example_input)
                print(f"Best quantized model saved to {save_path}")
    
    print("Training completed!")


if __name__ == '__main__':
    main()
