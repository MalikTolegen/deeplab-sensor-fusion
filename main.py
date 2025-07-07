import os
import json
import time
import datetime
from tqdm import tqdm
from pathlib import Path

import torch
from torch import nn
from torch.optim import SGD, Adam
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose
import cv2
import numpy as np
import matplotlib.pyplot as plt

from config.model_cfg import *
from models import get_model
from utils import ImgClsDataset
from utils import get_transforms
from utils.pruning_utils import apply_pruning_schedule
from utils.metrics import calculate_metrics, analyze_pruning
from models.loss import DiceFocalLoss
from util import base_collate_fn
from config.model_cfg import DEVICE


def get_loss_for_loss(losses: dict):
    log_losses = dict()
    for key, val in losses.items():
        log_losses[key] = val.detach().cpu().item()
    
    return log_losses


def get_eta(remaining_iters, time):
    total_time = remaining_iters * time
    days = int(total_time // 86400)
    hours = int((total_time - 86400 * days) // 3600)
    mins = int((total_time - (86400 * days) - (3600 * hours)) // 60)
    secs = int(total_time - (86400 * days) - (3600 * hours) - (60 * mins))

    eta = "eta: "
    if days != 0:
        if days == 1:
            eta += f"{days} day "
        else:
            eta += f"{days} days "

    eta += f"{hours}:{mins:02d}:{secs:02d}"

    return eta


def get_log_msg(log_msg, losses, remaining_iters):
    msg = log_msg
    remaining_iters = remaining_iters - TRAIN_CFG.log_step

    info = dict()
    for t in losses:
        for key, val in t.items():
            if key not in info:
                info[key] = 0
            info[key] += val
    mean_time = info['time'] / TRAIN_CFG.log_step
    ETA = get_eta(remaining_iters, mean_time)
    msg = msg + ETA + " "
    msg = msg + ', '.join([f"{key}: {val/TRAIN_CFG.log_step:.4f}" for key, val in info.items()])

    return msg, remaining_iters


def model_load(model):
    if MODEL_CFG.load_from is not None:
        # Load the state dict
        state_dict = torch.load(MODEL_CFG.load_from, map_location=DEVICE)
        
        # Handle pruned models by removing pruning buffers if they exist
        if any('weight_orig' in key or 'weight_mask' in key for key in state_dict.keys()):
            print("Loading pruned model, removing pruning buffers...")
            # Create a new state dict without pruning buffers
            new_state_dict = {}
            for key, value in state_dict.items():
                if 'weight_orig' in key:
                    # Convert pruned weights back to regular weights
                    module_name = key.replace('.weight_orig', '')
                    mask_key = f"{module_name}.weight_mask"
                    if mask_key in state_dict:
                        pruned_weight = value * state_dict[mask_key]
                        new_state_dict[module_name + '.weight'] = pruned_weight
                elif 'weight_mask' not in key and 'bias_orig' not in key and 'bias_mask' not in key:
                    new_state_dict[key] = value
            state_dict = new_state_dict
        
        # Load the state dict into the model
        model.load_state_dict(state_dict, strict=False)
        # For pre-trained weights, always start from epoch 0
        return model, 0
    else:
        return model, 0


def train():
    # Setup experiment tracking
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"runs/experiment_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    
    # Save config
    config_dict = {
        'batch_size': TRAIN_DATALOADER_CFG.batch_size,
        'pruning_enabled': PRUNING_CFG.enabled,
        'target_sparsity': getattr(PRUNING_CFG, 'target_sparsity', 0),
        'epochs': TRAIN_CFG.num_epoch
    }
    with open(f"{log_dir}/config.json", 'w') as f:
        json.dump(config_dict, f, indent=2)

    # Setup data loaders
    train_anno_path = os.path.join(DATA_CFG.root_path, DATA_CFG.train_anno_path)
    with open(train_anno_path, 'r') as rf:
        train_anno = json.load(rf)

    transforms_lst = []
    for config in TRAIN_PIPE:
        transforms_lst.append(get_transforms(config))
    transforms_lst = Compose(transforms_lst)
    
    train_dataset = ImgClsDataset(DATA_CFG, 
                                 train_anno,
                                 transforms=transforms_lst)

    train_dataloader = DataLoader(dataset=train_dataset,
                                 batch_size=TRAIN_DATALOADER_CFG.batch_size,
                                 shuffle=TRAIN_DATALOADER_CFG.shuffle,
                                 num_workers=TRAIN_DATALOADER_CFG.num_worker,
                                 collate_fn=base_collate_fn)

    valid_anno_path = os.path.join(
        DATA_CFG.root_path,
        DATA_CFG.valid_anno_path
    )

    with open(valid_anno_path, 'r') as rf:
        valid_anno = json.load(rf)

    transforms_lst = []
    for config in VALID_PIPE:
        transforms_lst.append(get_transforms(config))
    transforms_lst = Compose(transforms_lst)

    valid_dataset = ImgClsDataset(
        DATA_CFG, 
        valid_anno,
        transforms=transforms_lst
    )

    valid_dataloader = DataLoader(
        dataset=valid_dataset,
        batch_size=VALID_DATALOADER_CFG.batch_size,
        shuffle=VALID_DATALOADER_CFG.shuffle,
        num_workers=VALID_DATALOADER_CFG.num_worker,
        collate_fn=base_collate_fn
    )

    model = get_model(MODEL_TYPE, MODEL_CFG)
    model = model.to(DEVICE)
    
    # Initialize pruning if enabled
    pruning_hook = None
    if PRUNING_CFG.enabled and TRAIN_CFG.num_epoch > 0:
        print("\nInitializing pruning...")
        pruner, pruning_hook = apply_pruning_schedule(
            model=model,
            config=PRUNING_CFG,
            epochs=TRAIN_CFG.num_epoch,
            initial_sparsity=PRUNING_CFG.initial_sparsity,
            target_sparsity=PRUNING_CFG.target_sparsity,
            prune_epochs=PRUNING_CFG.prune_epochs or None
        )
    
    loss_fn = nn.MSELoss(reduction='mean') # DiceFocalLoss()
    optim = Adam(model.parameters(), lr=1e-5)
    # optim = SGD(model.parameters(), lr=0.02, weight_decay=1e-4, momentum=0.9) # SGDW
    # lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=optim, total_iters=500, start_factor=0.02)
    num_batch = len(train_dataloader)
    digits = len(str(num_batch))

    total_epoch = TRAIN_CFG.num_epoch
    start_epoch = 0
    if MODEL_CFG.load_from is not None:
        model, start_epoch = model_load(model)

    remaining_iters = len(train_dataloader) * (TRAIN_CFG.num_epoch - start_epoch + 1)

    # Training loop
    global_step = 0
    best_iou = 0.0
    log_losses = []  # Initialize log_losses list
    
    for epoch in range(start_epoch, total_epoch+1):
        # Apply pruning if enabled and it's time to prune
        if pruning_hook is not None:
            pruning_hook(epoch)
        
        # Log sparsity if pruning is enabled
        if PRUNING_CFG.enabled and epoch > 0:
            sparsity_info = analyze_pruning(model)
            writer.add_scalar('Sparsity/Overall', sparsity_info['sparsity'], epoch)
            print(f"Epoch {epoch}: Model sparsity = {sparsity_info['sparsity']:.2f}%")
            
        batch_cnt = 0
        epoch_loss = 0.0
        model = model.train()
        
        # Progress bar
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{total_epoch}")
        
        for images, masks, sensors, annotations in pbar:
            global_step += 1
            batch_cnt += 1
            start_time = time.time()
            
            # Move data to device
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            sensors = sensors.to(DEVICE) if sensors is not None else None
            
            # Forward pass
            optim.zero_grad()
            outs = model(images=images, sensors=sensors)['out'].squeeze()
            
            # Calculate loss
            loss = loss_fn(outs, masks)
            
            # Backward pass and optimize
            loss.backward()
            optim.step()
            
            # Update metrics
            epoch_loss += loss.item()
            
            # Log batch metrics
            if batch_cnt % TRAIN_CFG.log_step == 0:
                # Calculate metrics
                with torch.no_grad():
                    metrics = calculate_metrics(outs.detach(), masks.detach() > 0.5)
                
                # Log to TensorBoard
                writer.add_scalar('Loss/train', loss.item(), global_step)
                for metric_name, metric_value in metrics.items():
                    writer.add_scalar(f'Metrics/train_{metric_name}', metric_value, global_step)
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'iou': f"{metrics['iou']:.4f}"
                })

            iter_time = time.time() - start_time
            if not log_losses:  # If log_losses is empty, append a new dict
                log_losses.append({'time': iter_time})
            else:
                log_losses[-1].update(time=iter_time)

            if batch_cnt % TRAIN_CFG.log_step == 0 or batch_cnt == len(train_dataloader):
                now = datetime.datetime.now().strftime("%y/%m/%d %H:%M:%S")
                log_lsg = f"[{now}] Epoch(Train) [{epoch}/{total_epoch}][{batch_cnt: {digits}}/{num_batch}] "
                log_msg, remaining_iters = get_log_msg(log_lsg, log_losses, remaining_iters)
                log_losses = list()
                print(log_msg, flush=True)

        # Validation
        if (epoch + 1) % 1 == 0:
            model.eval()
            val_loss = 0.0
            val_metrics = {'iou': 0.0, 'f1': 0.0, 'precision': 0.0, 'recall': 0.0}
            val_batches = 0
            
            with torch.no_grad():
                for images, masks, sensors, _ in valid_dataloader:
                    images = images.to(DEVICE)
                    masks = masks.to(DEVICE)
                    sensors = sensors.to(DEVICE) if sensors is not None else None
                    
                    outs = model(images=images, sensors=sensors)['out'].squeeze()
                    batch_loss = loss_fn(outs, masks)
                    
                    val_loss += batch_loss.item()
                    batch_metrics = calculate_metrics(outs, masks > 0.5)
                    
                    # Accumulate metrics
                    for k in val_metrics:
                        val_metrics[k] += batch_metrics[k]
                    val_batches += 1
            
            # Average metrics
            val_loss /= val_batches
            for k in val_metrics:
                val_metrics[k] /= val_batches
            
            # Log validation metrics
            print(f"\nValidation - Loss: {val_loss:.4f}, "
                  f"IoU: {val_metrics['iou']:.4f}, "
                  f"F1: {val_metrics['f1']:.4f}")
            
            # Log to TensorBoard
            writer.add_scalar('Loss/val', val_loss, epoch)
            for metric_name, metric_value in val_metrics.items():
                writer.add_scalar(f'Metrics/val_{metric_name}', metric_value, epoch)
            
            # Save best model
            if val_metrics['iou'] > best_iou:
                best_iou = val_metrics['iou']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optim.state_dict(),
                    'val_iou': best_iou,
                    'val_loss': val_loss,
                }, f"{log_dir}/best_model.pth")
                print(f"Saved new best model with IoU: {best_iou:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:  # Save every 5 epochs
            checkpoint_path = f"{log_dir}/checkpoint_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'val_iou': best_iou,
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    
    # Save final model
    final_model_path = f"{log_dir}/final_model.pth"
    torch.save({
        'epoch': total_epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
        'val_iou': best_iou,
        'val_loss': val_loss if 'val_loss' in locals() else None,
        'config': config_dict
    }, final_model_path)
    
    print(f"\nTraining complete! Best validation IoU: {best_iou:.4f}")
    print(f"Run 'tensorboard --logdir={log_dir}' to view metrics.")
    
    return best_iou


def valid(model_path=None):
    """Run validation on the validation set.
    
    Args:
        model_path: Path to the model checkpoint. If None, uses the model from MODEL_CFG.
    """
    # Setup data loader
    valid_anno_path = os.path.join(DATA_CFG.root_path, DATA_CFG.valid_anno_path)
    
    with open(valid_anno_path, 'r') as rf:
        valid_anno = json.load(rf)

    transforms_lst = Compose([get_transforms(config) for config in VALID_PIPE])
    valid_dataset = ImgClsDataset(DATA_CFG, valid_anno, transforms=transforms_lst)
    
    valid_dataloader = DataLoader(
        dataset=valid_dataset,
        batch_size=VALID_DATALOADER_CFG.batch_size,
        shuffle=False,  # No need to shuffle for validation
        num_workers=VALID_DATALOADER_CFG.num_worker,
        collate_fn=base_collate_fn
    )

    # Initialize model
    model = get_model(MODEL_TYPE, MODEL_CFG).to(DEVICE)
    
    # Load weights if provided
    if model_path and os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=DEVICE)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded model from {model_path}")
    elif MODEL_CFG.load_from:
        model, _ = model_load(model)
    
    # Validate
    model.eval()
    val_loss = 0.0
    val_metrics = {'iou': 0.0, 'f1': 0.0, 'precision': 0.0, 'recall': 0.0}
    val_batches = 0
    
    with torch.no_grad():
        for images, masks, sensors, _ in tqdm(valid_dataloader, desc="Validating"):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            sensors = sensors.to(DEVICE) if sensors is not None else None
            
            # Forward pass
            outs = model(images=images, sensors=sensors)['out'].squeeze()
            
            # Calculate loss and metrics
            loss = nn.MSELoss()(outs, masks)
            batch_metrics = calculate_metrics(outs, masks > 0.5)
            
            # Accumulate
            val_loss += loss.item()
            for k in val_metrics:
                val_metrics[k] += batch_metrics[k]
            val_batches += 1
    
    # Calculate averages
    val_loss /= val_batches
    for k in val_metrics:
        val_metrics[k] /= val_batches
    
    # Print results
    print("\nValidation Results:")
    print(f"Loss: {val_loss:.4f}")
    for name, value in val_metrics.items():
        print(f"{name.upper()}: {value:.4f}")
    
    return {'loss': val_loss, **val_metrics}


def test():
    model = get_model(MODEL_TYPE, MODEL_CFG)
    if MODEL_CFG.load_from is not None:
        model, _ = model_load(model)
    # model = model.train()
    model = model.to(DEVICE)

    if TEST_MODE:
        print("Running in TEST MODE using PASCAL VOC dataset")
        # Use PASCAL VOC dataset for testing
        transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
        ])
        
        # Download PASCAL VOC dataset
        train_dataset = dset.VOCSegmentation(
            root='./data',
            year='2012',
            image_set='train',
            download=True,
            transform=transform,
            target_transform=transform  # For simplicity, using same transform for mask
        )
        
        valid_dataset = dset.VOCSegmentation(
            root='./data',
            year='2012',
            image_set='val',
            download=True,
            transform=transform,
            target_transform=transform
        )
        
        num_classes = 21  # PASCAL VOC has 20 classes + background
    else:
        # Original dataset loading code
        test_anno_path = os.path.join(DATA_CFG.root_path,
                                  DATA_CFG.test_anno_path)

        with open(test_anno_path, 'r') as rf:
            test_anno = json.load(rf)

        transforms_lst = []
        for config in TEST_PIPE:
            transforms_lst.append(get_transforms(config))
        transforms_lst = Compose(transforms_lst)

        test_dataset = ImgClsDataset(DATA_CFG, 
                                test_anno,
                                transforms=transforms_lst
                                )

        num_classes = 5  # Your original 5 classes

    # Create dataloaders
    batch_size = 4 if TEST_MODE else TEST_DATALOADER_CFG.batch_size
    num_workers = 2 if TEST_MODE else TEST_DATALOADER_CFG.num_worker
    
    if TEST_MODE:
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=None
        )
        
        valid_dataloader = DataLoader(
            dataset=valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=None
        )
    else:
        test_dataloader = DataLoader(dataset=test_dataset,
                                  batch_size=batch_size,
                                  shuffle=TEST_DATALOADER_CFG.shuffle,
                                  num_workers=num_workers,
                                  collate_fn=base_collate_fn)

    model = model.eval()

    if TEST_MODE:
        for images, masks, _ in train_dataloader:
            outs = model(images)['out'].squeeze()
    else:
        for images, masks, annotations in test_dataloader:
            outs = model(images)['out'].squeeze()

            num_batch = len(outs)
            for ix in range(num_batch):
                image = images[ix].detach().cpu().numpy()
                image = image.transpose(1, 2, 0)
                out = outs[ix].round().detach().cpu().numpy().astype(np.uint8)
                mask = masks[ix].detach().cpu().numpy().astype(np.uint8)
                fig, arr = plt.subplots(1, 3)
                arr[0].imshow(image)
                arr[0].set_title("image")
                arr[1].imshow(out)
                arr[1].set_title("Pred_Mask")
                arr[2].imshow(mask)
                arr[2].set_title("True_Mask")
                plt.savefig("test.png", dpi=300)
                plt.close()
        num_batch = len(outs)
        for ix in range(num_batch):
            image = images[ix].detach().cpu().numpy()
            image = image.transpose(1, 2, 0)
            out = outs[ix].round().detach().cpu().numpy().astype(np.uint8)
            mask = masks[ix].detach().cpu().numpy().astype(np.uint8)
            fig, arr = plt.subplots(1, 3)
            arr[0].imshow(image)
            arr[0].set_title("image")
            arr[1].imshow(out)
            arr[1].set_title("Pred_Mask")
            arr[2].imshow(mask)
            arr[2].set_title("True_Mask")
            plt.savefig("test.png", dpi=300)
            plt.close()

if __name__ == "__main__":
    train() # data_test, test_one_img
