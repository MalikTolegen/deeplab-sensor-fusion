import os
import json
import time
import datetime
from tqdm import tqdm

import torch
from torch import nn
from torch.optim import SGD, Adam
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
import cv2
import numpy as np
import matplotlib.pyplot as plt

from config.model_cfg import *
from models import get_model
from utils import ImgClsDataset
from utils import get_transforms
from models.loss import DiceFocalLoss
from models.metrics import meanIoU
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
    start_epoch, _ = os.path.splitext(os.path.basename(MODEL_CFG.load_from))
    chkp = torch.load(MODEL_CFG.load_from)
    model.load_state_dict(chkp, strict=False)
    
    print(f"{MODEL_CFG.load_from} is load.")
    return model, int(start_epoch)


def train():
    train_anno_path = os.path.join(DATA_CFG.root_path, DATA_CFG.train_anno_path)
    with open(train_anno_path, 'r') as rf:
        train_anno = json.load(rf)

    transforms_lst = []
    for config in TRAIN_PIPE:
        transforms_lst.append(get_transforms(config))
    transforms_lst = Compose(transforms_lst)
    
    train_dataset = ImgClsDataset(DATA_CFG, 
                                  train_anno,
                                  transforms=transforms_lst
                                  )

    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=TRAIN_DATALOADER_CFG.bathc_size,
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
        batch_size=VALID_DATALOADER_CFG.bathc_size,
        shuffle=VALID_DATALOADER_CFG.shuffle,
        num_workers=VALID_DATALOADER_CFG.num_worker,
        collate_fn=base_collate_fn
    )

    model = get_model(MODEL_TYPE, MODEL_CFG)
    model = model.to(DEVICE)
    
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

    for epoch in range(start_epoch, total_epoch+1):
        batch_cnt = 0
        log_losses = list()
        model = model.train()
        for images, masks, sensors, annotations in train_dataloader:
            batch_cnt += 1
            start_time = time.time()
            # images = images
            outs = model(model, images=images, sensors=sensors)['out'].squeeze()
            losses = dict(
                loss=loss_fn(outs, masks)
            )

            log_losses.append(get_loss_for_loss(losses))
            losses = sum(losses.values())
            losses.backward()
            if batch_cnt % TRAIN_CFG.accum_step == 0:
                optim.step()
                optim.zero_grad()
                # lr_scheduler.step()

            iter_time = time.time() - start_time
            log_losses[-1].update(time=iter_time)

            if batch_cnt % TRAIN_CFG.log_step == 0 or batch_cnt == len(train_dataloader):
                now = datetime.datetime.now().strftime("%y/%m/%d %H:%M:%S")
                log_lsg = f"[{now}] Epoch(Train) [{epoch}/{total_epoch}][{batch_cnt: {digits}}/{num_batch}] "
                log_msg, remaining_iters = get_log_msg(log_lsg, log_losses, remaining_iters)
                log_losses = list()
                print(log_msg, flush=True)

        torch.save(model.state_dict(), f"output/{epoch}.pth")

        model = model.eval()
        miou_lst = []
        with torch.no_grad():
            for images, masks, sensors, _ in valid_dataloader:
                outs = model(model, images, sensors)['out'].squeeze().detach().cpu()
                miou_lst.append(meanIoU(outs, masks.detach().cpu()))

        print(f"[{now}] Epoch(Valid) [{epoch}/{total_epoch}] mIoU: {sum(miou_lst) / len(miou_lst):.4f}")


def valid():
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
        batch_size=VALID_DATALOADER_CFG.bathc_size,
        shuffle=VALID_DATALOADER_CFG.shuffle,
        num_workers=VALID_DATALOADER_CFG.num_worker,
        collate_fn=base_collate_fn
    )

    model = get_model(MODEL_TYPE, MODEL_CFG)
    if MODEL_CFG.load_from is not None:
        model, _ = model_load(model)

    model = model.to(DEVICE)
    miou_lst = []
    for images, masks, sensors, _ in tqdm(valid_dataloader):
        outs = model(model, images, sensors)['out'].squeeze().detach().cpu()
        miou_lst.append(meanIoU(outs, masks.detach().cpu()))

    print(f"mIoU: {sum(miou_lst) / len(miou_lst):.4f}")


def test():
    model = get_model(MODEL_TYPE, MODEL_CFG)
    if MODEL_CFG.load_from is not None:
        model, _ = model_load(model)
    # model = model.train()
    model = model.to(DEVICE)

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

    test_dataloader = DataLoader(dataset=test_dataset,
                                  batch_size=TEST_DATALOADER_CFG.bathc_size,
                                  shuffle=TEST_DATALOADER_CFG.shuffle,
                                  num_workers=TEST_DATALOADER_CFG.num_worker,
                                  collate_fn=base_collate_fn)

    model = model.eval()

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

if __name__ == "__main__":
    train() # data_test, test_one_img