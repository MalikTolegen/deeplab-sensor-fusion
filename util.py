import torch

from config import DEVICE

def base_collate_fn(batch):
    img_lst = list()
    masks = list()
    annotations = list()
    sensors = list()
    for sample in batch:
        img_lst.append(sample[0])
        masks.append(sample[1])
        sensors.append(torch.Tensor([
            sample[2]["humi"],
            sample[2]["pressure"],
            sample[2]["objectTemp"],
            sample[2]["latitude"],
            sample[2]["longitude"],
            sample[2]["height"],
        ]))
        annotations.append(sample[3])

    return torch.stack(img_lst).to(DEVICE), torch.stack(masks).to(DEVICE), torch.stack(sensors).to(DEVICE), annotations
