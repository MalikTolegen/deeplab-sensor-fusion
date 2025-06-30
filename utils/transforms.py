from typing import (
    List,
    Tuple,
    Optional
)

import cv2
import numpy as np
from torchvision import transforms as T


def get_transforms(config: List[dict]):
    if hasattr(T, config['type']):
        return getattr(T, config['type'])(**config['params'])
    else:
        raise ValueError(f"{config['type']} is not supported.")


def get_mask_from_polygon(img_size: tuple, num_classes: int, annotations):
    scale_cnt = len(img_size)
    assert scale_cnt == 1 or scale_cnt == 2, ValueError(f"Resize ptr error. resize_scale's shape is only 1 or 2. \
                                            {scale_cnt} is not support.")
    mask = np.zeros((img_size[0], img_size[-1]), dtype=np.uint8)
    for anno in annotations:
        if 'segmentation' in anno:
            cat_id = anno['category_id']
            for seg in anno['segmentation']:
                polygon = np.array(seg, dtype=np.int32).reshape((-1, 2))
                cv2.fillPoly(mask, [polygon], cat_id)

    return mask


def resize_seg_ptrs(
    annotations: List[dict],
    original_size: Tuple[int, int], # h, w
    resize_scale: Optional[Tuple[int, int] | Tuple[int]] # h, w
    ) -> Tuple[float, float]:
    scale_cnt = len(resize_scale)
    assert scale_cnt == 1 or scale_cnt == 2, ValueError(f"Resize ptr error. resize_scale's shape is only 1 or 2. \
                                              {scale_cnt} is not support.")

    y_ratio = resize_scale[0] / original_size[0]
    x_ratio = resize_scale[-1] / original_size[1]
    num_annotation = len(annotations)
    for ix in range(num_annotation):
        anno = annotations[ix]
        if anno['is_scale'] is False and 'segmentation' in anno:
            for seg in anno['segmentation']:
                for seg_ix in range(0, len(seg), 2):
                    seg[seg_ix] = seg[seg_ix] * x_ratio
                    seg[seg_ix+1] = seg[seg_ix+1] * y_ratio
            anno['is_rescale'] = True

