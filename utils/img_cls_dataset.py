from typing import (
    Optional,
    Any
)
import os
import torch
import cv2
from torchvision.transforms import Resize
from albumentations import Compose
from torch.utils.data import Dataset
from entity import DataEntity
from .transforms import (
    get_mask_from_polygon,
)


class ImgClsDataset(Dataset):
    def __init__(self, config: DataEntity, raw_data: dict, transforms: Optional[Compose]=None):
        self.config = config
        super(ImgClsDataset, self).__init__()
        self.images: list = raw_data['images']
        self.annotations: list = raw_data['annotations']
        self.categories: list = raw_data['categories']
        self.transforms: Optional[Compose] = transforms
        self.ia_map: dict = self.__init_ia_map()

        self.resize_scale = False
        if transforms is not None:
            for t in transforms.transforms:
                if type(t) == Resize:
                    self.resize_scale = t.size
                    break

    def __init_ia_map(self):
        mapper = {}

        num_anno = len(self.annotations)
        for anno_ix in range(num_anno):
            anno_info = self.annotations[anno_ix]
            anno_info.update(is_scale=False)
            if anno_info['image_id'] not in mapper:
                mapper[anno_info['image_id']] = []

            mapper[anno_info['image_id']].append(anno_ix)

        rm_objs = list()
        for img_info in self.images:
            img_id = img_info["id"]
            if img_id not in mapper:
                rm_objs.append(img_info)

        for rm_obj in rm_objs:
            self.images.remove(rm_obj)

        return mapper

    def __get_idices_elems(self, indices) -> list:
        return [self.annotations[i] for i in indices]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index) -> Any:
        image_info = self.images[index]
        image_id = image_info["id"]
        image_fn = image_info["file_name"]
        image_path = os.path.join(self.config.root_path, self.config.image_base_path, image_fn)
        image = cv2.imread(image_path)

        annotations = self.__get_idices_elems(self.ia_map[image_id])
        mask = get_mask_from_polygon(image.shape[: 2], len(self.categories), annotations)
        if self.resize_scale is not None:
            mask = cv2.resize(mask, self.resize_scale)
        mask = torch.tensor(mask, dtype=torch.float32)
        sensors = image_info["sensor_info"]
        [anno.update({"image_path": image_path}) for anno in annotations]

        if self.transforms is not None:
            image = self.transforms(image)
        return image, mask, sensors, annotations
