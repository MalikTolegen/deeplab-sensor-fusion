from ..base_entity import BaseEntity


class DataEntity(BaseEntity):
    root_path: str
    image_base_path: str
    train_anno_path: str
    valid_anno_path: str
    test_anno_path: str
