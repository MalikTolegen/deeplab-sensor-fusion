from ..base_entity import BaseEntity


class DataloaderEntity(BaseEntity):
    batch_size: int = 16
    num_worker: int = 0
    shuffle: bool = True