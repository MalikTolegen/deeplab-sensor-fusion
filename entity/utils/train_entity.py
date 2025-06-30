from ..base_entity import BaseEntity


class TrainEntity(BaseEntity):
    num_epoch: int = 10
    accum_step: int = 1
    log_step: int = 1