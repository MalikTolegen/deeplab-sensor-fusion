from typing import Optional

from ..base_entity import BaseEntity


class BaseModelEntity(BaseEntity):
    params: dict = {}
    load_from: Optional[str] = None
    num_classes: Optional[int] = None