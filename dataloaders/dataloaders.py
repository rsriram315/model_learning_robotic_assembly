from .dataset import DemoDataset
from base import BaseDataLoader


class DemoDataLoader(BaseDataLoader):
    """
    data loading demo using BaseDataLoader
    """
    def __init__(self, dataset_cfg, dataloader_cfg):
        self.dataset = DemoDataset(**dataset_cfg["params"])
        super().__init__(self.dataset, **dataloader_cfg["params"])
