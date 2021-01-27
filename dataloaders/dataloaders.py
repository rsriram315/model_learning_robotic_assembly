from base import BaseDataLoader


class DemoDataLoader(BaseDataLoader):
    """
    data loading demo using BaseDataLoader
    """
    def __init__(self, dataset, dataloader_cfg):
        super().__init__(dataset, **dataloader_cfg["params"])
