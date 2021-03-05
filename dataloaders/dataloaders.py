from base import BaseDataLoader


class DemoDataLoader(BaseDataLoader):
    """
    data loading demo using BaseDataLoader
    """
    def __init__(self, dataset, dl_cfg):
        super().__init__(dataset, dl_cfg)
