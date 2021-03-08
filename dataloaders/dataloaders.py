from base import BaseDataLoader


class DemoDataLoader(BaseDataLoader):
    """
    data loading demo using BaseDataLoader
    """
    def __init__(self, dataset, dl_cfg):
        self.demo_fnames = dataset.get_fnames()
        super().__init__(dataset, dl_cfg)

    def get_fnames(self):
        return self.demo_fnames
