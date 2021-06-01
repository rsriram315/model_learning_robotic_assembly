import numpy as np

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


class BaseDataLoader(DataLoader):
    """
    base class for all data loaders
    """
    def __init__(self, dataset, dl_cfg):
        self.validation_split = dl_cfg["validation_split"]
        self.shuffle = dl_cfg["shuffle"]

        self.batch_idx = 0
        self.n_samples = len(dataset)
        # print(f"total samples {self.n_samples}")

        self.sampler, self.valid_sampler = \
            self._split_sampler(self.validation_split)

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': dl_cfg["batch_size"],
            'shuffle': self.shuffle,
            'num_workers': dl_cfg["num_workers"],
            'pin_memory': True
        }
        super().__init__(sampler=self.sampler, **self.init_kwargs)

    def _split_sampler(self, split):
        idx_full = np.arange(self.n_samples)

        np.random.seed(0)
        np.random.shuffle(idx_full)

        if isinstance(split, int):
            assert split > 0
            assert split < self.n_samples, ("validation size is configured to"
                                            " be larger than entire dataset.")
            len_val = split
        else:
            len_val = int(self.n_samples * split)

        valid_idx = idx_full[0:len_val]
        train_idx = np.delete(idx_full, np.arange(0, len_val))

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)
