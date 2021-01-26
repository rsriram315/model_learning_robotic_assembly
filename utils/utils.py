import os
import json
import torch
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
# from types import SimpleNamespace


def ensure_dir(fname):
    # fname = Path(fname)
    # if not fname.is_dir():
    #     fname.mkdir(parents=True, exist_ok=False)
    os.makedirs(os.path.dirname(fname), exist_ok=True)


def read_json(fname):
    """
    Get params from json file
    Args:
        json_file: path of the json file
    Return:
        params: parameters from the json file (dict)
    """

    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

    # with open(json_file, 'r') as f:
    #     try:
    #         params = json.load(f, object_hook=lambda d: SimpleNamespace(**d))
    #     except ValueError as err:
    #         print(f"... Invalid json: {err}")
    #         return -1
    # return params


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(dataloader):
    """
    wrapper function for endless data loader.
    """
    for loader in repeat(dataloader):
        yield from loader


def prepare_device(n_gpu_use):
    """
    setup GPU device if available.
    get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU")
        n_gpu_use = n_gpu
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use "
              f"is {n_gpu_use}, but only {n_gpu} are available on"
              f"this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys,
                                  columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = (self._data.total[key] /
                                   self._data.counts[key])

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)
