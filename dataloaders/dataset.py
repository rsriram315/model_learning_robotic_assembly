import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from utils import read_one_demo, pair_state_action


class DemoDataset(Dataset):
    def __init__(self, root, fnames, sample_freq,
                 state_attrs, action_attrs, transform=None):
        """
        form (state, action) pair as x, and state at right next time stamp
        as label, pack them together.

        Args:
            params: dataset dict in the config.json
        """
        super().__init__()
        self.root = Path(root)
        self.data_paths = [self.root / fn for fn in fnames]

        self.sample_freq = sample_freq
        self.state_attrs = state_attrs
        self.action_attrs = action_attrs

        self.states = []
        self.actions = []
        self.state_action_idx = []
        self._read_all_demos()

        self.transform = transform

    def __len__(self):
        return len(self.state_action_idx)

    def __getitem__(self, idx):
        state_idx, action_idx = self.state_action_idx[idx]
        sample = np.concatenate((self.states[state_idx],
                                 self.actions[action_idx]))
        # simpliest case where action is the target state
        target = self.actions[action_idx]

        if self.transform:
            sample = self.transform(sample)
        return np.float32(sample), np.float32(target)

    def _read_all_demos(self):
        offset = [0, 0]  # state and action offset

        for data_path in self.data_paths:
            states_time, actions_time = read_one_demo(data_path,
                                                      self.state_attrs,
                                                      self.action_attrs)

            self.states.extend(states_time["data"])
            self.actions.extend(actions_time["data"])

            state_action_idx = pair_state_action(self.sample_freq,
                                                 states_time, actions_time)
            for i in range(len(state_action_idx)):
                state_action_idx[i] = \
                    tuple(map(lambda x, y: x + y, state_action_idx[i], offset))
            self.state_action_idx.extend(state_action_idx)

            offset[0] += len(states_time["data"])
            offset[1] += len(actions_time["data"])
