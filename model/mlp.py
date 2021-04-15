import torch.nn as nn
import torch  # noqa
import torch.nn.functional as F
from .base_model import BaseModel
from dataloaders import compute_rotation_matrix_from_ortho6d  # noqa


class MLP(BaseModel):
    def __init__(self, input_dims, output_dims, ds_stats):
        super().__init__()
        self.fc1 = nn.Linear(input_dims, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, output_dims)

        self.rot_stat_1 = (torch.tensor(ds_stats["stat_3"],
                                        dtype=torch.float32,
                                        requires_grad=False).cuda())
        self.rot_stat_2 = torch.tensor(ds_stats["stat_4"],
                                       dtype=torch.float32,
                                       requires_grad=False).cuda()
        # leave the non-rotation unchanged
        self.rot_stat_1[:, :6] = torch.zeros(6) - 1
        self.rot_stat_2[:, :6] = torch.ones(6) + 1

    def forward(self, x):
        input_state = torch.zeros_like(x[:, :15], requires_grad=False).cuda()
        input_state[:, 6:] = x.detach()[:, 6:15]

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # 6D rotation representation
        x = compute_rotation_matrix_from_ortho6d(x)
        x = x - input_state
        x = (x - self.rot_stat_1) / self.rot_stat_2
        x = (x - 0.5) * 2
        return x


class MCDropout(BaseModel):
    def __init__(self, input_dims, output_dims, ds_stats, dropout_prob=0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_dims, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, output_dims)
        self.dropout = nn.Dropout(p=dropout_prob)

        self.rot_stat_1 = (torch.tensor(ds_stats["stat_3"],
                                        dtype=torch.float32,
                                        requires_grad=False).cuda())
        self.rot_stat_2 = torch.tensor(ds_stats["stat_4"],
                                       dtype=torch.float32,
                                       requires_grad=False).cuda()
        # leave the rotation unchanged
        self.rot_stat_1[:, :6] = torch.zeros(6) - 1
        self.rot_stat_2[:, :6] = torch.ones(6) + 1

    def forward(self, x):
        input_state = torch.zeros_like(x[:, :15], requires_grad=False).cuda()
        input_state[:, 6:] = x.detach()[:, 6:15]

        x = F.relu(self.fc1(x))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)

        x = compute_rotation_matrix_from_ortho6d(x)
        x = x - input_state
        x = (x - self.rot_stat_1) / self.rot_stat_2
        x = (x - 0.5) * 2
        return x
