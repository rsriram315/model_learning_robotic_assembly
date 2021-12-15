import torch.nn as nn
import torch  # noqa
import torch.nn.functional as F
from zmq.backend import device

from model.base_model import BaseModel
from model.helper import compute_rotation_matrix_from_ortho6d


class MLP(BaseModel):
    def __init__(self, input_dims, output_dims, device):
        super().__init__()
        self.fc1 = nn.Linear(input_dims, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, output_dims)
        self.device = device

    def forward(self, x):
        x = (self.fc1(x))
        x = (self.fc2(x))
        x = (self.fc3(x))

        # 6D rotation representation
        x = compute_rotation_matrix_from_ortho6d(x, device=self.device)
        return x


class MCDropout(BaseModel):
    def __init__(self, input_dims, output_dims, device, dropout_prob=0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_dims, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, output_dims)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.device = device

    def forward(self, x):
        input_state = torch.zeros_like(x[:, :15], requires_grad=False).to(self.device)
        input_state[:, 6:] = x.detach()[:, 6:15]

        x = F.relu(self.fc1(x))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)

        x = compute_rotation_matrix_from_ortho6d(x)
        return x
