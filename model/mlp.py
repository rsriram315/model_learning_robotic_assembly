import torch.nn as nn
import torch  # noqa
import torch.nn.functional as F
from .base_model import BaseModel
from dataloaders import compute_rotation_matrix_from_ortho6d  # noqa


class MLP(BaseModel):
    def __init__(self, input_dims, output_dims):
        super().__init__()
        self.fc1 = nn.Linear(input_dims, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, output_dims)
        self.state_size = int(input_dims / 2)

    def forward(self, x):
        batch_size = x.shape[0]
        input = torch.zeros((batch_size, self.state_size),
                            requires_grad=False).cuda()
        input[:, 6:] = x.detach()[:, 6:15]

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # 6D rotation representation
        x = compute_rotation_matrix_from_ortho6d(x)
        x = x - input

        return x


class MCDropout(BaseModel):
    def __init__(self, input_dims, output_dims, dropout_prob=0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_dims, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, output_dims)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.state_size = int(input_dims / 2)

    def forward(self, x):
        batch_size = x.shape[0]
        input = torch.zeros((batch_size, self.state_size),
                            requires_grad=False).cuda()
        input[:, 6:] = x.detach()[:, 6:15]

        x = F.relu(self.fc1(x))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)

        x = compute_rotation_matrix_from_ortho6d(x)
        x = x - input

        return x
