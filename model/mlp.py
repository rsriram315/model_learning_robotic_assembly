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
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)

        # 6D rotation representation
        x = compute_rotation_matrix_from_ortho6d(x)

        # if use cosine and sine, use tanh or clamp
        # x[:, 6:] = torch.tanh(x[:, 6:])
        # x[:, 6:] = torch.clamp(x[:, 6:].clone(), min=-1, max=1)
        return x  # should I use softmax for the output


class MCDropout(BaseModel):
    def __init__(self, input_dims, output_dims, dropout_prob=0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_dims, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, output_dims)
        self.dropout = nn.Dropout(p=dropout_prob)
        # self.dropout_prob = dropout_prob
        # self.is_dropout = not self.training

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # dropout in the second last layer
        # x = F.dropout(F.relu(self.fc2(x)),
        #               p=self.dropout_prob,
        #               training=self.is_dropout)
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x  # should I use softmax for the output
