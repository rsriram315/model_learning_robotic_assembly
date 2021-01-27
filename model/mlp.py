import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


class MLP(BaseModel):
    def __init__(self, input_dims, output_dims, dropout_prob=0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_dims, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, output_dims)
        # dropout and batch norm can add afterwards
        # self.dropout = nn.Dropout(dropout_prob)
        # self.batchnorm1 = nn.BatchNorm1d(500)
        # self.batchnorm2 = nn.BatchNorm1d(500)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # x = self.dropout(x, training=self.training)
        return x  # should I use softmax for the output
