import torch.nn as nn
import numpy as np
from abc import abstractclassmethod


class BaseModel(nn.Module):
    """
    Base class for all models
    """
    @abstractclassmethod
    def forward(self, *inputs):
        """
        forward pass logic

        Returns:
            model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + f'\nTrainable parameters: {params}'
