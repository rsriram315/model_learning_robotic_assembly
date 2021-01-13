import torch.nn as nn


def mpc_loss(outputs, targets):
    loss = 0
    for output, target in enumerate(zip(outputs, targets)):
        loss += nn.MSELoss(output, target, reduction='sum')
    return loss
