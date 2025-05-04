import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        """
        Focal Loss for binary classification
        
        Args:
            alpha (float): Weighting factor for the rare class
            gamma (float): Focusing parameter that reduces the loss for well-classified examples
            reduction (str): Reduction method ('mean', 'sum', or 'none')
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets.float(), reduction='none')
        pt = torch.exp(-BCE_loss)  # prevents nans when probability 0
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss


def get_loss_function(loss_name='bce', **kwargs):
    """
    Get the specified loss function
    
    Args:
        loss_name (str): Name of the loss function ('bce', 'focal', 'ce')
        **kwargs: Additional arguments for the loss function
        
    Returns:
        torch.nn.Module: Loss function
    """
    if loss_name == 'bce':
        return nn.BCEWithLogitsLoss(**kwargs)
    elif loss_name == 'focal':
        return FocalLoss(**kwargs)
    elif loss_name == 'ce':
        return nn.CrossEntropyLoss(**kwargs)
    else:
        raise ValueError(f"Loss function '{loss_name}' not recognized")