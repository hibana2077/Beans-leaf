import torch
import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ReduceLROnPlateau,
    StepLR,
    OneCycleLR
)
from .opts.taylorexp import TaylorExp

def get_optimizer(model, optimizer_name='adam', lr=0.001, weight_decay=0.0001, **kwargs):
    """
    Get the specified optimizer
    
    Args:
        model: PyTorch model
        optimizer_name (str): Name of the optimizer ('adam', 'sgd', 'adamw')
        lr (float): Learning rate
        weight_decay (float): Weight decay (L2 penalty)
        **kwargs: Additional arguments for the optimizer
        
    Returns:
        torch.optim.Optimizer: Optimizer
    """
    # Get all parameters that require gradients
    params = [p for p in model.parameters() if p.requires_grad]
    
    if optimizer_name.lower() == 'adam':
        return optim.Adam(params, lr=lr, weight_decay=weight_decay, **kwargs)
    elif optimizer_name.lower() == 'sgd':
        return optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay, **kwargs)
    elif optimizer_name.lower() == 'adamw':
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay, **kwargs)
    elif optimizer_name.lower() == 'taylorexp':
        return TaylorExp(params, lr=lr, weight_decay=weight_decay, **kwargs)
    else:
        raise ValueError(f"Optimizer '{optimizer_name}' not recognized")


def get_scheduler(optimizer, scheduler_name='cosine', num_epochs=100, **kwargs):
    """
    Get the specified learning rate scheduler
    
    Args:
        optimizer: PyTorch optimizer
        scheduler_name (str): Name of the scheduler ('cosine', 'plateau', 'step', 'onecycle')
        num_epochs (int): Total number of epochs
        **kwargs: Additional arguments for the scheduler
        
    Returns:
        torch.optim.lr_scheduler: Learning rate scheduler
    """
    if scheduler_name.lower() == 'cosine':
        return CosineAnnealingLR(optimizer, T_max=num_epochs, **kwargs)
    elif scheduler_name.lower() == 'plateau':
        return ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, **kwargs)
    elif scheduler_name.lower() == 'step':
        return StepLR(optimizer, step_size=30, gamma=0.1, **kwargs)
    elif scheduler_name.lower() == 'onecycle':
        return OneCycleLR(optimizer, max_lr=kwargs.get('max_lr', 0.01), total_steps=num_epochs, **kwargs)
    else:
        raise ValueError(f"Scheduler '{scheduler_name}' not recognized")