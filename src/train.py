import os
import argparse
import yaml
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import timm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from utils.dataset import get_data_loaders
from utils.loss import get_loss_function
from utils.optimizer import get_optimizer, get_scheduler


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    
    for batch_idx, batch in enumerate(progress_bar):
        images = batch['image'].to(device)
        labels = batch['labels'].to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        
        # For binary classification, reshape outputs to match labels
        if outputs.shape[1] == 1:
            outputs = outputs.squeeze(1)
            loss = criterion(outputs, labels.float())
            preds = (torch.sigmoid(outputs) > 0.5).long()
        else:
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        total += labels.size(0)
        correct += (preds == labels).sum().item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': running_loss / (batch_idx + 1),
            'acc': 100. * correct / total
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(val_loader, desc="Validation")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
            images = batch['image'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(images)
            
            # For binary classification, reshape outputs to match labels
            if outputs.shape[1] == 1:
                outputs = outputs.squeeze(1)
                loss = criterion(outputs, labels.float())
                preds = (torch.sigmoid(outputs) > 0.5).long()
            else:
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
            
            # Statistics
            running_loss += loss.item()
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def main(config):
    # Set seeds for reproducibility
    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = config.get('experiment_name', 'beans_classification')
    output_dir = os.path.join('outputs', f"{experiment_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save configuration
    with open(os.path.join(output_dir, 'config.yml'), 'w') as f:
        yaml.dump(config, f)
    
    # Create tensorboard writer
    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'logs'))
    
    # Data loaders
    data_loaders = get_data_loaders(
        batch_size=config.get('batch_size', 32),
        num_workers=config.get('num_workers', 4)
    )
    
    train_loader = data_loaders['train']
    val_loader = data_loaders['val']
    
    # Create model
    model_name = config.get('model_name', 'resnet18')
    num_classes = config.get('num_classes', 1)  # 1 for binary classification with sigmoid
    
    print(f"Creating model: {model_name}")
    
    model = timm.create_model(
        model_name,
        pretrained=config.get('pretrained', True),
        num_classes=num_classes
    )
    model = model.to(device)
    
    # Loss function
    criterion = get_loss_function(
        loss_name=config.get('loss', 'bce'),
        **config.get('loss_args', {})
    )
    
    # Optimizer
    optimizer = get_optimizer(
        model,
        optimizer_name=config.get('optimizer', 'adam'),
        lr=config.get('learning_rate', 0.001),
        weight_decay=config.get('weight_decay', 0.0001),
        **config.get('optimizer_args', {})
    )
    
    # Learning rate scheduler
    scheduler = get_scheduler(
        optimizer,
        scheduler_name=config.get('scheduler', 'cosine'),
        num_epochs=config.get('epochs', 30),
        **config.get('scheduler_args', {})
    )
    
    # Training loop
    best_val_acc = 0.0
    num_epochs = config.get('epochs', 30)
    
    for epoch in range(1, num_epochs + 1):
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Log to tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        
        # Update learning rate
        if config.get('scheduler', 'cosine') == 'plateau':
            scheduler.step(val_loss)
        else:
            scheduler.step()
        
        # Print current LR
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('LR', current_lr, epoch)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
            print(f"Saved best model with validation accuracy: {val_acc:.2f}%")
        
        # Save checkpoint every few epochs
        if epoch % config.get('checkpoint_interval', 5) == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
            }, os.path.join(output_dir, f'checkpoint_epoch{epoch}.pth'))
        
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Learning rate: {current_lr:.6f}")
        print("-" * 50)
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(output_dir, 'final_model.pth'))
    
    # Close tensorboard writer
    writer.close()
    
    print(f"Training completed! Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Models and logs saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a beans classifier model")
    parser.add_argument("--config", type=str, default="src/run.yml",
                        help="Path to config YAML file")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    main(config)