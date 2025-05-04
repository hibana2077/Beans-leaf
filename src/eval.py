import os
import argparse
import yaml
import torch
import numpy as np
from tqdm import tqdm
import timm
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from dataset import get_data_loaders


def evaluate(model, data_loader, device, threshold=0.5):
    """
    Evaluate the model on the given data loader
    
    Args:
        model: PyTorch model
        data_loader: PyTorch DataLoader
        device: Device to use for evaluation
        threshold: Classification threshold for binary classification
        
    Returns:
        dict: Dictionary with evaluation metrics
    """
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluation"):
            images = batch['image'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Handle binary classification
            if outputs.shape[1] == 1:
                outputs = outputs.squeeze(1)
                probs = torch.sigmoid(outputs)
                preds = (probs > threshold).long()
            else:
                probs, preds = torch.max(torch.softmax(outputs, dim=1), dim=1)
            
            # Collect results
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    accuracy = (all_preds == all_labels).mean() * 100
    
    # Classification report
    report = classification_report(all_labels, all_preds, output_dict=True)
    
    # Return results
    return {
        'labels': all_labels,
        'predictions': all_preds,
        'probabilities': all_probs,
        'accuracy': accuracy,
        'report': report
    }


def plot_confusion_matrix(labels, predictions, output_path):
    """
    Plot confusion matrix
    
    Args:
        labels: True labels
        predictions: Predicted labels
        output_path: Path to save the confusion matrix plot
    """
    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.savefig(output_path)
    plt.close()
    
    print(f"Confusion matrix saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained beans classifier model")
    parser.add_argument("--config", type=str, default="src/run.yml",
                        help="Path to config YAML file")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained model weights")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"],
                        help="Data split to evaluate on")
    parser.add_argument("--output_dir", type=str, default="outputs/evaluation",
                        help="Directory to save evaluation results")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data loaders
    data_loaders = get_data_loaders(
        batch_size=config.get('batch_size', 32),
        num_workers=config.get('num_workers', 4)
    )
    
    eval_loader = data_loaders[args.split]
    
    # Create model
    model_name = config.get('model_name', 'resnet18')
    num_classes = config.get('num_classes', 1)  # 1 for binary classification with sigmoid
    
    print(f"Creating model: {model_name}")
    
    model = timm.create_model(
        model_name,
        pretrained=False,  # We'll load weights from the trained model
        num_classes=num_classes
    )
    
    # Load trained weights
    print(f"Loading weights from {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    
    # Evaluate model
    results = evaluate(model, eval_loader, device)
    
    # Print results
    print(f"\nEvaluation on {args.split} split:")
    print(f"Accuracy: {results['accuracy']:.2f}%")
    
    # Print classification report
    print("\nClassification Report:")
    for cls, metrics in results['report'].items():
        if cls in ['0', '1']:  # Print metrics for each class
            print(f"Class {cls}:")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1-Score:  {metrics['f1-score']:.4f}")
            print(f"  Support:   {metrics['support']}")
    
    # Plot confusion matrix
    cm_path = os.path.join(args.output_dir, f"confusion_matrix_{args.split}.png")
    plot_confusion_matrix(results['labels'], results['predictions'], cm_path)
    
    # Save results to YAML
    results_to_save = {
        'accuracy': float(results['accuracy']),
        'classification_report': {
            k: {kk: vv for kk, vv in v.items() if isinstance(vv, (int, float))}
            for k, v in results['report'].items() if isinstance(v, dict)
        }
    }
    
    with open(os.path.join(args.output_dir, f"results_{args.split}.yml"), 'w') as f:
        yaml.dump(results_to_save, f)
    
    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()