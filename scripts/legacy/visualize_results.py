"""
Visualization helpers for training curves and cross-model comparisons.
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict


def plot_training_history(history_path: str, save_path: str = None):
    """
    Plot standard training/validation curves.

    Args:
        history_path: Path to a JSON file containing history metrics.
        save_path: Optional path to save the resulting figure.
    """
    # Load training history
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    # Create figure canvas
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training history', fontsize=16, fontweight='bold')
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', marker='o')
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', marker='s')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss curves')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(epochs, history['train_accuracy'], 'b-', label='Train Acc', marker='o')
    axes[0, 1].plot(epochs, history['val_accuracy'], 'r-', label='Val Acc', marker='s')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Accuracy curves')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # AUROC
    axes[1, 0].plot(epochs, history['val_auroc'], 'g-', label='Val AUROC', marker='^')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('AUROC')
    axes[1, 0].set_title('AUROC curve')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # F1
    axes[1, 1].plot(epochs, history['val_f1'], 'm-', label='Val F1', marker='d')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('F1 Score')
    axes[1, 1].set_title('F1 Score curve')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure saved to: {save_path}")
    
    plt.show()


def compare_models(model_results: Dict[str, str], save_path: str = None):
    """
    Compare multiple models side-by-side.

    Args:
        model_results: Mapping from model name to prediction CSV path.
        save_path: Optional figure output path.
    """
    from sklearn.metrics import roc_curve, auc, confusion_matrix
    import seaborn as sns
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle('Model performance comparison', fontsize=16, fontweight='bold')
    
    # 1. ROC curves
    for model_name, csv_path in model_results.items():
        df = pd.read_csv(csv_path)
        fpr, tpr, _ = roc_curve(df['label'], df['probability'])
        roc_auc = auc(fpr, tpr)
        
        axes[0].plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})', linewidth=2)
    
    axes[0].plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC curves')
    axes[0].legend(loc='lower right')
    axes[0].grid(True, alpha=0.3)
    
    # 2. Metric comparison (bar chart)
    from utils.metrics import compute_metrics
    
    metrics_data = []
    for model_name, csv_path in model_results.items():
        df = pd.read_csv(csv_path)
        metrics = compute_metrics(
            predictions=df['prediction'].values,
            labels=df['label'].values,
            probabilities=df['probability'].values
        )
        metrics['model'] = model_name
        metrics_data.append(metrics)
    
    metrics_df = pd.DataFrame(metrics_data)
    
    # Metrics to visualize
    metrics_to_plot = ['accuracy', 'auroc', 'f1', 'precision', 'recall']
    x = range(len(metrics_to_plot))
    width = 0.25
    
    for i, model_name in enumerate(model_results.keys()):
        model_data = metrics_df[metrics_df['model'] == model_name]
        values = [model_data[m].values[0] for m in metrics_to_plot]
        axes[1].bar([xi + i*width for xi in x], values, width, label=model_name)
    
    axes[1].set_xlabel('Metric')
    axes[1].set_ylabel('Score')
    axes[1].set_title('Metric comparison')
    axes[1].set_xticks([xi + width for xi in x])
    axes[1].set_xticklabels([m.upper() for m in metrics_to_plot], rotation=45)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Comparison figure saved to: {save_path}")
    
    plt.show()


def main():
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Visualize training/eval results")
    parser.add_argument(
        "--mode",
        type=str,
        choices=['history', 'compare'],
        default='history',
        help="Visualization mode"
    )
    parser.add_argument(
        "--history_path",
        type=str,
        default="./checkpoints/vilt/training_history.json",
        help="Training history JSON path"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Optional figure output path"
    )
    
    args = parser.parse_args()
    
    if args.mode == 'history':
        print("Plotting training history...")
        plot_training_history(args.history_path, args.save_path)
    
    elif args.mode == 'compare':
        print("Comparing model performance...")
        
        # Example: compare three models
        model_results = {
            'ViLT': './results/vilt_predictions.csv',
            'BERT': './results/bert_predictions.csv',
            'ViT': './results/vit_predictions.csv',
        }
        
        # Ensure the referenced CSV files exist
        model_results = {k: v for k, v in model_results.items() if Path(v).exists()}
        
        if not model_results:
            print("❌ No prediction files found; run evaluation first.")
            return
        
        compare_models(model_results, args.save_path)


if __name__ == "__main__":
    main()

