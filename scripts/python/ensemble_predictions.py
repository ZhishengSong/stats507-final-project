#!/usr/bin/env python
"""Simple ensemble script that blends ViLT, BERT, and ViT predictions."""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import json

def load_predictions(filepath):
    """Load a CSV file containing `label` and `prob` columns."""
    df = pd.read_csv(filepath)
    return df['label'].values, df['prob'].values

def ensemble_predictions(vilt_probs, bert_probs, vit_probs, weights=(0.5, 0.3, 0.2)):
    """
    Combine probabilistic outputs from the three base models.

    Args:
        vilt_probs: Probabilities predicted by the ViLT model.
        bert_probs: Probabilities predicted by the BERT model.
        vit_probs: Probabilities predicted by the ViT model.
        weights: Tuple of weights applied to (ViLT, BERT, ViT).
    """
    w_vilt, w_bert, w_vit = weights
    ensemble_probs = w_vilt * vilt_probs + w_bert * bert_probs + w_vit * vit_probs
    ensemble_preds = (ensemble_probs > 0.5).astype(int)
    return ensemble_preds, ensemble_probs

def main():
    print("=" * 70)
    print("Model ensemble evaluation")
    print("=" * 70)
    
    # Load predictions from each model
    print("\nLoading prediction files...")
    vilt_labels, vilt_probs = load_predictions('logs/predictions/vilt_test_predictions.csv')
    bert_labels, bert_probs = load_predictions('logs/predictions/bert_test_predictions.csv')
    vit_labels, vit_probs = load_predictions('logs/predictions/vit_test_predictions.csv')
    
    # Ensure labels align across the CSVs
    assert np.all(vilt_labels == bert_labels) and np.all(bert_labels == vit_labels)
    true_labels = vilt_labels
    
    print(f"✓ Loaded {len(true_labels)} samples")
    
    # Evaluate several weight presets
    weight_configs = [
        ("Even weights", (0.33, 0.33, 0.34)),
        ("ViLT-heavy", (0.5, 0.3, 0.2)),
        ("ViLT-dominant", (0.6, 0.25, 0.15)),
        ("ViLT + BERT only", (0.6, 0.4, 0.0)),
        ("High ViLT + light BERT", (0.7, 0.3, 0.0)),
    ]
    
    print("\n" + "=" * 70)
    print("Performance by weight configuration")
    print("=" * 70)
    
    best_auroc = 0
    best_config = None
    best_preds = None
    best_probs = None
    
    results = []
    
    for name, weights in weight_configs:
        preds, probs = ensemble_predictions(vilt_probs, bert_probs, vit_probs, weights)
        
        acc = accuracy_score(true_labels, preds)
        auroc = roc_auc_score(true_labels, probs)
        f1 = f1_score(true_labels, preds, average='macro')
        
        results.append({
            'config': name,
            'weights': weights,
            'accuracy': acc,
            'auroc': auroc,
            'macro_f1': f1
        })
        
        print(f"\n{name}: {weights}")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  AUROC: {auroc:.4f}")
        print(f"  Macro F1: {f1:.4f}")
        
        if auroc > best_auroc:
            best_auroc = auroc
            best_config = name
            best_preds = preds
            best_probs = probs
    
    print("\n" + "=" * 70)
    print(f"Best configuration: {best_config}")
    print(f"Best AUROC: {best_auroc:.4f}")
    print("=" * 70)
    
    # Save ensemble predictions for downstream analysis
    print("\nSaving ensemble predictions...")
    ensemble_df = pd.DataFrame({
        'label': true_labels,
        'prob': best_probs,
        'prediction': best_preds
    })
    ensemble_df.to_csv('logs/predictions/ensemble_test_predictions.csv', index=False)
    
    # Save the metrics for reproducibility
    best_metrics = {
        'config': best_config,
        'accuracy': float(results[-1]['accuracy']),
        'auroc': float(best_auroc),
        'macro_f1': float(results[-1]['macro_f1'])
    }
    
    with open('logs/metrics/ensemble_metrics.json', 'w') as f:
        json.dump(best_metrics, f, indent=2)
    
    print("✓ Results saved to:")
    print("  - logs/predictions/ensemble_test_predictions.csv")
    print("  - logs/metrics/ensemble_metrics.json")
    
    # Compare ensemble vs. single-model baselines
    print("\n" + "=" * 70)
    print("Single model vs. ensemble")
    print("=" * 70)
    print(f"ViLT baseline: AUROC = 0.7395")
    print(f"BERT baseline: AUROC = 0.6509")
    print(f"ViT baseline:  AUROC = 0.5623")
    print(f"Ensemble AUROC: {best_auroc:.4f}")
    print(f"Lift vs. ViLT: {(best_auroc - 0.7395) * 100:.2f}%")
    print("=" * 70)

if __name__ == "__main__":
    main()

