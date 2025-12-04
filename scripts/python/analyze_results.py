#!/usr/bin/env python
"""Analyze model predictions to uncover improvement opportunities."""

import json
import pandas as pd
import numpy as np
from pathlib import Path

def load_predictions(filepath):
    """Load the prediction CSV produced by evaluation."""
    df = pd.read_csv(filepath)
    return df

def analyze_errors(predictions_df):
    """Return error-focused statistics."""
    predictions_df['correct'] = (predictions_df['prediction'] == predictions_df['label'])
    
    # Overall error rate
    error_rate = (~predictions_df['correct']).mean()
    
    # Split accuracy by class label
    hateful_errors = predictions_df[predictions_df['label'] == 1]
    non_hateful_errors = predictions_df[predictions_df['label'] == 0]
    
    hateful_acc = hateful_errors['correct'].mean()
    non_hateful_acc = non_hateful_errors['correct'].mean()
    
    # Count false positives and false negatives
    false_positives = predictions_df[(predictions_df['label'] == 0) & (predictions_df['prediction'] == 1)]
    false_negatives = predictions_df[(predictions_df['label'] == 1) & (predictions_df['prediction'] == 0)]
    
    return {
        'error_rate': error_rate,
        'hateful_accuracy': hateful_acc,
        'non_hateful_accuracy': non_hateful_acc,
        'false_positives': len(false_positives),
        'false_negatives': len(false_negatives),
        'fp_rate': len(false_positives) / len(non_hateful_errors),
        'fn_rate': len(false_negatives) / len(hateful_errors),
    }

def analyze_confidence(predictions_df):
    """Summarize confidence calibration metrics."""
    correct = predictions_df[predictions_df['correct']]
    incorrect = predictions_df[~predictions_df['correct']]
    
    return {
        'avg_prob_correct': correct['prob'].mean(),
        'avg_prob_incorrect': incorrect['prob'].mean(),
        'low_confidence_errors': (incorrect['prob'].between(0.4, 0.6)).sum(),
    }

def main():
    print("=" * 70)
    print("Detailed evaluation breakdown")
    print("=" * 70)
    
    models = {
        'ViLT': 'logs/predictions/vilt_test_predictions.csv',
        'BERT': 'logs/predictions/bert_test_predictions.csv',
        'ViT': 'logs/predictions/vit_test_predictions.csv',
    }
    
    for model_name, filepath in models.items():
        if not Path(filepath).exists():
            print(f"\n⚠️  {model_name}: file not found")
            continue
        
        print(f"\n{'='*70}")
        print(f"{model_name} analysis")
        print(f"{'='*70}")
        
        df = load_predictions(filepath)
        
        # Basic breakdown
        print(f"\nTotal samples: {len(df)}")
        print(f"Hateful samples: {(df['label'] == 1).sum()} ({(df['label'] == 1).mean()*100:.1f}%)")
        print(f"Non-hateful samples: {(df['label'] == 0).sum()} ({(df['label'] == 0).mean()*100:.1f}%)")
        
        # Error analysis
        error_stats = analyze_errors(df)
        print(f"\nError rate: {error_stats['error_rate']*100:.2f}%")
        print(f"Hateful accuracy: {error_stats['hateful_accuracy']*100:.2f}%")
        print(f"Non-hateful accuracy: {error_stats['non_hateful_accuracy']*100:.2f}%")
        print(f"\nFalse positives (non-hateful predicted hateful): {error_stats['false_positives']} ({error_stats['fp_rate']*100:.1f}%)")
        print(f"False negatives (missed hateful): {error_stats['false_negatives']} ({error_stats['fn_rate']*100:.1f}%)")
        
        # Confidence analysis
        conf_stats = analyze_confidence(df)
        print(f"\nAvg confidence (correct): {conf_stats['avg_prob_correct']:.4f}")
        print(f"Avg confidence (incorrect): {conf_stats['avg_prob_incorrect']:.4f}")
        print(f"Low-confidence mistakes (0.4-0.6): {conf_stats['low_confidence_errors']}")
        
        # Key takeaways
        print(f"\n💡 Insights:")
        if error_stats['fn_rate'] > error_stats['fp_rate']:
            print(f"  - Model under-flags hateful content (FN {error_stats['fn_rate']*100:.1f}% > FP {error_stats['fp_rate']*100:.1f}%)")
            print("  - Recommendation: lower the decision threshold or up-weight hateful samples.")
        else:
            print(f"  - Model is overly sensitive (FP {error_stats['fp_rate']*100:.1f}% > FN {error_stats['fn_rate']*100:.1f}%)")
            print("  - Recommendation: raise the threshold or up-weight non-hateful samples.")
        
        if conf_stats['low_confidence_errors'] > 100:
            print(f"  - Many low-confidence mistakes ({conf_stats['low_confidence_errors']}); the model lacks certainty.")
            print("  - Recommendation: add data or strengthen regularization.")
    
    print("\n" + "=" * 70)
    print("Summary and next steps")
    print("=" * 70)
    print("\nRecommended order of improvements:")
    print("1. Finish ViLT v2 runs (hyper-parameter sweep).")
    print("2. Adjust loss class weights based on imbalance.")
    print("3. Introduce targeted data augmentation.")
    print("4. Try lightweight model ensembling.")
    print("5. If performance still stalls, consider larger backbones (e.g., CLIP).")

if __name__ == "__main__":
    main()

