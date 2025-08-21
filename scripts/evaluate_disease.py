import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from speech_disorder.dataset import MultiTaskSpeechDataset, collate_fn
from speech_disorder.trainer import MultiTaskTrainer
from speech_disorder.config import TrainingConfig
from torch.utils.data import DataLoader
import torch
import json
import numpy as np

def print_detailed_results(results):
    """Print comprehensive evaluation results"""
    print(f"\n{'='*60}")
    print(f"{'COMPREHENSIVE EVALUATION RESULTS':^60}")
    print(f"{'='*60}")
    
    # Overall metrics
    overall = results['overall']
    print(f"\n📊 OVERALL PERFORMANCE:")
    print(f"   Total Samples: {overall['samples']}")
    print(f"   WER:          {overall['wer']*100:.2f}%")
    print(f"   CER:          {overall['cer']*100:.2f}%")
    print(f"   Accuracy:     {overall['accuracy']*100:.2f}%")
    print(f"   Precision:    {overall['precision']*100:.2f}%")
    print(f"   Recall:       {overall['recall']*100:.2f}%")
    print(f"   F1-Score:     {overall['f1']*100:.2f}%")
    
    # Per-class metrics
    print(f"\n📋 PER-CLASS PERFORMANCE:")
    print(f"{'Class':<12} {'Samples':<8} {'WER':<8} {'CER':<8} {'Acc':<8} {'Prec':<8} {'Rec':<8} {'F1':<8}")
    print(f"{'-'*80}")
    
    for class_name, metrics in results['per_class'].items():
        print(f"{class_name:<12} "
              f"{metrics['samples']:<8} "
              f"{metrics['wer']*100:<7.1f}% "
              f"{metrics['cer']*100:<7.1f}% "
              f"{metrics['accuracy']*100:<7.1f}% "
              f"{metrics['precision']*100:<7.1f}% "
              f"{metrics['recall']*100:<7.1f}% "
              f"{metrics['f1']*100:<7.1f}%")
    
    # Confusion Matrix
    print(f"\n🔄 CONFUSION MATRIX:")
    conf_matrix = np.array(results['confusion_matrix'])
    class_names = ['Normal', 'Dysarthria', 'Dysphonia']
    
    print(f"{'Predicted →':<12}", end="")
    for name in class_names:
        print(f"{name:<12}", end="")
    print()
    
    for i, true_class in enumerate(class_names):
        print(f"{true_class:<12}", end="")
        for j in range(len(class_names)):
            print(f"{conf_matrix[i,j]:<12}", end="")
        print()
    
    # Classification report details
    print(f"\n📈 DETAILED CLASSIFICATION REPORT:")
    class_report = results['classification_report']
    for class_name in class_names:
        if class_name.lower() in class_report:
            metrics = class_report[class_name.lower()]
            print(f"   {class_name}:")
            print(f"      Precision: {metrics['precision']*100:.2f}%")
            print(f"      Recall:    {metrics['recall']*100:.2f}%")
            print(f"      F1-Score:  {metrics['f1-score']*100:.2f}%")
            print(f"      Support:   {metrics['support']} samples")

def main():
    parser = argparse.ArgumentParser(description="Comprehensive Multi-Task Model Evaluation")
    
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--test_file", type=str, default="custom_test.csv")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--save_results", type=str, default=None, help="Path to save JSON results")
    
    args = parser.parse_args()
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    
    # Resolve paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, args.data_dir)
    data_dir = os.path.normpath(data_dir)
    
    test_csv = os.path.join(data_dir, args.test_file)
    
    print(f"=== COMPREHENSIVE MULTI-TASK EVALUATION ===")
    print(f"Model: {args.model_path}")
    print(f"Test data: {test_csv}")
    print(f"Device: {device}")
    
    # Check files exist
    if not os.path.exists(test_csv):
        print(f"Error: Test file tidak ditemukan: {test_csv}")
        return
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model checkpoint tidak ditemukan: {args.model_path}")
        return
    
    # Load model
    print("Loading trained model...")
    trainer = MultiTaskTrainer.load_from_checkpoint(args.model_path)
    
    # Create test dataset
    print("Loading test dataset...")
    test_dataset = MultiTaskSpeechDataset(test_csv, trainer.config)
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True if device == "cuda" else False
    )
    
    print(f"Test dataset: {len(test_dataset)} samples")
    
    # Run comprehensive evaluation
    print("Running comprehensive evaluation...")
    results = trainer.evaluate_detailed(test_loader)
    
    # Print results
    print_detailed_results(results)
    
    # Save results if requested
    if args.save_results:
        print(f"\nSaving results to: {args.save_results}")
        with open(args.save_results, 'w') as f:
            json.dump(results, f, indent=2)
        print("✓ Results saved successfully")

if __name__ == "__main__":
    main()