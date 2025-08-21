import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from speech_disorder.dataset import MultiTaskSpeechDataset
from speech_disorder.trainer import MultiTaskTrainer
from torch.utils.data import DataLoader
import torch
import json
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix

def print_inference_results(results):
    """Print comprehensive inference results with all metrics"""
    print(f"\n{'='*80}")
    print(f"{'COMPREHENSIVE INFERENCE RESULTS':^80}")
    print(f"{'='*80}")
    
    summary = results['summary']
    inference_data = results['inference_results']
    
    # Calculate additional classification metrics
    true_labels = [0 if r['true_disease'] == 'Normal' else 1 if r['true_disease'] == 'Dysarthria' else 2 for r in inference_data]
    pred_labels = [0 if r['predicted_disease'] == 'Normal' else 1 if r['predicted_disease'] == 'Dysarthria' else 2 for r in inference_data]
    
    # Calculate precision, recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average='weighted', zero_division=0)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average='macro', zero_division=0)
    
    # Calculate per-class metrics
    per_class_precision, per_class_recall, per_class_f1, support = precision_recall_fscore_support(
        true_labels, pred_labels, average=None, zero_division=0
    )
    
    # Overall summary with all metrics
    print(f"\n📊 OVERALL PERFORMANCE SUMMARY:")
    print(f"   Total Samples: {summary['total_samples']}")
    print(f"   " + "="*50)
    print(f"   🎯 TRANSCRIPTION METRICS:")
    print(f"      Overall WER: {summary['overall_wer']:.4f}")
    print(f"      Overall CER: {summary['overall_cer']:.4f}")
    print(f"   " + "="*50)
    print(f"   🏥 DISEASE CLASSIFICATION METRICS:")
    print(f"      Accuracy: {summary['disease_accuracy']:.4f} ({summary['disease_correct']}/{summary['total_samples']})")
    print(f"      Weighted Precision: {precision:.4f}")
    print(f"      Weighted Recall: {recall:.4f}")
    print(f"      Weighted F1-Score: {f1:.4f}")
    print(f"   " + "-"*50)
    print(f"      Macro Precision: {macro_precision:.4f}")
    print(f"      Macro Recall: {macro_recall:.4f}")
    print(f"      Macro F1-Score: {macro_f1:.4f}")
    
    # Sample predictions with all metrics (first 10)
    print(f"\n📋 SAMPLE PREDICTIONS (First 10):")
    print(f"{'File':<15} {'True':<10} {'Pred':<10} {'WER':<6} {'CER':<6} {'Conf':<6} {'Text':<40}")
    print(f"{'-'*100}")
    
    for i, sample in enumerate(inference_data[:10]):
        file_name = os.path.basename(sample['file_path'])[:12] + "..."
        predicted_text = sample['predicted_text'][:37] + "..." if len(sample['predicted_text']) > 40 else sample['predicted_text']
        
        print(f"{file_name:<15} "
              f"{sample['true_disease']:<10} "
              f"{sample['predicted_disease']:<10} "
              f"{sample['wer']:<5.3f} "
              f"{sample['cer']:<5.3f} "
              f"{sample['disease_confidence']:<5.3f} "
              f"{predicted_text:<40}")
    
    # Per-class detailed metrics
    print(f"\n📈 PER-CLASS DETAILED METRICS:")
    disease_names = ['Normal', 'Dysarthria', 'Dysphonia']
    
    print(f"{'Class':<12} {'Samples':<8} {'Acc':<6} {'Prec':<6} {'Rec':<6} {'F1':<6} {'WER':<6} {'CER':<6}")
    print(f"{'-'*70}")
    
    for i, disease in enumerate(disease_names):
        if 'per_class_metrics' in summary and disease in summary['per_class_metrics']:
            metrics = summary['per_class_metrics'][disease]
            class_precision = per_class_precision[i] if i < len(per_class_precision) else 0.0
            class_recall = per_class_recall[i] if i < len(per_class_recall) else 0.0
            class_f1 = per_class_f1[i] if i < len(per_class_f1) else 0.0
            
            print(f"{disease:<12} "
                  f"{metrics['samples']:<8} "
                  f"{metrics['accuracy']:<5.3f} "
                  f"{class_precision:<5.3f} "
                  f"{class_recall:<5.3f} "
                  f"{class_f1:<5.3f} "
                  f"{metrics['wer']:<5.3f} "
                  f"{metrics['cer']:<5.3f}")
    
    # Confusion Matrix
    print(f"\n📊 CONFUSION MATRIX:")
    conf_matrix = confusion_matrix(true_labels, pred_labels)
    print(f"{'Actual // Predicted':<15} {'Normal':<8} {'Dysarthria':<12} {'Dysphonia':<10}")
    print(f"{'-'*50}")
    for i, disease in enumerate(disease_names):
        row_str = f"{disease:<15} "
        for j in range(len(disease_names)):
            row_str += f"{conf_matrix[i][j]:<8} " if j == 0 else f"{conf_matrix[i][j]:<12} " if j == 1 else f"{conf_matrix[i][j]:<10}"
        print(row_str)
    
    # Classification Report
    print(f"\n📋 DETAILED CLASSIFICATION REPORT:")
    class_report = classification_report(
        true_labels, pred_labels, 
        target_names=disease_names, 
        digits=4, 
        zero_division=0
    )
    print(class_report)
    
    # WER/CER distribution analysis
    print(f"\n📊 TRANSCRIPTION QUALITY DISTRIBUTION:")
    wer_values = [r['wer'] for r in inference_data]
    cer_values = [r['cer'] for r in inference_data]
    
    print(f"   WER Statistics:")
    print(f"      Min: {min(wer_values):.4f}, Max: {max(wer_values):.4f}")
    print(f"      Mean: {sum(wer_values)/len(wer_values):.4f}")
    print(f"      Median: {sorted(wer_values)[len(wer_values)//2]:.4f}")
    
    print(f"   CER Statistics:")
    print(f"      Min: {min(cer_values):.4f}, Max: {max(cer_values):.4f}")
    print(f"      Mean: {sum(cer_values)/len(cer_values):.4f}")
    print(f"      Median: {sorted(cer_values)[len(cer_values)//2]:.4f}")
    
    # Quality buckets
    perfect_transcriptions = [r for r in inference_data if r['wer'] == 0.0]
    good_transcriptions = [r for r in inference_data if 0.0 < r['wer'] <= 0.2]
    fair_transcriptions = [r for r in inference_data if 0.2 < r['wer'] <= 0.5]
    poor_transcriptions = [r for r in inference_data if r['wer'] > 0.5]
    
    print(f"\n   Transcription Quality Buckets:")
    print(f"      Perfect (WER=0.0): {len(perfect_transcriptions)} ({len(perfect_transcriptions)/len(inference_data)*100:.1f}%)")
    print(f"      Good (0.0<WER≤0.2): {len(good_transcriptions)} ({len(good_transcriptions)/len(inference_data)*100:.1f}%)")
    print(f"      Fair (0.2<WER≤0.5): {len(fair_transcriptions)} ({len(fair_transcriptions)/len(inference_data)*100:.1f}%)")
    print(f"      Poor (WER>0.5): {len(poor_transcriptions)} ({len(poor_transcriptions)/len(inference_data)*100:.1f}%)")
    
    # Error analysis
    print(f"\n❌ ERROR ANALYSIS:")
    disease_errors = [s for s in inference_data if not s['disease_correct']]
    print(f"   Disease Classification Errors: {len(disease_errors)} ({len(disease_errors)/len(inference_data)*100:.1f}%)")
    
    if disease_errors:
        print(f"   Disease Misclassifications:")
        error_pairs = {}
        for error in disease_errors:
            pair = f"{error['true_disease']} → {error['predicted_disease']}"
            error_pairs[pair] = error_pairs.get(pair, 0) + 1
        
        for pair, count in sorted(error_pairs.items(), key=lambda x: x[1], reverse=True):
            percentage = count / len(disease_errors) * 100
            print(f"      {pair}: {count} cases ({percentage:.1f}%)")
    
    # Best and worst performing samples
    print(f"\n🏆 BEST PERFORMING SAMPLES (Lowest WER + Correct Disease):")
    correct_disease_samples = [r for r in inference_data if r['disease_correct']]
    if correct_disease_samples:
        best_samples = sorted(correct_disease_samples, key=lambda x: x['wer'])[:3]
        for i, sample in enumerate(best_samples):
            print(f"   {i+1}. {os.path.basename(sample['file_path'])} - WER: {sample['wer']:.3f}, Disease: {sample['true_disease']}")
            print(f"      Original: {sample['original_text'][:60]}...")
            print(f"      Predicted: {sample['predicted_text'][:60]}...")
    
    print(f"\n💥 WORST PERFORMING SAMPLES (Highest WER or Wrong Disease):")
    # Combine high WER and wrong disease predictions
    worst_samples = sorted(inference_data, key=lambda x: (not x['disease_correct'], x['wer']), reverse=True)[:3]
    for i, sample in enumerate(worst_samples):
        disease_status = "✗" if not sample['disease_correct'] else "✓"
        print(f"   {i+1}. {os.path.basename(sample['file_path'])} - WER: {sample['wer']:.3f}, Disease: {disease_status}")
        print(f"      True: {sample['true_disease']} | Pred: {sample['predicted_disease']} (conf: {sample['disease_confidence']:.3f})")
        print(f"      Original: {sample['original_text'][:60]}...")
        print(f"      Predicted: {sample['predicted_text'][:60]}...")

def save_inference_results(results, output_path):
    """Save comprehensive inference results with all metrics"""
    inference_data = results['inference_results']
    
    # Calculate additional metrics for saving
    true_labels = [0 if r['true_disease'] == 'Normal' else 1 if r['true_disease'] == 'Dysarthria' else 2 for r in inference_data]
    pred_labels = [0 if r['predicted_disease'] == 'Normal' else 1 if r['predicted_disease'] == 'Dysarthria' else 2 for r in inference_data]
    
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average='weighted', zero_division=0)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average='macro', zero_division=0)
    
    # Save detailed results as JSON
    json_path = output_path.replace('.csv', '_detailed.json')
    
    # Add computed metrics to results
    enhanced_results = results.copy()
    enhanced_results['summary']['weighted_precision'] = precision
    enhanced_results['summary']['weighted_recall'] = recall
    enhanced_results['summary']['weighted_f1'] = f1
    enhanced_results['summary']['macro_precision'] = macro_precision
    enhanced_results['summary']['macro_recall'] = macro_recall
    enhanced_results['summary']['macro_f1'] = macro_f1
    
    with open(json_path, 'w') as f:
        json.dump(enhanced_results, f, indent=2)
    print(f"✓ Detailed results saved: {json_path}")
    
    # Save comprehensive CSV
    csv_data = []
    for sample in inference_data:
        csv_data.append({
            'file_path': sample['file_path'],
            'original_text': sample['original_text'],
            'predicted_text': sample['predicted_text'],
            'original_text_normalized': sample['original_text_normalized'],
            'predicted_text_normalized': sample['predicted_text_normalized'],
            'wer': sample['wer'],
            'cer': sample['cer'],
            'true_disease': sample['true_disease'],
            'predicted_disease': sample['predicted_disease'],
            'disease_confidence': sample['disease_confidence'],
            'disease_correct': sample['disease_correct'],
            'normal_prob': sample['all_disease_probs']['Normal'],
            'dysarthria_prob': sample['all_disease_probs']['Dysarthria'],
            'dysphonia_prob': sample['all_disease_probs']['Dysphonia']
        })
    
    df = pd.DataFrame(csv_data)
    df.to_csv(output_path, index=False)
    print(f"✓ Comprehensive CSV saved: {output_path}")
    
    # Save metrics summary
    summary_stats_path = output_path.replace('.csv', '_metrics_summary.txt')
    with open(summary_stats_path, 'w') as f:
        f.write("=== COMPREHENSIVE INFERENCE METRICS SUMMARY ===\n\n")
        f.write(f"Total Samples: {results['summary']['total_samples']}\n\n")
        
        f.write("TRANSCRIPTION METRICS:\n")
        f.write(f"  Overall WER: {results['summary']['overall_wer']:.4f}\n")
        f.write(f"  Overall CER: {results['summary']['overall_cer']:.4f}\n\n")
        
        f.write("DISEASE CLASSIFICATION METRICS:\n")
        f.write(f"  Accuracy: {results['summary']['disease_accuracy']:.4f}\n")
        f.write(f"  Weighted Precision: {precision:.4f}\n")
        f.write(f"  Weighted Recall: {recall:.4f}\n")
        f.write(f"  Weighted F1-Score: {f1:.4f}\n")
        f.write(f"  Macro Precision: {macro_precision:.4f}\n")
        f.write(f"  Macro Recall: {macro_recall:.4f}\n")
        f.write(f"  Macro F1-Score: {macro_f1:.4f}\n\n")
        
        if 'per_class_metrics' in results['summary']:
            f.write("PER-CLASS METRICS:\n")
            for disease, metrics in results['summary']['per_class_metrics'].items():
                f.write(f"  {disease}:\n")
                f.write(f"    Samples: {metrics['samples']}\n")
                f.write(f"    Accuracy: {metrics['accuracy']:.4f}\n")
                f.write(f"    WER: {metrics['wer']:.4f}\n")
                f.write(f"    CER: {metrics['cer']:.4f}\n\n")
    
    print(f"✓ Metrics summary saved: {summary_stats_path}")

def main():
    parser = argparse.ArgumentParser(description="Multi-Task Model Inference with Comprehensive Metrics")
    
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--data_dir", type=str, default="../data", help="Data directory")
    parser.add_argument("--test_file", type=str, default="custom_test.csv", help="Test CSV file")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    parser.add_argument("--save_results", type=str, default=None, help="Path to save results CSV")
    
    args = parser.parse_args()
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    
    # Resolve paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, args.data_dir)
    data_dir = os.path.normpath(data_dir)
    
    test_csv = os.path.join(data_dir, args.test_file)
    
    print(f"=== MULTI-TASK MODEL COMPREHENSIVE INFERENCE ===")
    print(f"Model: {args.model_path}")
    print(f"Test data: {test_csv}")
    print(f"Device: {device}")
    
    # Check files exist
    if not os.path.exists(test_csv):
        print(f"Error: Test file not found: {test_csv}")
        return
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model checkpoint not found: {args.model_path}")
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
        collate_fn=test_dataset.get_collate_fn(),
        num_workers=2,
        pin_memory=True if device == "cuda" else False
    )
    
    print(f"Test dataset: {len(test_dataset)} samples")
    
    # Run inference
    print("Running comprehensive model inference...")
    results = trainer.inference_detailed(test_loader)
    
    # Print comprehensive results
    print_inference_results(results)
    
    # Save results if requested
    if args.save_results:
        save_inference_results(results, args.save_results)
    
    print(f"\n{'='*80}")
    print(f"{'COMPREHENSIVE INFERENCE COMPLETED':^80}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()