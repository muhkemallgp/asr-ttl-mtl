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
import jiwer
import numpy as np

def print_inference_results(results):
    """Print comprehensive inference results with all metrics"""
    print(f"\n{'='*80}")
    print(f"{'COMPREHENSIVE INFERENCE RESULTS':^80}")
    print(f"{'='*80}")
    
    # Extract metrics directly from results structure
    inference_data = results['inference_results']
    total_samples = results['total_samples']
    overall_wer = results['overall_wer']
    overall_cer = results['overall_cer']
    disease_accuracy = results['disease_accuracy']
    disease_correct = results['disease_correct']
    per_class_metrics = results['per_class_metrics']
    
    # ✅ USE DYNAMIC DISEASE MAPPING (from actual data, not hardcoded)
    unique_diseases = set()
    for r in inference_data:
        unique_diseases.add(r['true_disease'].lower())
        unique_diseases.add(r['predicted_disease'].lower())
    
    disease_list = sorted(list(unique_diseases))
    disease_to_class_dynamic = {disease: idx for idx, disease in enumerate(disease_list)}
    
    # Calculate additional classification metrics
    true_labels = []
    pred_labels = []
    
    for r in inference_data:
        true_disease = r['true_disease'].lower()
        pred_disease = r['predicted_disease'].lower()
        
        true_labels.append(disease_to_class_dynamic.get(true_disease, 0))
        pred_labels.append(disease_to_class_dynamic.get(pred_disease, 0))
    
    # Calculate precision, recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average='weighted', zero_division=0)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average='macro', zero_division=0)
    
    # Calculate per-class metrics
    per_class_precision, per_class_recall, per_class_f1, support = precision_recall_fscore_support(
        true_labels, pred_labels, average=None, zero_division=0
    )
    
    # Overall summary with all metrics
    print(f"\n📊 OVERALL PERFORMANCE SUMMARY:")
    print(f"   Total Samples: {total_samples}")
    print(f"   " + "="*50)
    print(f"   🎯 TRANSCRIPTION METRICS:")
    print(f"      Overall WER: {overall_wer:.4f}")
    print(f"      Overall CER: {overall_cer:.4f}")
    print(f"   " + "="*50)
    print(f"   🏥 DISEASE CLASSIFICATION METRICS:")
    print(f"      Accuracy: {disease_accuracy:.4f} ({disease_correct}/{total_samples})")
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
    
    # ✅ DYNAMIC PER-CLASS METRICS (not hardcoded)
    print(f"\n📈 PER-CLASS DETAILED METRICS:")
    display_names = [disease.capitalize() for disease in disease_list]
    
    print(f"{'Class':<12} {'Samples':<8} {'Acc':<6} {'Prec':<6} {'Rec':<6} {'F1':<6} {'WER':<6} {'CER':<6}")
    print(f"{'-'*70}")
    
    for i, (disease, display_name) in enumerate(zip(disease_list, display_names)):
        if disease in per_class_metrics:
            metrics = per_class_metrics[disease]
            class_precision = per_class_precision[i] if i < len(per_class_precision) else 0.0
            class_recall = per_class_recall[i] if i < len(per_class_recall) else 0.0
            class_f1 = per_class_f1[i] if i < len(per_class_f1) else 0.0
            
            print(f"{display_name:<12} "
                  f"{metrics['samples']:<8} "
                  f"{metrics['accuracy']:<5.3f} "
                  f"{class_precision:<5.3f} "
                  f"{class_recall:<5.3f} "
                  f"{class_f1:<5.3f} "
                  f"{metrics['wer']:<5.3f} "
                  f"{metrics['cer']:<5.3f}")
    
    # ✅ DYNAMIC CONFUSION MATRIX
    print(f"\n📊 CONFUSION MATRIX:")
    conf_matrix = confusion_matrix(true_labels, pred_labels)
    print(f"{'Actual // Predicted':<15} ", end="")
    for name in display_names:
        print(f"{name:<12}", end="")
    print()
    print(f"{'-'*(15 + 12*len(display_names))}")
    
    for i, display_name in enumerate(display_names):
        print(f"{display_name:<15} ", end="")
        for j in range(len(display_names)):
            print(f"{conf_matrix[i][j]:<12}", end="")
        print()
    
    # ✅ DYNAMIC CLASSIFICATION REPORT
    print(f"\n📋 DETAILED CLASSIFICATION REPORT:")
    class_report = classification_report(
        true_labels, pred_labels, 
        target_names=display_names, 
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
    
    # ✅ USE DYNAMIC DISEASE MAPPING (not hardcoded)
    unique_diseases = set()
    for r in inference_data:
        unique_diseases.add(r['true_disease'].lower())
        unique_diseases.add(r['predicted_disease'].lower())
    
    disease_list = sorted(list(unique_diseases))
    disease_to_class_dynamic = {disease: idx for idx, disease in enumerate(disease_list)}
    
    # Calculate additional metrics for saving
    true_labels = []
    pred_labels = []
    
    for r in inference_data:
        true_disease = r['true_disease'].lower()
        pred_disease = r['predicted_disease'].lower()
        true_labels.append(disease_to_class_dynamic.get(true_disease, 0))
        pred_labels.append(disease_to_class_dynamic.get(pred_disease, 0))
    
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average='weighted', zero_division=0)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average='macro', zero_division=0)
    
    # Save detailed results as JSON
    json_path = output_path.replace('.csv', '_detailed.json')
    
    # Add computed metrics to results
    enhanced_results = results.copy()
    enhanced_results['weighted_precision'] = precision
    enhanced_results['weighted_recall'] = recall
    enhanced_results['weighted_f1'] = f1
    enhanced_results['macro_precision'] = macro_precision
    enhanced_results['macro_recall'] = macro_recall
    enhanced_results['macro_f1'] = macro_f1
    
    with open(json_path, 'w') as f:
        json.dump(enhanced_results, f, indent=2)
    print(f"✓ Detailed results saved: {json_path}")
    
    # ✅ DYNAMIC CSV COLUMNS (support any disease type)
    csv_data = []
    for sample in inference_data:
        row = {
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
            'disease_correct': sample['disease_correct']
        }
        
        # Add all disease probabilities dynamically
        for disease, prob in sample['all_disease_probs'].items():
            row[f'{disease}_prob'] = prob
        
        csv_data.append(row)
    
    df = pd.DataFrame(csv_data)
    df.to_csv(output_path, index=False)
    print(f"✓ Comprehensive CSV saved: {output_path}")
    
    # Save metrics summary
    summary_stats_path = output_path.replace('.csv', '_metrics_summary.txt')
    with open(summary_stats_path, 'w') as f:
        f.write("=== COMPREHENSIVE INFERENCE METRICS SUMMARY ===\n\n")
        f.write(f"Total Samples: {results['total_samples']}\n\n")
        
        f.write("TRANSCRIPTION METRICS:\n")
        f.write(f"  Overall WER: {results['overall_wer']:.4f}\n")
        f.write(f"  Overall CER: {results['overall_cer']:.4f}\n\n")
        
        f.write("DISEASE CLASSIFICATION METRICS:\n")
        f.write(f"  Accuracy: {results['disease_accuracy']:.4f}\n")
        f.write(f"  Weighted Precision: {precision:.4f}\n")
        f.write(f"  Weighted Recall: {recall:.4f}\n")
        f.write(f"  Weighted F1-Score: {f1:.4f}\n")
        f.write(f"  Macro Precision: {macro_precision:.4f}\n")
        f.write(f"  Macro Recall: {macro_recall:.4f}\n")
        f.write(f"  Macro F1-Score: {macro_f1:.4f}\n\n")
        
        if 'per_class_metrics' in results:
            f.write("PER-CLASS METRICS:\n")
            for disease, metrics in results['per_class_metrics'].items():
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
    trainer.model.to(device)
    trainer.disease_classifier.to(device)
    
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
    results = run_inference_detailed(trainer, test_loader)
    
    # Print comprehensive results
    print_inference_results(results)
    
    # Save results if requested
    if args.save_results:
        save_inference_results(results, args.save_results)
    
    print(f"\n{'='*80}")
    print(f"{'COMPREHENSIVE INFERENCE COMPLETED':^80}")
    print(f"{'='*80}")

def run_inference_detailed(trainer, dataloader):
    """Run detailed inference that EXACTLY matches trainer's evaluation structure"""
    trainer.model.eval()
    trainer.disease_classifier.eval()
    
    all_results = []
    total_wer = 0
    total_cer = 0
    total_correct_disease = 0
    total_samples = 0
    
    # ✅ USE TRAINER'S DISEASE MAPPING (not hardcoded)
    class_to_disease = trainer.class_to_disease
    disease_to_class = trainer.disease_to_class
    
    # ✅ DYNAMIC PER-CLASS TRACKING (based on trainer's mapping)
    disease_names = list(class_to_disease.values())
    per_class_metrics = {disease: {'correct': 0, 'total': 0, 'wer_sum': 0, 'cer_sum': 0} 
                        for disease in disease_names}
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            mels = batch_data['mels'].to(trainer.device)
            input_tokens = batch_data['input_tokens'].to(trainer.device)
            target_tokens = batch_data['target_tokens'].to(trainer.device)
            classes = batch_data['classes'].to(trainer.device)
            texts = batch_data['texts']
            paths = batch_data['paths']
            
            try:
                # ✅ EXACT SAME FORWARD PASS AS TRAINER
                audio_features = trainer.model.encoder(mels)
                disease_logits, disease_preds = trainer.classify_disease_from_audio(audio_features)
                transcription_logits = trainer.model.decoder(input_tokens, audio_features)
                
                # Get disease probabilities
                disease_probs = torch.softmax(disease_logits, dim=-1)
                
                # ✅ EXACT SAME DECODING AS TRAINER
                pred_texts = trainer.decode_predictions(transcription_logits)
                
                # Process each sample in batch
                for i in range(len(texts)):
                    # Normalize texts for comparison
                    original_text = texts[i].strip()
                    predicted_text = pred_texts[i].strip()
                    
                    original_normalized = original_text.lower()
                    predicted_normalized = predicted_text.lower()
                    
                    # ✅ SAME WER/CER CALCULATION AS TRAINER
                    try:
                        if original_normalized and predicted_normalized:
                            wer = jiwer.wer([original_normalized], [predicted_normalized])
                            cer = jiwer.cer([original_normalized], [predicted_normalized])
                        else:
                            wer, cer = 1.0, 1.0
                    except:
                        wer, cer = 1.0, 1.0
                    
                    # ✅ USE TRAINER'S DISEASE MAPPING
                    true_class = classes[i].item()
                    pred_class = disease_preds[i].item()
                    
                    true_disease = class_to_disease.get(true_class, 'normal')
                    predicted_disease = class_to_disease.get(pred_class, 'normal')
                    
                    disease_correct = true_class == pred_class
                    disease_confidence = disease_probs[i][pred_class].item()
                    
                    # ✅ GET ALL DISEASE PROBABILITIES USING TRAINER'S MAPPING
                    all_disease_probs = {}
                    for class_id, disease_name in class_to_disease.items():
                        all_disease_probs[disease_name] = disease_probs[i][class_id].item()
                    
                    # Store result
                    result = {
                        'file_path': paths[i],
                        'original_text': original_text,
                        'predicted_text': predicted_text,
                        'original_text_normalized': original_normalized,
                        'predicted_text_normalized': predicted_normalized,
                        'wer': wer,
                        'cer': cer,
                        'true_disease': true_disease,
                        'predicted_disease': predicted_disease,
                        'disease_confidence': disease_confidence,
                        'disease_correct': disease_correct,
                        'all_disease_probs': all_disease_probs
                    }
                    
                    all_results.append(result)
                    
                    # Update totals
                    total_wer += wer
                    total_cer += cer
                    if disease_correct:
                        total_correct_disease += 1
                    total_samples += 1
                    
                    # Update per-class metrics
                    if true_disease in per_class_metrics:
                        per_class_metrics[true_disease]['total'] += 1
                        per_class_metrics[true_disease]['wer_sum'] += wer
                        per_class_metrics[true_disease]['cer_sum'] += cer
                        if disease_correct:
                            per_class_metrics[true_disease]['correct'] += 1
                            
            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                continue
    
    # Calculate summary metrics
    overall_wer = total_wer / total_samples if total_samples > 0 else 1.0
    overall_cer = total_cer / total_samples if total_samples > 0 else 1.0
    disease_accuracy = total_correct_disease / total_samples if total_samples > 0 else 0.0
    
    # Calculate per-class summary
    per_class_summary = {}
    for disease, metrics in per_class_metrics.items():
        if metrics['total'] > 0:
            per_class_summary[disease] = {
                'samples': metrics['total'],
                'accuracy': metrics['correct'] / metrics['total'],
                'wer': metrics['wer_sum'] / metrics['total'],
                'cer': metrics['cer_sum'] / metrics['total']
            }
        else:
            per_class_summary[disease] = {
                'samples': 0,
                'accuracy': 0.0,
                'wer': 1.0,
                'cer': 1.0
            }
    
    # ✅ EXACT SAME STRUCTURE AS TRAINER's compute_detailed_metrics
    results = {
        'total_samples': total_samples,
        'overall_wer': overall_wer,
        'overall_cer': overall_cer,
        'disease_accuracy': disease_accuracy,
        'disease_correct': total_correct_disease,
        'per_class_metrics': per_class_summary,
        'inference_results': all_results
    }
    
    return results

if __name__ == "__main__":
    main()