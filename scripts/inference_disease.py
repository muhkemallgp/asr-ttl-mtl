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
from datetime import datetime

def run_inference(trainer, dataloader):
    """Run production inference that matches trainer's evaluation exactly"""
    trainer.model.eval()
    trainer.disease_classifier.eval()
    
    all_results = []
    total_wer = 0
    total_cer = 0
    total_correct_disease = 0
    total_samples = 0
    
    # Use trainer's exact disease mapping
    class_to_disease = trainer.class_to_disease
    disease_names = list(class_to_disease.values())
    per_class_metrics = {disease: {'correct': 0, 'total': 0, 'wer_sum': 0, 'cer_sum': 0} 
                        for disease in disease_names}
    
    with torch.no_grad():
        for batch_data in dataloader:
            mels = batch_data['mels'].to(trainer.device)
            input_tokens = batch_data['input_tokens'].to(trainer.device)
            target_tokens = batch_data['target_tokens'].to(trainer.device)
            classes = batch_data['classes'].to(trainer.device)
            texts = batch_data['texts']
            paths = batch_data['paths']
            
            try:
                # Forward pass - exact same as trainer
                audio_features = trainer.model.encoder(mels)
                disease_logits, disease_preds = trainer.classify_disease_from_audio(audio_features)
                transcription_logits = trainer.model.decoder(input_tokens, audio_features)
                
                # Get disease probabilities
                disease_probs = torch.softmax(disease_logits, dim=-1)
                
                # Decode transcriptions - exact same as trainer
                pred_texts = trainer.decode_predictions(transcription_logits)
                
                # Process each sample in batch
                for i in range(len(texts)):
                    original_text = texts[i].strip()
                    predicted_text = pred_texts[i].strip()
                    
                    original_normalized = original_text.lower()
                    predicted_normalized = predicted_text.lower()
                    
                    # Calculate WER/CER - same as trainer
                    try:
                        if original_normalized and predicted_normalized:
                            wer = jiwer.wer([original_normalized], [predicted_normalized])
                            cer = jiwer.cer([original_normalized], [predicted_normalized])
                        else:
                            wer, cer = 1.0, 1.0
                    except:
                        wer, cer = 1.0, 1.0
                    
                    # Disease classification - use trainer's mapping
                    true_class = classes[i].item()
                    pred_class = disease_preds[i].item()
                    
                    true_disease = class_to_disease.get(true_class, 'normal')
                    predicted_disease = class_to_disease.get(pred_class, 'normal')
                    
                    disease_correct = true_class == pred_class
                    disease_confidence = disease_probs[i][pred_class].item()
                    
                    # Get all disease probabilities
                    all_disease_probs = {}
                    for class_id, disease_name in class_to_disease.items():
                        if class_id < disease_probs.size(1):
                            all_disease_probs[disease_name] = disease_probs[i][class_id].item()
                        else:
                            all_disease_probs[disease_name] = 0.0
                    
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
                        'true_class': true_class,
                        'predicted_class': pred_class,
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
                print(f"Error processing batch: {e}")
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
    
    results = {
        'total_samples': total_samples,
        'overall_wer': overall_wer,
        'overall_cer': overall_cer,
        'disease_accuracy': disease_accuracy,
        'disease_correct': total_correct_disease,
        'per_class_metrics': per_class_summary,
        'inference_results': all_results,
        'model_info': {
            'class_to_disease': class_to_disease,
            'model_size': trainer.config.model_size,
            'is_english_only': trainer.is_english_only
        }
    }
    
    return results

def calculate_additional_metrics(results):
    """Calculate precision, recall, F1 scores"""
    inference_data = results['inference_results']
    class_to_disease = results['model_info']['class_to_disease']
    
    # Extract true and predicted classes
    true_classes = [r['true_class'] for r in inference_data]
    pred_classes = [r['predicted_class'] for r in inference_data]
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_classes, pred_classes, average='weighted', zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        true_classes, pred_classes, average='macro', zero_division=0
    )
    per_class_precision, per_class_recall, per_class_f1, support = precision_recall_fscore_support(
        true_classes, pred_classes, average=None, zero_division=0
    )
    
    return {
        'weighted_precision': precision,
        'weighted_recall': recall,
        'weighted_f1': f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'per_class_precision': per_class_precision.tolist(),
        'per_class_recall': per_class_recall.tolist(),
        'per_class_f1': per_class_f1.tolist(),
        'per_class_support': support.tolist()
    }

def print_results(results, additional_metrics):
    """Print comprehensive results"""
    print(f"\n{'='*80}")
    print(f"{'INFERENCE RESULTS':^80}")
    print(f"{'='*80}")
    
    # Model info
    model_info = results['model_info']
    print(f"\nModel Information:")
    print(f"  Model Size: {model_info['model_size']}")
    print(f"  Model Type: {'English-only' if model_info['is_english_only'] else 'Multilingual'}")
    print(f"  Disease Classes: {list(model_info['class_to_disease'].values())}")
    
    # Overall metrics
    print(f"\nOverall Performance:")
    print(f"  Total Samples: {results['total_samples']}")
    print(f"  Disease Accuracy: {results['disease_accuracy']:.4f} ({results['disease_correct']}/{results['total_samples']})")
    print(f"  Overall WER: {results['overall_wer']:.4f}")
    print(f"  Overall CER: {results['overall_cer']:.4f}")
    
    # Classification metrics
    print(f"\nClassification Metrics:")
    print(f"  Weighted Precision: {additional_metrics['weighted_precision']:.4f}")
    print(f"  Weighted Recall: {additional_metrics['weighted_recall']:.4f}")
    print(f"  Weighted F1-Score: {additional_metrics['weighted_f1']:.4f}")
    print(f"  Macro Precision: {additional_metrics['macro_precision']:.4f}")
    print(f"  Macro Recall: {additional_metrics['macro_recall']:.4f}")
    print(f"  Macro F1-Score: {additional_metrics['macro_f1']:.4f}")
    
    # Per-class metrics
    print(f"\nPer-Class Performance:")
    print(f"{'Disease':<12} {'Samples':<8} {'Accuracy':<9} {'Precision':<10} {'Recall':<8} {'F1-Score':<9} {'WER':<6} {'CER':<6}")
    print(f"{'-'*80}")
    
    class_to_disease = model_info['class_to_disease']
    for class_id, disease in class_to_disease.items():
        metrics = results['per_class_metrics'][disease]
        precision = additional_metrics['per_class_precision'][class_id]
        recall = additional_metrics['per_class_recall'][class_id]
        f1 = additional_metrics['per_class_f1'][class_id]
        
        print(f"{disease.capitalize():<12} "
              f"{metrics['samples']:<8} "
              f"{metrics['accuracy']:<9.4f} "
              f"{precision:<10.4f} "
              f"{recall:<8.4f} "
              f"{f1:<9.4f} "
              f"{metrics['wer']:<6.3f} "
              f"{metrics['cer']:<6.3f}")
    
    # Confusion matrix
    inference_data = results['inference_results']
    true_classes = [r['true_class'] for r in inference_data]
    pred_classes = [r['predicted_class'] for r in inference_data]
    conf_matrix = confusion_matrix(true_classes, pred_classes)
    
    print(f"\nConfusion Matrix:")
    disease_names = [disease.capitalize() for disease in class_to_disease.values()]
    header_label = "Actual \\ Predicted"
    print(f"{header_label:<15} " + " ".join(f"{name:<10}" for name in disease_names))
    print(f"{'-'*(15 + 11*len(disease_names))}")
    
    for i, disease in enumerate(disease_names):
        print(f"{disease:<15} " + " ".join(f"{conf_matrix[i][j]:<10}" for j in range(len(disease_names))))
    
    # Sample predictions
    print(f"\nSample Predictions (First 5):")
    print(f"{'File':<20} {'True':<12} {'Pred':<12} {'Conf':<6} {'WER':<6} {'Text':<30}")
    print(f"{'-'*90}")
    
    for i, sample in enumerate(inference_data[:5]):
        filename = os.path.basename(sample['file_path'])[:17] + "..."
        text_preview = sample['predicted_text'][:27] + "..." if len(sample['predicted_text']) > 30 else sample['predicted_text']
        
        print(f"{filename:<20} "
              f"{sample['true_disease']:<12} "
              f"{sample['predicted_disease']:<12} "
              f"{sample['disease_confidence']:<6.3f} "
              f"{sample['wer']:<6.3f} "
              f"{text_preview:<30}")

def save_results(results, additional_metrics, output_path):
    """Save results to files"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_path = output_path.replace('.csv', f'_{timestamp}')
    
    # Save detailed CSV
    csv_path = f"{base_path}.csv"
    csv_data = []
    
    for sample in results['inference_results']:
        row = {
            'file_path': sample['file_path'],
            'original_text': sample['original_text'],
            'predicted_text': sample['predicted_text'],
            'wer': sample['wer'],
            'cer': sample['cer'],
            'true_disease': sample['true_disease'],
            'predicted_disease': sample['predicted_disease'],
            'true_class': sample['true_class'],
            'predicted_class': sample['predicted_class'],
            'disease_confidence': sample['disease_confidence'],
            'disease_correct': sample['disease_correct']
        }
        
        # Add disease probabilities
        for disease, prob in sample['all_disease_probs'].items():
            row[f'{disease}_prob'] = prob
        
        csv_data.append(row)
    
    df = pd.DataFrame(csv_data)
    df.to_csv(csv_path, index=False)
    print(f"Results saved to: {csv_path}")
    
    # Save summary JSON
    json_path = f"{base_path}_summary.json"
    summary = {
        'model_info': results['model_info'],
        'overall_metrics': {
            'total_samples': results['total_samples'],
            'disease_accuracy': results['disease_accuracy'],
            'overall_wer': results['overall_wer'],
            'overall_cer': results['overall_cer']
        },
        'classification_metrics': additional_metrics,
        'per_class_metrics': results['per_class_metrics'],
        'timestamp': timestamp
    }
    
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {json_path}")

def main():
    parser = argparse.ArgumentParser(description="Multi-Task Model Inference")
    
    parser.add_argument("--model_path", type=str, required=True, 
                       help="Path to trained model checkpoint")
    parser.add_argument("--data_dir", type=str, default="../data", 
                       help="Data directory")
    parser.add_argument("--test_file", type=str, default="custom_test.csv", 
                       help="Test CSV file")
    parser.add_argument("--batch_size", type=int, default=8, 
                       help="Batch size for inference")
    parser.add_argument("--device", type=str, default="auto", 
                       help="Device to use (auto/cpu/cuda)")
    parser.add_argument("--save_results", type=str, default=None, 
                       help="Path to save results (optional)")
    
    args = parser.parse_args()
    
    # Set device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    # Resolve paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, args.data_dir)
    data_dir = os.path.normpath(data_dir)
    test_csv = os.path.join(data_dir, args.test_file)
    
    print(f"Multi-Task Model Inference")
    print(f"Model: {args.model_path}")
    print(f"Test data: {test_csv}")
    print(f"Device: {device}")
    
    # Validate files
    if not os.path.exists(test_csv):
        print(f"Error: Test file not found: {test_csv}")
        return
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model checkpoint not found: {args.model_path}")
        return
    
    # Load model
    print("Loading model...")
    trainer = MultiTaskTrainer.load_from_checkpoint(args.model_path)
    trainer.model.to(device)
    trainer.disease_classifier.to(device)
    
    # Create dataset
    print("Loading dataset...")
    test_dataset = MultiTaskSpeechDataset(test_csv, trainer.config)
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=test_dataset.get_collate_fn(),
        num_workers=2,
        pin_memory=True if device == "cuda" else False
    )
    
    print(f"Dataset loaded: {len(test_dataset)} samples")
    
    # Run inference
    print("Running inference...")
    results = run_inference(trainer, test_loader)
    
    # Calculate additional metrics
    additional_metrics = calculate_additional_metrics(results)
    
    # Print results
    print_results(results, additional_metrics)
    
    # Save results if requested
    if args.save_results:
        save_results(results, additional_metrics, args.save_results)
    
    print(f"\nInference completed successfully!")

if __name__ == "__main__":
    main()