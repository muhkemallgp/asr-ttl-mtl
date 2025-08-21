import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, classification_report
import os
import csv
from tqdm import tqdm
import jiwer
import numpy as np
import json
import whisper

from .config import TrainingConfig
from whisper import load_model
from whisper.tokenizer import get_tokenizer

class MultiTaskTrainer:
    """Trainer khusus untuk multi-task learning"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        print(f"=== Multi-Task Learning Trainer ===")
        print(f"Device: {self.device}")
        
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            print(f"CUDA device: {torch.cuda.get_device_name()}")
            print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Load Whisper model
        self.model = load_model(config.model_size, device=self.device)
        print(f"✔ Whisper model loaded on {self.device}")
        
        # Check if model is english only
        self.is_english_only = '.en' in config.model_size

        # Get tokenizer dengan disease tokens
        if self.is_english_only:
            self.tokenizer = get_tokenizer(
                multilingual=False,
                language=None,
                task=None,
                include_diseases=True,
            )
        else:
            self.tokenizer = get_tokenizer(
                multilingual=self.model.is_multilingual,
                num_languages=self.model.num_languages,
                language="en",
                task="transcribe",
                include_diseases=True
        )
        
        # Print token information
        print(f"✔ Tokenizer loaded with disease support")
        print(f"  EOT token: {self.tokenizer.eot}")
        print(f"  SOT token: {self.tokenizer.sot}")
        if not self.is_english_only:
            print(f"  Language token: {getattr(self.tokenizer, 'language_token', 'N/A')}")
        print(f"  Transcribe token: {getattr(self.tokenizer, 'transcribe', 'N/A')}")
        if hasattr(self.tokenizer, 'disease_tokens'):
            print(f"  Disease tokens: {list(self.tokenizer.disease_tokens.keys())}")
        
        # Calculate vocabulary sizes
        original_vocab_size = self.model.dims.n_vocab
        
        try:
            new_vocab_size = self.tokenizer.encoding.n_vocab
        except AttributeError:
            try:
                special_tokens_count = len(self.tokenizer.encoding._special_tokens)
                mergeable_ranks_count = len(self.tokenizer.encoding._mergeable_ranks)
                new_vocab_size = special_tokens_count + mergeable_ranks_count
            except AttributeError:
                disease_tokens_count = len(config.disease_tokens) if hasattr(config, 'disease_tokens') else 3
                new_vocab_size = original_vocab_size + disease_tokens_count
        
        print(f"Vocabulary sizes: Original={original_vocab_size}, New={new_vocab_size}")
        
        # Expand vocabulary if needed
        if new_vocab_size > original_vocab_size:
            print(f"Expanding vocabulary: {original_vocab_size} -> {new_vocab_size}")
            self.model.resize_token_embeddings(new_vocab_size)
            print(f"✔ Vocabulary expanded")
        
        # Disease classification head
        self.disease_classifier = nn.Sequential(
            nn.Linear(self.model.dims.n_audio_state, self.model.dims.n_audio_state // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.model.dims.n_audio_state // 2, 3),
            nn.LogSoftmax(dim=-1)
        ).to(self.device)
        print(f"✔ Disease classifier created on {self.device}")

        # Freeze encoder if requested
        if config.freeze_encoder:
            for param in self.model.encoder.parameters():
                param.requires_grad = False
            print("✔ Encoder frozen")
        else:
            print("Encoder is trainable")
        
        # Collect trainable parameters (proper parameter collection)
        trainable_params = []
        
        # Add disease classifier parameters
        trainable_params.extend(self.disease_classifier.parameters())
        
        # Add decoder parameters (fine-tuning)
        for param in self.model.decoder.parameters():
            if param.requires_grad:
                trainable_params.append(param)
        
        # Filter out frozen embeddings manually in optimizer
        if hasattr(self.model.decoder, 'token_embedding'):
            embedding_weight = self.model.decoder.token_embedding.weight
            if embedding_weight.size(0) > original_vocab_size:
                # Create custom parameter group for embeddings
                new_embedding_params = [embedding_weight[original_vocab_size:]]
                trainable_params.extend(new_embedding_params)
                print(f"✔ New embeddings added to training: {embedding_weight.size(0) - original_vocab_size} tokens")

        print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
        
        # Optimizer with custom parameter groups
        param_groups = []
        
        # Disease classifier with normal learning rate
        param_groups.append({
            'params': list(self.disease_classifier.parameters()),
            'lr': config.learning_rate,
            'weight_decay': config.weight_decay
        })
        
        # Encoder parameters with lower learning rate
        if not config.freeze_encoder:
            encoder_params = [p for p in self.model.encoder.parameters() if p.requires_grad]
            if encoder_params:
                param_groups.append({
                    'params': encoder_params,
                    'lr': config.learning_rate * 0.5,
                    'weight_decay': config.weight_decay
                })
                print(f"✔ Encoder parameters added to training: {sum(p.numel() for p in encoder_params):,}")
            
        # Decoder parameters with normal learning rate
        decoder_params = [p for p in self.model.decoder.parameters() if p.requires_grad and p.numel() < 1000000]
        if decoder_params:
            param_groups.append({
                'params': decoder_params,
                'lr': config.learning_rate,
                'weight_decay': config.weight_decay
            })
        
        # New token embeddings with lower learning rate
        if hasattr(self.model.decoder, 'token_embedding'):
            embedding_weight = self.model.decoder.token_embedding.weight
            if embedding_weight.size(0) > original_vocab_size:
                param_groups.append({
                    'params': [embedding_weight],
                    'lr': config.learning_rate * 0.1,  # Lower LR for embeddings
                    'weight_decay': 0.0
                })
        
        self.optimizer = torch.optim.AdamW(param_groups)
        
        # Loss functions
        self.transcription_loss = nn.CrossEntropyLoss(ignore_index=-100)
        self.classification_loss = nn.NLLLoss()
        
        self.transcription_loss = self.transcription_loss.to(self.device)
        self.classification_loss = self.classification_loss.to(self.device)
        
        # Multi-task weights
        self.alpha = config.alpha
        self.beta = config.beta
        
        self.class_to_disease = config.class_to_disease
        self.best_metric = 0.0
        
        print(f"✔ Trainer initialized successfully")

    def compute_alpha_beta(self, epoch, trans_loss, class_loss):
        """Compute adaptive alpha and beta based on loss values"""
        if epoch >= 0:
            trans_loss_value = trans_loss.item() + 1e-6  # Avoid division by zero
            class_loss_value = class_loss.item() + 1e-6
            alpha = 1 / trans_loss_value
            beta = 1 / class_loss_value

            # Normalize weights
            total_weight = alpha + beta
            alpha /= total_weight
            beta /= total_weight
            return alpha, beta
        else:
            # Initial weights for first epoch
            return self.alpha, self.beta
    
    def update_loss_weights(self, epoch, trans_loss=None, class_loss=None):
        """Update alpha dan beta berdasarkan loss values"""
        if trans_loss is not None and class_loss is not None:
            # Use adaptive weights based on loss values
            self.alpha, self.beta = self.compute_alpha_beta(epoch, trans_loss, class_loss)
        else:
            # Fallback to epoch-based weights for first epoch
            print(f"Using epoch-based weights for epoch {epoch+1}")
            progress = epoch / self.config.epochs
            if progress < 0.3:
                self.alpha = 0.8
                self.beta = 0.2
            elif progress < 0.7:
                self.alpha = 0.6
                self.beta = 0.4
            else:
                self.alpha = 0.5
                self.beta = 0.5
        
        print(f"Epoch {epoch+1}: Updated loss weights α={self.alpha:.4f}, β={self.beta:.4f}")

    def classify_disease_from_audio(self, audio_features):
        """Disease classification from audio encoder features"""
        pooled_features = audio_features.mean(dim=1)
        disease_logits = self.disease_classifier(pooled_features)
        predicted_classes = torch.argmax(disease_logits, dim=-1)
        return disease_logits, predicted_classes

    def compute_metrics_simplified(self, transcription_logits, target_tokens, disease_logits, disease_labels, texts, debug_mode=False):
        """Simplified metrics computation with precision/recall, using proper Whisper tokens"""
        
        # TRANSCRIPTION METRICS
        pred_tokens = torch.argmax(transcription_logits, dim=-1)
        
        # Replace -100 with proper EOT token from tokenizer
        pred_clean = pred_tokens.clone()
        target_clean = target_tokens.clone()
        pred_clean[pred_clean == -100] = self.tokenizer.eot
        target_clean[target_clean == -100] = self.tokenizer.eot
        
        pred_texts = []
        ref_texts = []
        
        if debug_mode:
            print(f"\n=== METRICS DEBUG (Using EOT: {self.tokenizer.eot}) ===")
            print(f"Processing {len(texts)} samples...")
        
        for i, (pred, target, original_text) in enumerate(zip(pred_clean, target_clean, texts)):
            try:
                # Decode prediction tokens
                pred_text = self.tokenizer.decode(pred.cpu().numpy())
                
                # Use original text as reference (like reference code)
                ref_text = original_text.strip()
                
                # Clean special tokens from both texts using Whisper tokenizer constants
                special_token_strings = []
                
                # Add Whisper standard tokens
                if hasattr(self.tokenizer, 'sot_sequence'):
                    for token_id in self.tokenizer.sot_sequence:
                        token_str = self.tokenizer.decode([token_id])
                        if token_str and token_str not in special_token_strings:
                            special_token_strings.append(token_str)
                
                # Add common special tokens
                common_tokens = [
                    "<|startoftranscript|>", "<|en|>", "<|transcribe|>", 
                    "<|endoftext|>", "<|notimestamps|>", "<|nospeech|>",
                    "<|startoflm|>", "<|startofprev|>", "<|translate|>"
                ]
                special_token_strings.extend(common_tokens)
                
                # Add disease tokens if available
                if hasattr(self.tokenizer, 'disease_tokens'):
                    for disease_name, token_id in self.tokenizer.disease_tokens.items():
                        token_str = self.tokenizer.decode([token_id])
                        if token_str and token_str not in special_token_strings:
                            special_token_strings.append(token_str)
                        # Also add formatted versions
                        formatted_tokens = [f"<|{disease_name}|>", f"<|{disease_name.lower()}|>"]
                        special_token_strings.extend(formatted_tokens)
                
                # Clean tokens from both texts
                for token_str in special_token_strings:
                    pred_text = pred_text.replace(token_str, "")
                    ref_text = ref_text.replace(token_str, "")
                
                # Normalize text (like reference code)
                pred_text = pred_text.strip().lower()
                ref_text = ref_text.strip().lower()
                
                # Only add non-empty references (like reference code)
                if ref_text:
                    pred_texts.append(pred_text)
                    ref_texts.append(ref_text)
                    
                    if debug_mode and i < 3:
                        print(f"Sample {i}:")
                        print(f"  Pred: '{pred_text[:50]}...'")
                        print(f"  Ref:  '{ref_text[:50]}...'")
            
            except Exception as e:
                if debug_mode and i < 3:
                    print(f"Sample {i} decode error: {e}")
                continue
        
        # Calculate WER/CER (like reference code)
        if pred_texts and ref_texts:
            try:
                wer = jiwer.wer(ref_texts, pred_texts)
                cer = jiwer.cer(ref_texts, pred_texts)
            except Exception as e:
                if debug_mode:
                    print(f"WER/CER calculation error: {e}")
                wer, cer = 1.0, 1.0
        else:
            wer, cer = 1.0, 1.0
        
        # CLASSIFICATION METRICS (like reference code)
        _, disease_preds = torch.max(disease_logits, 1)
        disease_acc = (disease_preds == disease_labels).float().mean().item()
        
        # Calculate precision, recall, F1
        disease_labels_np = disease_labels.cpu().numpy()
        disease_preds_np = disease_preds.cpu().numpy()
        
        from sklearn.metrics import precision_recall_fscore_support
        precision, recall, f1, _ = precision_recall_fscore_support(
            disease_labels_np, disease_preds_np, average='weighted', zero_division=0
        )
        
        if debug_mode:
            print(f"\nMetrics computed (using proper Whisper tokens):")
            print(f"  WER: {wer:.4f}, CER: {cer:.4f}")
            print(f"  Disease Acc: {disease_acc:.4f}")
            print(f"  Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
            print("=" * 50)
        
        return {
            'wer': wer,
            'cer': cer,
            'disease_acc': disease_acc,
            'disease_f1': f1,
            'disease_precision': precision,
            'disease_recall': recall,
            'pred_texts': pred_texts,  # Return for debugging
            'ref_texts': ref_texts      # Return for debugging
        }

    def print_epoch_examples(self, pred_texts, ref_texts, disease_labels, disease_preds, epoch_type="TRAINING", epoch_num=None):
        """Print prediction and reference examples for each epoch"""
        print(f"\n{'='*70}")
        if epoch_num is not None:
            print(f"📝 {epoch_type} EPOCH {epoch_num} - TEXT EXAMPLES:")
        else:
            print(f"📝 {epoch_type} - TEXT EXAMPLES:")
        print(f"{'='*70}")

        # Use mapping from config: 0: 'normal', 1: 'dysphonia', 2: 'dysarthria'
        disease_names = {0: 'Normal', 1: 'Dysphonia', 2: 'Dysarthria'}
        
        # Show up to 5 examples
        num_examples = min(5, len(pred_texts))
        
        for i in range(num_examples):
            if i < len(disease_labels) and i < len(disease_preds):
                true_disease = disease_names[disease_labels[i]]
                pred_disease = disease_names[disease_preds[i]]
                disease_match = "✓" if disease_labels[i] == disease_preds[i] else "✗"
            else:
                true_disease = "N/A"
                pred_disease = "N/A"
                disease_match = "?"
            
            print(f"\nExample {i+1}:")
            print(f"  Reference: '{ref_texts[i][:80]}{'...' if len(ref_texts[i]) > 80 else ''}'")
            print(f"  Predicted: '{pred_texts[i][:80]}{'...' if len(pred_texts[i]) > 80 else ''}'")
            print(f"  Disease: {true_disease} → {pred_disease} {disease_match}")
        
        print(f"{'='*70}")

    def train_epoch(self, dataloader, epoch):
        """Training epoch with detailed text examples per epoch"""
        self.model.train()
        self.disease_classifier.train()
        
        total_loss = 0
        total_cls_loss = 0
        total_trans_loss = 0
        
        all_metrics = {
            'wer': [], 'cer': [], 'disease_acc': [], 'disease_f1': [], 
            'disease_precision': [], 'disease_recall': []
        }
        batch_count = 0
        
        # Collect ALL predictions and references for epoch summary
        epoch_pred_texts = []
        epoch_ref_texts = []
        epoch_disease_labels = []
        epoch_disease_preds = []
        
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        total_batches = len(dataloader)
        
        for batch_idx, batch_data in enumerate(tqdm(dataloader, desc=f"Training Epoch {epoch+1}")):
            self.optimizer.zero_grad()
            
            mels = batch_data['mels'].to(self.device, non_blocking=True)
            # Use correct keys from collate function
            input_tokens = batch_data['input_tokens'].to(self.device, non_blocking=True)
            target_tokens = batch_data['target_tokens'].to(self.device, non_blocking=True)
            classes = batch_data['classes'].to(self.device, non_blocking=True)
            texts = batch_data['texts']
            
            try:
                # Forward pass - use already separated tokens
                audio_features = self.model.encoder(mels)
                disease_logits, _ = self.classify_disease_from_audio(audio_features)
                transcription_logits = self.model.decoder(input_tokens, audio_features)
                
                # Compute losses
                cls_loss = self.classification_loss(disease_logits, classes)
                trans_loss = self.transcription_loss(
                    transcription_logits.reshape(-1, transcription_logits.size(-1)),
                    target_tokens.reshape(-1)
                )
    
                # Initialize alpha and beta
                if self.alpha == 0.0 or self.beta == 0.0:
                    self.alpha, self.beta = self.compute_alpha_beta(epoch, trans_loss, cls_loss)

                combined_loss = self.alpha * trans_loss + self.beta * cls_loss

                # Backward pass
                combined_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad], 
                    max_norm=self.config.gradient_clip_norm
                )
                torch.nn.utils.clip_grad_norm_(
                    self.disease_classifier.parameters(),
                    max_norm=self.config.gradient_clip_norm
                )
                
                self.optimizer.step()
                
                # Compute metrics
                debug_mode = (batch_idx < 2)  # Debug first few batches
                metrics = self.compute_metrics_simplified(
                    transcription_logits, target_tokens, disease_logits, classes, texts, 
                    debug_mode=debug_mode
                )
                
                # Collect predictions for epoch summary
                epoch_pred_texts.extend(metrics['pred_texts'])
                epoch_ref_texts.extend(metrics['ref_texts'])
                epoch_disease_labels.extend(classes.cpu().numpy())
                epoch_disease_preds.extend(torch.max(disease_logits, 1)[1].cpu().numpy())
                
                total_loss += combined_loss.item()
                total_cls_loss += cls_loss.item()
                total_trans_loss += trans_loss.item()
                
                # Add metrics to accumulation
                for key in ['wer', 'cer', 'disease_acc', 'disease_f1', 'disease_precision', 'disease_recall']:
                    all_metrics[key].append(metrics[key])
                
                batch_count += 1
                
                if self.device.type == 'cuda' and batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                if batch_idx < 3:
                    print(f"Training batch {batch_idx} failed: {e}")
                continue
        
        # Print epoch summary with examples
        self.print_epoch_examples(
            epoch_pred_texts, epoch_ref_texts, 
            epoch_disease_labels, epoch_disease_preds,
            "TRAINING", epoch+1
        )
        
        # Return averaged metrics
        if batch_count > 0:
            result = {
                'loss': total_loss / batch_count,
                'cls_loss': total_cls_loss / batch_count,
                'trans_loss': total_trans_loss / batch_count,
                'wer': np.mean(all_metrics['wer']),
                'cer': np.mean(all_metrics['cer']),
                'disease_acc': np.mean(all_metrics['disease_acc']),
                'disease_f1': np.mean(all_metrics['disease_f1']),
                'disease_precision': np.mean(all_metrics['disease_precision']),
                'disease_recall': np.mean(all_metrics['disease_recall']),
                'alpha': self.alpha,
                'beta': self.beta
            }
            return result
        
        return {
            'loss': 0, 'cls_loss': 0, 'trans_loss': 0, 'wer': 1.0, 'cer': 1.0, 
            'disease_acc': 0, 'disease_f1': 0, 'disease_precision': 0, 'disease_recall': 0, 
            'alpha': self.alpha, 'beta': self.beta
        }

    def evaluate(self, dataloader):
        """Evaluation with detailed text examples per epoch"""
        self.model.eval()
        self.disease_classifier.eval()
        
        total_loss = 0
        total_cls_loss = 0
        total_trans_loss = 0
        
        all_metrics = {
            'wer': [], 'cer': [], 'disease_acc': [], 'disease_f1': [],
            'disease_precision': [], 'disease_recall': []
        }
        batch_count = 0
        
        # Collect ALL predictions and references for evaluation summary
        eval_pred_texts = []
        eval_ref_texts = []
        eval_disease_labels = []
        eval_disease_preds = []
        
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        total_batches = len(dataloader)
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(tqdm(dataloader, desc="Evaluating")):
                mels = batch_data['mels'].to(self.device, non_blocking=True)
                # Use correct keys from collate function
                input_tokens = batch_data['input_tokens'].to(self.device, non_blocking=True)
                target_tokens = batch_data['target_tokens'].to(self.device, non_blocking=True)
                classes = batch_data['classes'].to(self.device, non_blocking=True)
                texts = batch_data['texts']
                
                try:
                    audio_features = self.model.encoder(mels)
                    disease_logits, _ = self.classify_disease_from_audio(audio_features)
                    transcription_logits = self.model.decoder(input_tokens, audio_features)
                    
                    cls_loss = self.classification_loss(disease_logits, classes)
                    trans_loss = self.transcription_loss(
                        transcription_logits.reshape(-1, transcription_logits.size(-1)),
                        target_tokens.reshape(-1)
                    )
                    combined_loss = self.alpha * trans_loss + self.beta * cls_loss

                    # Compute metrics
                    debug_mode = (batch_idx < 2)  # Debug first few batches
                    try:
                        metrics = self.compute_metrics_simplified(
                            transcription_logits, target_tokens, disease_logits, classes, texts,
                            debug_mode=debug_mode
                        )
                        
                        # Collect predictions for evaluation summary
                        eval_pred_texts.extend(metrics['pred_texts'])
                        eval_ref_texts.extend(metrics['ref_texts'])
                        eval_disease_labels.extend(classes.cpu().numpy())
                        eval_disease_preds.extend(torch.max(disease_logits, 1)[1].cpu().numpy())
                        
                        for key in ['wer', 'cer', 'disease_acc', 'disease_f1', 'disease_precision', 'disease_recall']:
                            if key in metrics:
                                all_metrics[key].append(metrics[key])
                            
                    except Exception as metrics_error:
                        print(f"Eval metrics computation failed for batch {batch_idx}: {metrics_error}")
                        # Add default values
                        for key in all_metrics.keys():
                            if key.startswith('disease'):
                                all_metrics[key].append(0.0)
                            else:
                                all_metrics[key].append(1.0)
                    
                    total_loss += combined_loss.item()
                    total_cls_loss += cls_loss.item()
                    total_trans_loss += trans_loss.item()
                    batch_count += 1
                    
                except Exception as e:
                    print(f"Eval batch {batch_idx} failed: {e}")
                    # Add default metrics for failed batches
                    for key in all_metrics.keys():
                        if key.startswith('disease'):
                            all_metrics[key].append(0.0)
                        else:
                            all_metrics[key].append(1.0)
                    continue
        
        # Print evaluation summary with examples
        self.print_epoch_examples(
            eval_pred_texts, eval_ref_texts, 
            eval_disease_labels, eval_disease_preds,
            "VALIDATION"
        )
        
        if batch_count > 0:
            return {
                'loss': total_loss / batch_count,
                'cls_loss': total_cls_loss / batch_count,
                'trans_loss': total_trans_loss / batch_count,
                'wer': np.mean(all_metrics['wer']) if all_metrics['wer'] else 1.0,
                'cer': np.mean(all_metrics['cer']) if all_metrics['cer'] else 1.0,
                'disease_acc': np.mean(all_metrics['disease_acc']) if all_metrics['disease_acc'] else 0.0,
                'disease_f1': np.mean(all_metrics['disease_f1']) if all_metrics['disease_f1'] else 0.0,
                'disease_precision': np.mean(all_metrics['disease_precision']) if all_metrics['disease_precision'] else 0.0,
                'disease_recall': np.mean(all_metrics['disease_recall']) if all_metrics['disease_recall'] else 0.0
            }
        
        print("No valid batches in evaluation, returning default metrics")
        return {
            'loss': 0, 'cls_loss': 0, 'trans_loss': 0, 
            'wer': 1.0, 'cer': 1.0, 
            'disease_acc': 0, 'disease_f1': 0,
            'disease_precision': 0, 'disease_recall': 0
        }

    # Training display method
    def train(self, train_loader, val_loader):
        """Main training loop with metrics display"""
        print(f"\n=== Multi-Task Training Started ===")
        print(f"Model: Whisper-{self.config.model_size}")
        print(f"Device: {self.device}")
        print(f"Epochs: {self.config.epochs}")
        print(f"Architecture: Audio → [Encoder] → Features → [Disease Classifier + Decoder]")
        print(f"Format: [SOT][EN][DISEASE][TRANSCRIBE][TEXT][EOT]")
        if hasattr(self.tokenizer, 'disease_tokens'):
            print(f"Disease tokens: {list(self.tokenizer.disease_tokens.keys())}")
        
        training_history = []
        patience_counter = 0
        best_loss = float('inf')
        
        for epoch in range(self.config.epochs):
            print(f"\n{'='*50}")
            print(f"EPOCH {epoch+1}/{self.config.epochs}")
            print(f"{'='*50}")
            
            train_metrics = self.train_epoch(train_loader, epoch)

            # Training metrics display
            print(f"\n📈 TRAINING METRICS:")
            print(f"   Loss: {train_metrics['loss']:.4f} (Cls: {train_metrics['cls_loss']:.4f}, Trans: {train_metrics['trans_loss']:.4f})")
            print(f"   Disease - Acc: {train_metrics['disease_acc']:.4f}, F1: {train_metrics['disease_f1']:.4f}")
            print(f"   Disease - Prec: {train_metrics['disease_precision']:.4f}, Rec: {train_metrics['disease_recall']:.4f}")
            print(f"   Speech  - WER: {train_metrics['wer']:.4f}, CER: {train_metrics['cer']:.4f}")
            print(f"   Weights - α: {train_metrics['alpha']:.4f}, β: {train_metrics['beta']:.4f}")
            
            val_metrics = self.evaluate(val_loader)

            print(f"\n📊 VALIDATION METRICS:")
            print(f"   Loss: {val_metrics['loss']:.4f}")
            print(f"   Disease - Acc: {val_metrics['disease_acc']:.4f}, F1: {val_metrics['disease_f1']:.4f}")
            print(f"   Disease - Prec: {val_metrics['disease_precision']:.4f}, Rec: {val_metrics['disease_recall']:.4f}")
            print(f"   Speech  - WER: {val_metrics['wer']:.4f}, CER: {val_metrics['cer']:.4f}")
            
            current_loss = val_metrics["loss"]

            training_history.append({
                'epoch': epoch + 1,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'wer': val_metrics['wer'],
                'cer': val_metrics['cer'],
                'loss': val_metrics['loss'],
                'alpha': self.alpha,
                'beta': self.beta
            })

            if current_loss < best_loss:
                self.best_metric = val_metrics['disease_acc'] * 0.6 + max(0, 1 - val_metrics['wer']) * 0.4
                best_loss = current_loss
                patience_counter = 0

                # Store best weights
                best_alpha = self.alpha
                best_beta = self.beta

                checkpoint = {
                    'model_state_dict': self.model.state_dict(),
                    'disease_classifier_state_dict': self.disease_classifier.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'config': self.config,
                    'epoch': epoch,
                    'best_metric': self.best_metric,
                    'val_metrics': val_metrics,
                    'training_history': training_history,
                    'best_alpha': best_alpha,
                    'best_beta': best_beta,
                    'tokenizer_info': {
                        'eot_token': self.tokenizer.eot,
                        'sot_token': self.tokenizer.sot,
                        'disease_tokens': getattr(self.tokenizer, 'disease_tokens', {}),
                        'vocab_size': getattr(self.tokenizer.encoding, 'n_vocab', 50257)
                    }
                }
                
                save_path = os.path.join(self.config.save_dir, f'best_multitask_model_{self.config.model_size}.pt')
                torch.save(checkpoint, save_path, _use_new_zipfile_serialization=False)
                print(f"✅ Best model saved: {save_path}")
                
            else:
                patience_counter += 1
                print(f"⏳ No improvement. Patience: {patience_counter}/{self.config.early_stopping_patience}")
                
                if patience_counter >= self.config.early_stopping_patience:
                    print("🛑 Early stopping triggered!")
                    break
            
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # Save training history
        history_path = os.path.join(self.config.save_dir, f'training_history_{self.config.model_size}.json')
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=2)
        
        print(f"\n{'='*50}")
        print(f"TRAINING COMPLETE")
        print(f"{'='*50}")
        print(f"Best metric: {self.best_metric:.4f}")
        print(f"Training history saved: {history_path}")
        
        return {
            'best_metric': self.best_metric,
            'training_history': training_history
        }

    def inference_detailed(self, dataloader):
        """Inference model with comprehensive text examples"""
        self.model.eval()
        self.disease_classifier.eval()
        
        inference_results = []
        all_pred_texts = []
        all_ref_texts = []
        
        print(f"\n=== COMPREHENSIVE INFERENCE (EOT: {self.tokenizer.eot}) ===")
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(tqdm(dataloader, desc="Running Inference")):
                # Use correct keys from collate_fn
                mels = batch_data['mels'].to(self.device, non_blocking=True)
                classes = batch_data['classes'].to(self.device, non_blocking=True)
                texts = batch_data['texts']
                paths = batch_data.get('paths', [f"sample_{batch_idx}_{i}" for i in range(len(texts))])
                
                try:
                    # Get audio features for disease classification
                    audio_features = self.model.encoder(mels)
                    disease_logits, disease_preds = self.classify_disease_from_audio(audio_features)
                    
                    # Run inference with disease context
                    transcription_results = []
                    
                    for i in range(mels.shape[0]):
                        # Get predicted disease for this sample
                        predicted_disease_idx = disease_preds[i].item()
                        disease_name = self.config.class_to_disease[predicted_disease_idx]
                        
                        # Create initial tokens with disease context (like training)
                        initial_tokens = [self.tokenizer.sot]
                        
                        if not self.is_english_only:
                            initial_tokens.extend([
                                getattr(self.tokenizer, 'language_token', self.tokenizer.sot),
                                getattr(self.tokenizer, 'disease_tokens', {}).get(disease_name, self.tokenizer.eot),
                                getattr(self.tokenizer, 'transcribe', self.tokenizer.sot)
                            ])
                        else:
                            initial_tokens.append(
                                getattr(self.tokenizer, 'disease_tokens', {}).get(disease_name, self.tokenizer.eot)
                            )
                        try:
                            # Use decoder directly with disease context
                            single_mel = mels[i:i+1]
                            audio_feat = self.model.encoder(single_mel)
                            
                            # Start with disease context tokens
                            input_tokens = torch.tensor(initial_tokens, dtype=torch.long, device=self.device).unsqueeze(0)
                            
                            # Generate transcription tokens
                            max_length = 224  # Standard Whisper context length
                            generated_tokens = input_tokens.clone()
                            
                            for _ in range(max_length - len(initial_tokens)):
                                with torch.no_grad():
                                    logits = self.model.decoder(generated_tokens, audio_feat)
                                    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                                    
                                    # Stop if EOT token (using proper tokenizer EOT)
                                    if next_token.item() == self.tokenizer.eot:
                                        break
                                        
                                    generated_tokens = torch.cat([generated_tokens, next_token], dim=1)
                            
                            # Decode generated tokens
                            pred_text = self.tokenizer.decode(generated_tokens.squeeze().cpu().numpy())
                            
                        except Exception as decode_error:
                            print(f"Custom decode failed for sample {i}, using fallback: {decode_error}")
                            # Fallback: Use standard Whisper decode
                            decode_options = whisper.DecodingOptions(
                                language="en" if self.is_english_only else None,
                                without_timestamps=True,
                                task="transcribe"
                            )
                            result = self.model.decode(mels[i:i+1], decode_options)
                            pred_text = result.text if hasattr(result, 'text') else ""
                        
                        transcription_results.append(pred_text)
                    
                    # Process results
                    for i, (pred_text, class_id, original_text, path) in enumerate(zip(
                        transcription_results, classes, texts, paths)):
                        
                        # Clean and normalize text for WER/CER calculation
                        ref_text = original_text.strip()
                        
                        # Remove special token strings from both texts (using proper Whisper tokens)
                        special_token_strings = []
                        
                        # Add Whisper standard tokens
                        if hasattr(self.tokenizer, 'sot_sequence'):
                            for token_id in self.tokenizer.sot_sequence:
                                token_str = self.tokenizer.decode([token_id])
                                if token_str:
                                    special_token_strings.append(token_str)
                        
                        # Add common tokens
                        common_tokens = [
                            "<|startoftranscript|>", "<|en|>", "<|transcribe|>", 
                            "<|endoftext|>", "<|notimestamps|>", "<|nospeech|>",
                            "<|startoflm|>", "<|startofprev|>", "<|translate|>"
                        ]
                        special_token_strings.extend(common_tokens)
                        
                        # Add disease tokens
                        if hasattr(self.tokenizer, 'disease_tokens'):
                            for disease_name, token_id in self.tokenizer.disease_tokens.items():
                                token_str = self.tokenizer.decode([token_id])
                                if token_str:
                                    special_token_strings.append(token_str)
                                # Also add formatted versions
                                formatted_tokens = [f"<|{disease_name}|>", f"<|{disease_name.lower()}|>"]
                                special_token_strings.extend(formatted_tokens)
                        
                        for token_str in special_token_strings:
                            pred_text = pred_text.replace(token_str, "")
                            ref_text = ref_text.replace(token_str, "")
                        
                        # Normalize text (lowercase, clean whitespace)
                        pred_text_normalized = " ".join(pred_text.strip().split()).lower()
                        ref_text_normalized = " ".join(ref_text.strip().split()).lower()
                        
                        # Calculate individual WER and CER for this sample
                        if ref_text_normalized:
                            try:
                                sample_wer = jiwer.wer([ref_text_normalized], [pred_text_normalized])
                                sample_cer = jiwer.cer([ref_text_normalized], [pred_text_normalized])
                            except:
                                sample_wer = 1.0
                                sample_cer = 1.0
                        else:
                            sample_wer = 1.0 if pred_text_normalized else 0.0
                            sample_cer = 1.0 if pred_text_normalized else 0.0
                        
                        # Store for overall WER/CER calculation
                        if ref_text_normalized:  # Only include non-empty references
                            all_pred_texts.append(pred_text_normalized)
                            all_ref_texts.append(ref_text_normalized)
                        
                        # Get disease predictions
                        predicted_disease_idx = disease_preds[i].item()
                        true_disease_idx = class_id.item()
                        
                        disease_names = ['Normal', 'Dysarthria', 'Dysphonia']
                        predicted_disease = disease_names[predicted_disease_idx]
                        true_disease = disease_names[true_disease_idx]
                        
                        # Get confidence scores
                        disease_probs = torch.softmax(disease_logits[i], dim=0)
                        disease_confidence = disease_probs[predicted_disease_idx].item()
                        
                        sample_result = {
                            'file_path': path,
                            'original_text': ref_text,
                            'predicted_text': pred_text,
                            'original_text_normalized': ref_text_normalized,
                            'predicted_text_normalized': pred_text_normalized,
                            'wer': sample_wer,
                            'cer': sample_cer,
                            'true_disease': true_disease,
                            'predicted_disease': predicted_disease,
                            'disease_confidence': disease_confidence,
                            'disease_correct': predicted_disease_idx == true_disease_idx,
                            'all_disease_probs': {
                                'Normal': disease_probs[0].item(),
                                'Dysarthria': disease_probs[1].item(),
                                'Dysphonia': disease_probs[2].item()
                            }
                        }
                        inference_results.append(sample_result)
                    
                except Exception as e:
                    print(f"Enhanced inference batch {batch_idx} failed: {e}")
                    continue
        
        # Calculate overall statistics (same as before)
        disease_correct = sum(1 for r in inference_results if r['disease_correct'])
        disease_accuracy = disease_correct / len(inference_results) if inference_results else 0
        
        # Calculate overall WER and CER
        if all_ref_texts and all_pred_texts:
            try:
                overall_wer = jiwer.wer(all_ref_texts, all_pred_texts)
                overall_cer = jiwer.cer(all_ref_texts, all_pred_texts)
            except Exception as e:
                print(f"Error calculating overall WER/CER: {e}")
                overall_wer = sum(r['wer'] for r in inference_results) / len(inference_results)
                overall_cer = sum(r['cer'] for r in inference_results) / len(inference_results)
        else:
            overall_wer = 1.0
            overall_cer = 1.0
        
        # Calculate per-class WER/CER
        per_class_metrics = {}
        for disease in ['Normal', 'Dysarthria', 'Dysphonia']:
            disease_samples = [r for r in inference_results if r['true_disease'] == disease]
            if disease_samples:
                disease_wer = sum(r['wer'] for r in disease_samples) / len(disease_samples)
                disease_cer = sum(r['cer'] for r in disease_samples) / len(disease_samples)
                disease_acc = sum(1 for r in disease_samples if r['disease_correct']) / len(disease_samples)
                
                per_class_metrics[disease] = {
                    'samples': len(disease_samples),
                    'wer': disease_wer,
                    'cer': disease_cer,
                    'accuracy': disease_acc
                }
        
        print(f"\nComprehensive inference completed: {len(inference_results)} samples")
        print(f"Overall WER: {overall_wer:.4f}")
        print(f"Overall CER: {overall_cer:.4f}")
        print(f"Disease classification accuracy: {disease_accuracy:.4f}")
        print(f"Using EOT token: {self.tokenizer.eot}")
        
        return {
            'inference_results': inference_results,
            'summary': {
                'total_samples': len(inference_results),
                'overall_wer': overall_wer,
                'overall_cer': overall_cer,
                'disease_accuracy': disease_accuracy,
                'disease_correct': disease_correct,
                'per_class_metrics': per_class_metrics
            }
        }

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path):
        """Load trained model from checkpoint"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        except TypeError:
            # Fallback for older PyTorch versions
            print("Failed to load checkpoint with 'weights_only=False'. Trying without it.")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        config = checkpoint['config']
        
        trainer = cls(config)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.disease_classifier.load_state_dict(checkpoint['disease_classifier_state_dict'])
        
        if 'optimizer_state_dict' in checkpoint:
            trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'best_alpha' in checkpoint and 'best_beta' in checkpoint:
            trainer.alpha = checkpoint['best_alpha']
            trainer.beta = checkpoint['best_beta']
            print(f"✔ Best loss weights loaded: α={trainer.alpha:.4f}, β={trainer.beta:.4f}")
        else:
            print("No best loss weights found in checkpoint, using defaults: α=1.0, β=1.0")
            trainer.alpha = 1.0
            trainer.beta = 1.0

        trainer.model = trainer.model.to(trainer.device)
        trainer.disease_classifier = trainer.disease_classifier.to(trainer.device)
        
        print(f"✔ Multi-task model loaded from: {checkpoint_path}")
        print(f"✔ Model moved to: {trainer.device}")
        print(f"✔ Using proper Whisper tokens - EOT: {trainer.tokenizer.eot}")
        return trainer