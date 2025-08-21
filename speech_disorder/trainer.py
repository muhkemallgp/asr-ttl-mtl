import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, classification_report, precision_recall_fscore_support, confusion_matrix
import os
from tqdm import tqdm
import jiwer
import numpy as np
import json
import whisper

from .config import TrainingConfig
from whisper import load_model
from whisper.tokenizer import get_tokenizer

class MultiTaskTrainer:
    """Multi-Task Learning Trainer dengan proper transcription dan disease classification"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        print(f"=== Multi-Task Learning Trainer (Proper Architecture) ===")
        print(f"Device: {self.device}")
        
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            print(f"CUDA device: {torch.cuda.get_device_name()}")
            print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Load Whisper model on CPU first
        self.model = load_model(config.model_size, device='cpu')
        print(f"✔ Whisper model '{config.model_size}' loaded")
        
        # Check model type
        self.is_english_only = '.en' in config.model_size
        
        # Get tokenizer dengan disease support
        if self.is_english_only:
            self.tokenizer = get_tokenizer(
                multilingual=False,
                include_diseases=True
            )
        else:
            self.tokenizer = get_tokenizer(
                multilingual=True,
                language="en",
                task="transcribe", 
                include_diseases=True
            )
        
        print(f"✔ Tokenizer loaded:")
        print(f"  Type: {'English-only' if self.is_english_only else 'Multilingual'}")
        print(f"  EOT: {self.tokenizer.eot}, SOT: {self.tokenizer.sot}")
        print(f"  Disease tokens: {self.tokenizer.disease_tokens}")
        
        # Class to disease mapping
        self.class_to_disease = config.class_to_disease
        self.disease_to_class = {v: k for k, v in config.class_to_disease.items()}
        
        # Disease token IDs for classification
        self.disease_token_ids = {
            disease: self.tokenizer.disease_tokens[disease] 
            for disease in ['normal', 'dysphonia', 'dysarthria']
            if disease in self.tokenizer.disease_tokens
        }
        
        print(f"✔ Disease token mapping: {self.disease_token_ids}")
        
        # Token positions in dataset format
        if self.is_english_only:
            self.disease_token_position = 1
        else:
            self.disease_token_position = 2
        
        print(f"✔ Disease token position in dataset: {self.disease_token_position}")
        
        # EXPAND VOCABULARY dengan proper approach
        self._expand_vocabulary()
        
        # Add disease classifier head
        self._setup_disease_classifier()
        
        # Move model to device after vocabulary expansion
        self.model = self.model.to(self.device)
        self.disease_classifier = self.disease_classifier.to(self.device)
        
        # Setup optimizer
        self._setup_optimizer()
        
        # Loss functions
        self.transcription_loss = nn.CrossEntropyLoss(ignore_index=-100)
        self.classification_loss = nn.CrossEntropyLoss()
        
        # Multi-task weights
        self.alpha = config.alpha
        self.beta = config.beta

        print(f"✔ Trainer initialized successfully")
        print(f"  Disease classes: {self.class_to_disease}")
        print(f"  Architecture: Shared Encoder + Disease Classifier + Transcription Decoder")

    def _expand_vocabulary(self):
        """Expand vocabulary dengan disease tokens"""
        original_vocab_size = self.model.dims.n_vocab
        
        disease_tokens = len(self.tokenizer.disease_tokens)
        if disease_tokens == 0:
            print("⚠️ No disease tokens found")
            return
        
        max_token_id = max(self.tokenizer.special_tokens.values()) if self.tokenizer.special_tokens else original_vocab_size
        new_vocab_size = max_token_id + 1
        
        print(f"Vocabulary expansion:")
        print(f"  Original size: {original_vocab_size}")
        print(f"  New size needed: {new_vocab_size}")
        print(f"  Disease tokens: {disease_tokens}")
        
        if new_vocab_size > original_vocab_size:
            self.model.resize_token_embeddings(new_vocab_size)
            print(f"✔ Vocabulary expanded: {original_vocab_size} → {new_vocab_size}")
        else:
            print("✔ No vocabulary expansion needed")

    def _setup_disease_classifier(self):
        """Setup disease classifier head"""
        d_model = self.model.dims.n_audio_state
        
        self.disease_classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 3)  # 3 disease classes
        )
        
        print(f"✔ Disease classifier created: {d_model} → {d_model//2} → 3")

    def _setup_optimizer(self):
        """Setup optimizer"""
        param_groups = []
        
        # 1. Encoder
        if not self.config.freeze_encoder:
            encoder_params = [p for p in self.model.encoder.parameters() if p.requires_grad]
            if encoder_params:
                param_groups.append({
                    'params': encoder_params,
                    'lr': self.config.learning_rate * 0.1,
                    'weight_decay': self.config.weight_decay,
                    'name': 'encoder'
                })
        else:
            for param in self.model.encoder.parameters():
                param.requires_grad = False
            print("✔ Encoder frozen")
        
        # 2. Decoder layers
        decoder_params = []
        for name, param in self.model.decoder.named_parameters():
            if param.requires_grad and 'token_embedding' not in name:
                decoder_params.append(param)
        
        if decoder_params:
            param_groups.append({
                'params': decoder_params,
                'lr': self.config.learning_rate * 0.3,
                'weight_decay': self.config.weight_decay,
                'name': 'decoder_layers'
            })
        
        # 3. Token embeddings
        if hasattr(self.model.decoder, 'token_embedding'):
            embedding_weight = self.model.decoder.token_embedding.weight
            param_groups.append({
                'params': [embedding_weight],
                'lr': self.config.learning_rate,
                'weight_decay': 0.0,
                'name': 'embeddings'
            })
        
        # 4. Disease classifier
        param_groups.append({
            'params': self.disease_classifier.parameters(),
            'lr': self.config.learning_rate,
            'weight_decay': self.config.weight_decay,
            'name': 'disease_classifier'
        })
        
        self.optimizer = torch.optim.AdamW(param_groups)
        
        total_params = 0
        for i, group in enumerate(param_groups):
            group_params = sum(p.numel() for p in group['params'])
            total_params += group_params
            print(f"  Group {i+1} ({group.get('name', 'unnamed')}): {group_params:,} params, LR: {group['lr']:.2e}")
        
        print(f"✔ Optimizer setup: {len(param_groups)} groups, {total_params:,} total params")

    def classify_disease_from_audio(self, audio_features):
        """Classify disease dari audio features"""
        audio_repr = audio_features.mean(dim=1)
        disease_logits = self.disease_classifier(audio_repr)
        disease_predictions = torch.argmax(disease_logits, dim=-1)
        return disease_logits, disease_predictions

    def compute_alpha_beta(self, epoch, trans_loss, class_loss):
        """Compute adaptive alpha and beta based on loss values"""
        if epoch >= 0:
            trans_loss_value = trans_loss.item() + 1e-6
            class_loss_value = class_loss.item() + 1e-6
            alpha = 1 / class_loss_value
            beta = 1 / trans_loss_value

            total_weight = alpha + beta
            alpha /= total_weight
            beta /= total_weight
            return alpha, beta
        else:
            return self.alpha, self.beta

    def decode_predictions(self, logits):
        """Decode logits to text predictions"""
        pred_tokens = torch.argmax(logits, dim=-1)
        pred_texts = []
        
        for tokens in pred_tokens:
            tokens_np = tokens.cpu().numpy()
            valid_tokens = tokens_np[tokens_np != -100]
            text = self.tokenizer.decode(valid_tokens)
            
            special_tokens = [
                "<|startoftranscript|>", "<|endoftext|>", "<|en|>", 
                "<|transcribe|>", "<|notimestamps|>", "<|nospeech|>",
                "<|normal|>", "<|dysphonia|>", "<|dysarthria|>"
            ]
            
            for token in special_tokens:
                text = text.replace(token, "")
            
            text = text.strip()
            pred_texts.append(text)
        
        return pred_texts

    def compute_detailed_metrics(self, all_pred_texts, all_ref_texts, all_disease_predictions, all_disease_labels):
        """Compute detailed metrics including per-class analysis"""
        metrics = {}
        
        # Overall disease classification metrics
        if len(all_disease_predictions) > 0:
            disease_acc = accuracy_score(all_disease_labels, all_disease_predictions)
            weighted_f1 = f1_score(all_disease_labels, all_disease_predictions, average='weighted', zero_division=0)
            macro_f1 = f1_score(all_disease_labels, all_disease_predictions, average='macro', zero_division=0)
            
            weighted_precision, weighted_recall, _, _ = precision_recall_fscore_support(
                all_disease_labels, all_disease_predictions, average='weighted', zero_division=0
            )
            macro_precision, macro_recall, _, _ = precision_recall_fscore_support(
                all_disease_labels, all_disease_predictions, average='macro', zero_division=0
            )
            
            per_class_precision, per_class_recall, per_class_f1, support = precision_recall_fscore_support(
                all_disease_labels, all_disease_predictions, average=None, zero_division=0
            )
            
            metrics.update({
                'disease_acc': disease_acc,
                'weighted_f1': weighted_f1,
                'macro_f1': macro_f1,
                'weighted_precision': weighted_precision,
                'weighted_recall': weighted_recall,
                'macro_precision': macro_precision,
                'macro_recall': macro_recall,
                'per_class_precision': per_class_precision.tolist(),
                'per_class_recall': per_class_recall.tolist(),
                'per_class_f1': per_class_f1.tolist(),
                'per_class_support': support.tolist()
            })
        
        # Overall transcription metrics
        if len(all_pred_texts) > 0 and len(all_ref_texts) > 0:
            valid_pairs = [(p.lower(), r.lower()) for p, r in zip(all_pred_texts, all_ref_texts) if r.strip()]
            if valid_pairs:
                pred_valid, ref_valid = zip(*valid_pairs)
                overall_wer = jiwer.wer(list(ref_valid), list(pred_valid))
                overall_cer = jiwer.cer(list(ref_valid), list(pred_valid))
                metrics.update({'wer': overall_wer, 'cer': overall_cer})
        
        # Per-class transcription metrics
        per_class_transcription = {}
        disease_names = ['normal', 'dysphonia', 'dysarthria']
        
        if len(all_pred_texts) == len(all_ref_texts) == len(all_disease_labels) and len(all_pred_texts) > 0:
            for class_id, disease_name in enumerate(disease_names):
                class_indices = [i for i, label in enumerate(all_disease_labels) if label == class_id]
                
                if len(class_indices) > 0:
                    class_pred_texts = [all_pred_texts[i] for i in class_indices]
                    class_ref_texts = [all_ref_texts[i] for i in class_indices]
                    
                    class_valid_pairs = [(p.lower(), r.lower()) for p, r in zip(class_pred_texts, class_ref_texts) if r.strip()]
                    
                    if len(class_valid_pairs) > 0:
                        class_pred_valid, class_ref_valid = zip(*class_valid_pairs)
                        class_wer = jiwer.wer(list(class_ref_valid), list(class_pred_valid))
                        class_cer = jiwer.cer(list(class_ref_valid), list(class_pred_valid))
                        
                        per_class_transcription[disease_name] = {
                            'wer': class_wer,
                            'cer': class_cer,
                            'samples': len(class_indices),
                            'valid_samples': len(class_valid_pairs)
                        }
                    else:
                        per_class_transcription[disease_name] = {
                            'wer': 1.0,
                            'cer': 1.0,
                            'samples': len(class_indices),
                            'valid_samples': 0
                        }
        
        metrics['per_class_transcription'] = per_class_transcription
        return metrics

    def print_detailed_metrics(self, metrics, phase="Training"):
        """Print detailed metrics including per-class breakdown"""
        print(f"\n📊 {phase.upper()} DETAILED METRICS:")
        
        if 'disease_acc' in metrics:
            print(f"   🏥 Disease Classification:")
            print(f"      Overall Accuracy: {metrics['disease_acc']:.4f}")
            print(f"      Weighted Precision: {metrics.get('weighted_precision', 0.0):.4f}")
            print(f"      Weighted Recall: {metrics.get('weighted_recall', 0.0):.4f}")
            print(f"      Weighted F1: {metrics['weighted_f1']:.4f}")
            print(f"      Macro Precision: {metrics.get('macro_precision', 0.0):.4f}")
            print(f"      Macro Recall: {metrics.get('macro_recall', 0.0):.4f}")
            print(f"      Macro F1: {metrics['macro_f1']:.4f}")
        
        if 'wer' in metrics:
            print(f"   🎯 Transcription:")
            print(f"      Overall WER: {metrics['wer']:.4f}")
            print(f"      Overall CER: {metrics['cer']:.4f}")
        
        if 'per_class_precision' in metrics:
            disease_names = ['Normal', 'Dysphonia', 'Dysarthria']
            print(f"   📈 Per-Class Disease Classification:")
            print(f"      {'Class':<12} {'Prec':<6} {'Rec':<6} {'F1':<6} {'Support':<8}")
            print(f"      {'-'*40}")
            
            for i, disease in enumerate(disease_names):
                if i < len(metrics['per_class_precision']):
                    precision = metrics['per_class_precision'][i]
                    recall = metrics['per_class_recall'][i]
                    f1 = metrics['per_class_f1'][i]
                    support = metrics['per_class_support'][i]
                    print(f"      {disease:<12} {precision:<5.3f} {recall:<5.3f} {f1:<5.3f} {support:<8}")
        
        if 'per_class_transcription' in metrics and metrics['per_class_transcription']:
            print(f"   📝 Per-Class Transcription:")
            print(f"      {'Class':<12} {'WER':<6} {'CER':<6} {'Samples':<8} {'Valid':<8}")
            print(f"      {'-'*48}")
            
            for disease, trans_metrics in metrics['per_class_transcription'].items():
                wer = trans_metrics.get('wer', 1.0)
                cer = trans_metrics.get('cer', 1.0)
                samples = trans_metrics.get('samples', 0)
                valid_samples = trans_metrics.get('valid_samples', 0)
                
                print(f"      {disease.capitalize():<12} "
                      f"{wer:<5.3f} "
                      f"{cer:<5.3f} "
                      f"{samples:<8} "
                      f"{valid_samples:<8}")

    def train_epoch(self, dataloader, epoch):
        """Training epoch dengan complete data collection"""
        self.model.train()
        self.disease_classifier.train()
        
        total_loss = 0
        total_cls_loss = 0
        total_trans_loss = 0
        
        all_disease_predictions = []
        all_disease_labels = []
        all_pred_texts = []
        all_ref_texts = []
        
        for batch_idx, batch_data in enumerate(tqdm(dataloader, desc=f"Training Epoch {epoch+1}")):
            self.optimizer.zero_grad()
            
            mels = batch_data['mels'].to(self.device, non_blocking=True)
            input_tokens = batch_data['input_tokens'].to(self.device, non_blocking=True)
            target_tokens = batch_data['target_tokens'].to(self.device, non_blocking=True)
            classes = batch_data['classes'].to(self.device, non_blocking=True)
            texts = batch_data['texts']
            
            # Forward pass
            audio_features = self.model.encoder(mels)
            disease_logits, disease_preds = self.classify_disease_from_audio(audio_features)
            transcription_logits = self.model.decoder(input_tokens, audio_features)
            
            # Compute losses
            cls_loss = self.classification_loss(disease_logits, classes)
            trans_loss = self.transcription_loss(
                transcription_logits.reshape(-1, transcription_logits.size(-1)),
                target_tokens.reshape(-1)
            )
            
            # Update alpha beta if dynamic
            if self.alpha == 0.0 or self.beta == 0.0:
                self.alpha, self.beta = self.compute_alpha_beta(epoch, trans_loss, cls_loss)
            
            combined_loss = self.alpha * cls_loss + self.beta * trans_loss
            
            # Backward pass
            combined_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters()) + list(self.disease_classifier.parameters()),
                max_norm=self.config.gradient_clip_norm
            )
            
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += combined_loss.item()
            total_cls_loss += cls_loss.item()
            total_trans_loss += trans_loss.item()
            
            # Collect all predictions
            all_disease_predictions.extend(disease_preds.cpu().numpy())
            all_disease_labels.extend(classes.cpu().numpy())
            
            pred_texts = self.decode_predictions(transcription_logits)
            all_pred_texts.extend(pred_texts)
            all_ref_texts.extend(texts)
        
        # Compute detailed metrics
        detailed_metrics = self.compute_detailed_metrics(
            all_pred_texts, all_ref_texts, all_disease_predictions, all_disease_labels
        )
        
        # Print summary
        num_batches = len(dataloader)
        avg_loss = total_loss / num_batches
        avg_cls_loss = total_cls_loss / num_batches
        avg_trans_loss = total_trans_loss / num_batches
        
        print(f"\n📈 TRAINING EPOCH {epoch+1} SUMMARY:")
        print(f"   Loss: {avg_loss:.4f} (α·Cls: {avg_cls_loss:.4f}, β·Trans: {avg_trans_loss:.4f})")
        print(f"   Weights: α={self.alpha:.4f}, β={self.beta:.4f}")
        
        if 'disease_acc' in detailed_metrics:
            print(f"   Disease: Acc={detailed_metrics['disease_acc']:.4f}, "
                  f"W-P={detailed_metrics.get('weighted_precision', 0.0):.4f}, "
                  f"W-R={detailed_metrics.get('weighted_recall', 0.0):.4f}, "
                  f"W-F1={detailed_metrics['weighted_f1']:.4f}")
        
        if 'wer' in detailed_metrics:
            print(f"   Speech: WER={detailed_metrics['wer']:.4f}, CER={detailed_metrics['cer']:.4f}")
        
        base_metrics = {
            'loss': avg_loss,
            'cls_loss': avg_cls_loss,
            'trans_loss': avg_trans_loss,
            'alpha': self.alpha,
            'beta': self.beta
        }
        base_metrics.update(detailed_metrics)
        return base_metrics

    def evaluate(self, dataloader):
        """Evaluation dengan complete data collection"""
        self.model.eval()
        self.disease_classifier.eval()
        
        total_loss = 0
        total_cls_loss = 0
        total_trans_loss = 0
        
        all_disease_predictions = []
        all_disease_labels = []
        all_pred_texts = []
        all_ref_texts = []
        
        with torch.no_grad():
            for batch_data in tqdm(dataloader, desc="Evaluating"):
                mels = batch_data['mels'].to(self.device, non_blocking=True)
                input_tokens = batch_data['input_tokens'].to(self.device, non_blocking=True)
                target_tokens = batch_data['target_tokens'].to(self.device, non_blocking=True)
                classes = batch_data['classes'].to(self.device, non_blocking=True)
                texts = batch_data['texts']
                
                # Forward pass
                audio_features = self.model.encoder(mels)
                disease_logits, disease_preds = self.classify_disease_from_audio(audio_features)
                transcription_logits = self.model.decoder(input_tokens, audio_features)
                
                # Losses
                cls_loss = self.classification_loss(disease_logits, classes)
                trans_loss = self.transcription_loss(
                    transcription_logits.reshape(-1, transcription_logits.size(-1)),
                    target_tokens.reshape(-1)
                )
                combined_loss = self.alpha * cls_loss + self.beta * trans_loss
                
                total_loss += combined_loss.item()
                total_cls_loss += cls_loss.item()
                total_trans_loss += trans_loss.item()
                
                # Collect all predictions
                all_disease_predictions.extend(disease_preds.cpu().numpy())
                all_disease_labels.extend(classes.cpu().numpy())
                
                pred_texts = self.decode_predictions(transcription_logits)
                all_pred_texts.extend(pred_texts)
                all_ref_texts.extend(texts)
        
        # Compute detailed metrics
        detailed_metrics = self.compute_detailed_metrics(
            all_pred_texts, all_ref_texts, all_disease_predictions, all_disease_labels
        )
        
        # Print summary
        num_batches = len(dataloader)
        print(f"\n📊 VALIDATION SUMMARY:")
        
        # Print detailed metrics for validation
        self.print_detailed_metrics(detailed_metrics, "Validation")
        
        base_metrics = {
            'loss': total_loss / num_batches,
            'cls_loss': total_cls_loss / num_batches,
            'trans_loss': total_trans_loss / num_batches
        }
        base_metrics.update(detailed_metrics)
        return base_metrics

    def train(self, train_loader, val_loader):
        """Main training loop"""
        print(f"\n=== Multi-Task Training Started ===")
        print(f"Architecture: Shared Encoder + Disease Classifier + Transcription Decoder")
        print(f"Disease classes: {list(self.class_to_disease.values())}")
        
        best_loss = float('inf')
        patience_counter = 0
        training_history = []
        
        for epoch in range(self.config.epochs):
            print(f"\n{'='*60}")
            print(f"EPOCH {epoch+1}/{self.config.epochs}")
            print(f"{'='*60}")
            
            # Training
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validation
            val_metrics = self.evaluate(val_loader)
            
            # Save best model
            current_loss = val_metrics['loss']
            if current_loss < best_loss:
                best_loss = current_loss
                patience_counter = 0
                
                checkpoint = {
                    'model_state_dict': self.model.state_dict(),
                    'disease_classifier_state_dict': self.disease_classifier.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'config': self.config,
                    'epoch': epoch,
                    'best_loss': best_loss,
                    'val_metrics': val_metrics,
                    'train_metrics': train_metrics,
                    'alpha': self.alpha,
                    'beta': self.beta,
                    'tokenizer_info': {
                        'eot_token': self.tokenizer.eot,
                        'sot_token': self.tokenizer.sot,
                        'disease_tokens': self.tokenizer.disease_tokens,
                        'disease_token_ids': self.disease_token_ids,
                        'disease_token_position': self.disease_token_position
                    }
                }
                
                save_path = os.path.join(
                    self.config.save_dir or '.', 
                    f'best_multitask_model_{self.config.model_size}.pt'
                )
                os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
                torch.save(checkpoint, save_path)
                print(f"✅ Best model saved: {save_path}")
            else:
                patience_counter += 1
                print(f"⏳ No improvement. Patience: {patience_counter}/{self.config.early_stopping_patience}")
                
                if patience_counter >= self.config.early_stopping_patience:
                    print("🛑 Early stopping triggered!")
                    break
            
            training_history.append({
                'epoch': epoch + 1,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics
            })
        
        print(f"\n🎯 Training completed!")
        print(f"Best validation loss: {best_loss:.4f}")
        
        return {'best_loss': best_loss, 'training_history': training_history}

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path):
        """Load trained model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        config = checkpoint['config']
        
        trainer = cls(config)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.disease_classifier.load_state_dict(checkpoint['disease_classifier_state_dict'])
        
        trainer.alpha = checkpoint.get('alpha', 0.5)
        trainer.beta = checkpoint.get('beta', 0.5)
        
        print(f"✔ Model loaded from: {checkpoint_path}")
        print(f"✔ Architecture: Shared Encoder + Disease Classifier + Transcription Decoder")
        return trainer