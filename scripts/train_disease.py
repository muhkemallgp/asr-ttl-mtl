import argparse
import os
import sys
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from speech_disorder.config import TrainingConfig
from speech_disorder.dataset import MultiTaskSpeechDataset
from speech_disorder.trainer import MultiTaskTrainer
from torch.utils.data import DataLoader
import torch

def main():
    parser = argparse.ArgumentParser(description="Multi-Task Learning for Speech Disorder Detection")
    
    # Model arguments
    parser.add_argument("--model_size", type=str, default="tiny", 
                       choices=["tiny", "base", "small", "medium", "large", "tiny.en", "base.en", "small.en", "medium.en", "large.en"],
                       help="Whisper model size")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--val_batch_size", type=int, default=8, help="Validation batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--freeze_encoder", action="store_true", default=False, help="Freeze encoder weights")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--gradient_clip_norm", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--early_stopping_patience", type=int, default=10, help="Early stopping patience")
    
    # Multi-task loss weights
    parser.add_argument("--alpha", type=float, default=0.0, help="Classification loss weight (0 for dynamic)")
    parser.add_argument("--beta", type=float, default=0.0, help="Transcription loss weight (0 for dynamic)")

    # Data arguments
    parser.add_argument("--data_dir", type=str, default="../data", help="Data directory")
    parser.add_argument("--train_file", type=str, default="custom_train.csv", help="Training CSV file")
    parser.add_argument("--val_file", type=str, default="custom_val.csv", help="Validation CSV file")
    
    # Output arguments
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (auto/cpu/cuda)")
    
    args = parser.parse_args()
    
    # Set device with CUDA priority
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    # Resolve data paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, args.data_dir)
    data_dir = os.path.normpath(data_dir)
    
    train_csv = os.path.join(data_dir, args.train_file)
    val_csv = os.path.join(data_dir, args.val_file)
    
    # Verify data files exist
    for csv_file, name in [(train_csv, "Training"), (val_csv, "Validation")]:
        if not os.path.exists(csv_file):
            print(f"Error: {name} file not found: {csv_file}")
            return
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Create config with all parameters
    config = TrainingConfig(
        # Model settings
        model_size=args.model_size,
        device=device,
        
        # Training hyperparameters
        epochs=args.epochs,
        batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip_norm=args.gradient_clip_norm,
        early_stopping_patience=args.early_stopping_patience,
        
        # Multi-task loss weights
        alpha=args.alpha,
        beta=args.beta,
        
        # Model settings
        freeze_encoder=args.freeze_encoder,
        
        # Dataset paths
        train_csv=train_csv,
        val_csv=val_csv,
        
        # Output
        save_dir=args.save_dir
    )
    
    print(f"\n{'='*80}")
    print(f"{'MULTI-TASK LEARNING CONFIGURATION':^80}")
    print(f"{'='*80}")
    print(f"Model: Whisper-{config.model_size}")
    print(f"Model Type: {'English-only' if '.en' in config.model_size else 'Multilingual'}")
    print(f"Device: {config.device}")
    print(f"Architecture: Shared Encoder + Disease Classifier + Transcription Decoder")
    
    print(f"\nTraining Configuration:")
    print(f"  Epochs: {config.epochs}")
    print(f"  Batch sizes: Train={config.batch_size}, Val={config.val_batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Weight decay: {config.weight_decay}")
    print(f"  Gradient clip norm: {config.gradient_clip_norm}")
    print(f"  Early stopping patience: {config.early_stopping_patience}")
    print(f"  Freeze encoder: {config.freeze_encoder}")
    
    print(f"\nMulti-task Loss Configuration:")
    if config.alpha > 0 and config.beta > 0:
        print(f"  Static weights - α (classification): {config.alpha}, β (transcription): {config.beta}")
    else:
        print(f"  Dynamic weights - Adaptive based on loss magnitudes")
    
    print(f"\nData Configuration:")
    print(f"  Training data: {train_csv}")
    print(f"  Validation data: {val_csv}")
    print(f"  Audio processing: Whisper built-in (librosa backend)")
    print(f"  Save directory: {args.save_dir}")
    
    # Load and inspect datasets
    print(f"\n{'='*80}")
    print(f"{'LOADING DATASETS':^80}")
    print(f"{'='*80}")
    
    print("Loading training dataset...")
    train_dataset = MultiTaskSpeechDataset(config.train_csv, config)
    
    print("Loading validation dataset...")
    val_dataset = MultiTaskSpeechDataset(config.val_csv, config)
    
    print(f"\nDataset Summary:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    
    # Create data loaders with optimization for CUDA
    print("Creating data loaders...")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        collate_fn=train_dataset.get_collate_fn(),
        num_workers=2,
        pin_memory=True if device == "cuda" else False,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.val_batch_size, 
        shuffle=False, 
        collate_fn=val_dataset.get_collate_fn(),
        num_workers=2,
        pin_memory=True if device == "cuda" else False,
        persistent_workers=True
    )
    
    print(f"Data loaders created:")
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    
    # Initialize trainer
    print(f"\n{'='*80}")
    print(f"{'INITIALIZING MULTI-TASK TRAINER':^80}")
    print(f"{'='*80}")
    
    trainer = MultiTaskTrainer(config)
    
    # Print trainer info (optional but helpful)
    print(f"✓ Trainer initialized:")
    print(f"  Model parameters: {sum(p.numel() for p in trainer.model.parameters()):,}")
    print(f"  Trainable parameters: {sum(p.numel() for p in trainer.model.parameters() if p.requires_grad):,}")
    print(f"  Disease classification: Embedded in sequence (no separate classifier)")
    print(f"  Optimizer: {type(trainer.optimizer).__name__}")
    print(f"  Loss functions: CrossEntropyLoss (both tasks)")
    
    # Start training
    print(f"\n{'='*80}")
    print(f"{'STARTING MULTI-TASK TRAINING':^80}")
    print(f"{'='*80}")
    
    try:
        results = trainer.train(train_loader, val_loader)
        
        print(f"\n{'='*80}")
        print(f"{'TRAINING COMPLETED SUCCESSFULLY':^80}")
        print(f"{'='*80}")
        print(f"Best validation loss: {results['best_loss']:.4f}")
        
        # Save final training summary
        final_checkpoint_path = os.path.join(args.save_dir, f'best_multitask_model_{config.model_size}.pt')
        print(f"Best model saved at: {final_checkpoint_path}")
        
        if 'training_history' in results:
            history_path = os.path.join(args.save_dir, f'comprehensive_training_history_{config.model_size}.json')
            print(f"Training history saved at: {history_path}")
        
        # Print final summary
        print(f"\nFinal Training Summary:")
        print(f"  Total epochs trained: {len(results.get('training_history', []))}")
        print(f"  Best validation loss: {results['best_loss']:.4f}")
        print(f"  Model architecture: Multi-task Whisper with disease classification")
        print(f"  Save directory: {args.save_dir}")
        
        # Save training configuration for reproducibility
        config_save_path = os.path.join(args.save_dir, f'training_config_{config.model_size}.json')
        config_dict = {
            'model_size': config.model_size,
            'device': str(config.device),
            'epochs': config.epochs,
            'batch_size': config.batch_size,
            'val_batch_size': config.val_batch_size,
            'learning_rate': config.learning_rate,
            'weight_decay': config.weight_decay,
            'gradient_clip_norm': config.gradient_clip_norm,
            'early_stopping_patience': config.early_stopping_patience,
            'alpha': config.alpha,
            'beta': config.beta,
            'freeze_encoder': config.freeze_encoder,
            'train_csv': config.train_csv,
            'val_csv': config.val_csv,
            'save_dir': config.save_dir,
            'class_to_disease': config.class_to_disease,
            'disease_tokens': config.disease_tokens
        }
        
        with open(config_save_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        print(f"Training configuration saved: {config_save_path}")
        
    except KeyboardInterrupt:
        print(f"\n{'='*80}")
        print(f"{'TRAINING INTERRUPTED BY USER':^80}")
        print(f"{'='*80}")
        print("Training was stopped by user (Ctrl+C)")
        print("Partial model may have been saved if any validation improvement occurred.")
        
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"{'TRAINING FAILED':^80}")
        print(f"{'='*80}")
        print(f"Error occurred during training: {str(e)}")
        print("\nFull traceback:")
        import traceback
        traceback.print_exc()
        
        # Try to save debug info
        try:
            debug_path = os.path.join(args.save_dir, 'debug_info.txt')
            with open(debug_path, 'w') as f:
                f.write("TRAINING FAILED\n")
                f.write("="*50 + "\n")
                f.write(f"Error: {str(e)}\n\n")
                f.write("Full traceback:\n")
                traceback.print_exc(file=f)
                f.write(f"\nConfiguration:\n")
                f.write(f"  Model size: {config.model_size}\n")
                f.write(f"  Device: {config.device}\n")
                f.write(f"  Batch size: {config.batch_size}\n")
                f.write(f"  Train samples: {len(train_dataset) if 'train_dataset' in locals() else 'Unknown'}\n")
                f.write(f"  Val samples: {len(val_dataset) if 'val_dataset' in locals() else 'Unknown'}\n")
            print(f"Debug information saved: {debug_path}")
        except:
            print("Could not save debug information")
            
        return

    print(f"\n{'='*80}")
    print(f"{'MULTI-TASK TRAINING FINISHED':^80}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()