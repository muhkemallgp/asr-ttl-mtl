import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from speech_disorder.config import TrainingConfig
from speech_disorder.dataset import MultiTaskSpeechDataset
from speech_disorder.trainer import MultiTaskTrainer
from torch.utils.data import DataLoader
import torch

def main():
    parser = argparse.ArgumentParser(description="Multi-Task Learning untuk Speech Disorder Detection")
    
    # Model arguments
    parser.add_argument("--model_size", type=str, default="tiny", 
                       choices=["tiny", "base", "small", "medium", "large","tiny.en", "base.en", "small.en", "medium.en", "large.en"],)
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--val_batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--freeze_encoder", action="store_true", default=False)

    # Data arguments
    parser.add_argument("--data_dir", type=str, default="../data")
    
    # Output arguments
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    
    args = parser.parse_args()
    
    # Set device dengan CUDA priority
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    # Data paths
    train_csv = os.path.join(args.data_dir, "custom_train.csv")
    val_csv = os.path.join(args.data_dir, "custom_val.csv")
    
    # Verify data files
    for csv_file in [train_csv, val_csv]:
        if not os.path.exists(csv_file):
            print(f"Error: Data file tidak ditemukan: {csv_file}")
            return
    
    # Create config
    config = TrainingConfig(
        model_size=args.model_size,
        batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        freeze_encoder=args.freeze_encoder,
        train_csv=train_csv,
        val_csv=val_csv,
        save_dir=args.save_dir,
        device=device
    )
    
    print(f"\n=== Multi-Task Learning Configuration ===")
    print(f"Model: Whisper-{config.model_size}")
    print(f"Device: {config.device}")
    print(f"Batch sizes: Train={config.batch_size}, Val={config.val_batch_size}")
    print(f"Epochs: {config.epochs}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Freeze encoder: {args.freeze_encoder}")
    print(f"Audio processing: librosa")
    
    # Create datasets dengan librosa
    print("\nLoading datasets...")
    train_dataset = MultiTaskSpeechDataset(config.train_csv, config)
    val_dataset = MultiTaskSpeechDataset(config.val_csv, config)
    
    #Check model multilinguality
    is_english_only = '.en' in config.model_size
    if is_english_only:
        print("Using English-only tokenizer")
    else:
        print("Using multilingual tokenizer")
        
    # Create data loaders dengan CUDA optimization
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
    
    print(f"Dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset)}")
    print(f"Batches: Train={len(train_loader)}, Val={len(val_loader)}")
    
    # Initialize trainer
    print("\nInitializing multi-task trainer...")
    trainer = MultiTaskTrainer(config)
    
    # Start training
    print("\nStarting multi-task training...")
    results = trainer.train(train_loader, val_loader)
    
    print(f"\n=== Training Results ===")
    print(f"Best metric: {results['best_metric']:.4f}")

if __name__ == "__main__":
    main()