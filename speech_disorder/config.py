import torch
import os
from dataclasses import dataclass, field

@dataclass
class TrainingConfig:
    # Model settings
    model_size: str = "tiny"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Multi-task training only (fixed)
    mode: str = "multi_task"
    
    # Training hyperparameters
    epochs: int = 50
    batch_size: int = 16        # Training batch size
    val_batch_size: int = 8     # Validation batch size
    learning_rate: float = 1e-5  # Lower for transfer learning
    
    # Multi-task loss weights (dynamic update)
    alpha: float = 0.0  # Classification weight
    beta: float = 0.0  # Transcription weight
    
    # Optimization settings
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    early_stopping_patience: int = 7
    
    # Audio settings (librosa)
    sample_rate: int = 16000
    n_mels: int = 80
    n_fft: int = 400
    hop_length: int = 160
    win_length: int = 400
    max_audio_length: int = 480000  # 30 seconds at 16kHz
    
    # Dataset paths
    train_csv: str = "../data/custom_train.csv"
    val_csv: str = "../data/custom_val.csv"
    test_csv: str = "../data/custom_test.csv"
    
    # Model settings
    freeze_encoder: bool = False
    save_dir: str = None
    save_every: int = 5
    
    # Class mapping untuk disease tokens
    class_to_disease: dict = field(default_factory=lambda: {
        0: 'normal',
        1: 'dysphonia', 
        2: 'dysarthria'
    })
    
    # Disease token names (sesuai tokenizer)
    disease_tokens: list = field(default_factory=lambda: [
        'normal', 'dysphonia', 'dysarthria'
    ])

    def __post_init__(self):
        """Setup save directory and validate config"""
        if self.save_dir is None:
            self.save_dir = f"../checkpoints_exp/checkpoints_{self.model_size}"
        
        # Create checkpoint directory
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Ensure CUDA compatibility
        if self.device == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA not available, falling back to CPU")
            self.device = "cpu"
        
        print(f"Multi-task Training Configuration:")
        print(f"  Model: Whisper-{self.model_size}")
        print(f"  Mode: {self.mode}")
        print(f"  Device: {self.device}")
        print(f"  Batch sizes: Train={self.batch_size}, Val={self.val_batch_size}")
        print(f"  Loss weights: α={self.alpha}, β={self.beta}")
        print(f"  Audio: librosa (sr={self.sample_rate}, n_mels={self.n_mels})")
        print(f"  Save directory: {self.save_dir}")
        print(f"  Disease tokens: {self.disease_tokens}")

# Global constants untuk compatibility
DISORDER_TYPE = {
    0: "Normal",
    1: "Dysphonia",
    2: "Dysarthria"
}