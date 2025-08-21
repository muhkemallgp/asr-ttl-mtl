import torch
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
    early_stopping_patience: int = 10
    
    # Dataset paths
    train_csv: str = "../data/custom_train.csv"
    val_csv: str = "../data/custom_val.csv"
    test_csv: str = "../data/custom_test.csv"
    
    # Model settings
    freeze_encoder: bool = False
    save_dir: str = None
    
    # Disease classification mapping - KONSISTEN
    class_to_disease: dict = field(default_factory=lambda: {
        0: 'normal',      # Class 0 -> normal
        1: 'dysphonia',   # Class 1 -> dysphonia  
        2: 'dysarthria'   # Class 2 -> dysarthria
    })
    
    # Disease tokens untuk konsistensi
    disease_tokens: list = field(default_factory=lambda: [
        'normal', 'dysphonia', 'dysarthria'
    ])

# Global constants untuk compatibility
DISORDER_TYPE = {
    0: "Normal",
    1: "Dysphonia", 
    2: "Dysarthria"
}