import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset

# Import fungsi bawaan Whisper
from whisper.audio import load_audio, pad_or_trim, log_mel_spectrogram
from whisper.tokenizer import get_tokenizer

def check_tokenizer_eot_values():
    """Utility function to check EOT token values for different tokenizer types"""
    print("Checking EOT token values for different Whisper tokenizers:")
    
    # English-only tokenizer
    try:
        en_tokenizer = get_tokenizer(multilingual=False, language=None, task=None)
        print(f"English-only tokenizer EOT: {en_tokenizer.eot}")
    except:
        print("English-only tokenizer: Could not load")
    
    # Multilingual tokenizer
    try:
        multi_tokenizer = get_tokenizer(multilingual=True, language="en", task="transcribe")
        print(f"Multilingual tokenizer EOT: {multi_tokenizer.eot}")
    except:
        print("Multilingual tokenizer: Could not load")
    
    return

class MultiTaskSpeechDataset(Dataset):
    """Dataset untuk multi-task learning dengan Whisper's built-in audio processing"""
    
    def __init__(self, csv_file, config):
        self.df = pd.read_csv(csv_file)
        self.config = config
        
        # DETECT MODEL TYPE
        self.is_english_only = '.en' in getattr(config, 'model_size', '')
        
        # CONDITIONAL TOKENIZER dengan disease support
        if self.is_english_only:
            self.tokenizer = get_tokenizer(
                multilingual=False,
                include_diseases=True
            )
            print("Dataset using English-only tokenizer")
        else:
            self.tokenizer = get_tokenizer(
                multilingual=True,
                language="en",
                task="transcribe",
                include_diseases=True
            )
            print("Dataset using Multilingual tokenizer")
        
        # Disease mapping dari config
        self.disease_mapping = config.class_to_disease
        
        print(f"✓ Multi-task Dataset loaded: {len(self.df)} samples")
        print(f"✓ Model type: {'English-only' if self.is_english_only else 'Multilingual'}")
        print(f"✓ EOT token: {self.tokenizer.eot}")
        print(f"✓ SOT token: {self.tokenizer.sot}")
        print(f"✓ Disease tokens available: {list(self.tokenizer.disease_tokens.keys())}")
        
        # Print class distribution
        class_counts = self.df['class'].value_counts().sort_index()
        print("✓ Class distribution:")
        for class_id, count in class_counts.items():
            disease_name = self.disease_mapping.get(class_id, f"unknown_{class_id}")
            print(f"   {disease_name} (class {class_id}): {count} samples ({count/len(self.df)*100:.1f}%)")

    def __len__(self):
        return len(self.df)
    
    def load_and_process_audio(self, audio_path):
        """
        Load dan process audio menggunakan fungsi bawaan Whisper
        Returns mel spectrogram yang compatible dengan Whisper model
        """
        try:
            # Load audio menggunakan fungsi Whisper - otomatis 16kHz mono
            audio = load_audio(audio_path)
            
            # Pad atau trim ke 30 detik (480000 samples) - default Whisper
            audio = pad_or_trim(audio)
            
            # Convert ke mel spectrogram menggunakan fungsi Whisper
            # Ini akan return tensor dengan shape (80, 3000) yang compatible
            mel = log_mel_spectrogram(audio)
            
            return mel
            
        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}")
            # Return zero mel spectrogram dengan shape yang benar (80, 3000)
            return torch.zeros((80, 3000), dtype=torch.float32)
    
    def get_disease_token_id(self, class_id):
        """Get disease token ID berdasarkan class mapping"""
        disease_name = self.disease_mapping.get(class_id, 'normal')
        return self.tokenizer.disease_tokens.get(disease_name, self.tokenizer.eot)
    
    def create_sequence_with_disease_context(self, text, class_id):
        """
        Create sequence dengan disease context:
        - English-only: [SOT] [DISEASE] [TEXT] [EOT] 
        - Multilingual: [SOT] [EN] [DISEASE] [TRANSCRIBE] [TEXT] [EOT]
        """
        disease_token_id = self.get_disease_token_id(class_id)
        
        # Build sequence
        sequence = [self.tokenizer.sot]
        
        if not self.is_english_only:
            # Multilingual: [SOT] [EN] [DISEASE] [TRANSCRIBE] [TEXT] [EOT]
            sequence.extend([
                self.tokenizer.language_token,
                disease_token_id,
                self.tokenizer.transcribe
            ])
        else:
            # English-only: [SOT] [DISEASE] [TEXT] [EOT]
            sequence.append(disease_token_id)
        
        # Add text tokens - tambahkan spasi di depan untuk tokenization yang benar
        text_tokens = self.tokenizer.encode(" " + text.strip())
        sequence.extend(text_tokens)
        sequence.append(self.tokenizer.eot)
        
        return sequence
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        try:
            # Load dan process audio menggunakan fungsi Whisper
            audio_path = row['file']
            mel = self.load_and_process_audio(audio_path)
            
            # Get text dan class
            text = row['text']
            class_id = int(row['class'])
            
            # Create sequence dengan disease context
            token_sequence = self.create_sequence_with_disease_context(text, class_id)
            input_tokens = torch.tensor(token_sequence[:-1], dtype=torch.long)  # Remove last EOT
            target_tokens = torch.tensor(token_sequence[1:], dtype=torch.long)   # Remove first SOT
            
            return {
                'mel': mel,
                'input_tokens': input_tokens,
                'target_tokens': target_tokens,
                'class': torch.tensor(class_id, dtype=torch.long),
                'text': text,
                'path': audio_path,
            }
            
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            
            # Return dummy data dengan format yang konsisten
            dummy_mel = torch.zeros((80, 3000), dtype=torch.float32)  # Whisper standard shape
            dummy_sequence = self.create_sequence_with_disease_context("", 0)
            dummy_input_tokens = torch.tensor(dummy_sequence[:-1], dtype=torch.long)
            dummy_target_tokens = torch.tensor(dummy_sequence[1:], dtype=torch.long)
            
            return {
                'mel': dummy_mel,
                'input_tokens': dummy_input_tokens,
                'target_tokens': dummy_target_tokens,
                'class': torch.tensor(0, dtype=torch.long),
                'text': "",
                'path': row.get('file', 'unknown'),
            }
    
    def get_collate_fn(self):
        """Return a collate function dengan proper padding"""
        def collate_fn(batch):
            mels = torch.stack([item['mel'] for item in batch])
            classes = torch.stack([item['class'] for item in batch])
            texts = [item['text'] for item in batch]
            paths = [item['path'] for item in batch]
            
            # Collect sequences
            input_tokens = [item['input_tokens'] for item in batch]
            target_tokens = [item['target_tokens'] for item in batch]
            
            # Calculate max length
            max_len = max(max(len(inp), len(tgt)) for inp, tgt in zip(input_tokens, target_tokens))
            
            padded_inputs = []
            padded_targets = []
            
            for inp, tgt in zip(input_tokens, target_tokens):
                # Pad input dengan EOT token
                inp_padded = torch.cat([
                    inp, 
                    torch.full((max_len - len(inp),), self.tokenizer.eot, dtype=inp.dtype)
                ])
                
                # Pad target dengan -100 untuk ignore di loss
                tgt_padded = torch.cat([
                    tgt, 
                    torch.full((max_len - len(tgt),), -100, dtype=tgt.dtype)
                ])
                
                padded_inputs.append(inp_padded)
                padded_targets.append(tgt_padded)
            
            return {
                'mels': mels,
                'input_tokens': torch.stack(padded_inputs),
                'target_tokens': torch.stack(padded_targets),
                'classes': classes,
                'texts': texts,
                'paths': paths
            }
        
        return collate_fn