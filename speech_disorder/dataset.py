import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
import librosa
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
    """Dataset untuk multi-task learning dengan librosa audio processing"""
    
    def __init__(self, csv_file, config):
        self.df = pd.read_csv(csv_file)
        self.config = config
        
        # Audio parameters
        self.sample_rate = config.sample_rate
        self.n_mels = config.n_mels
        self.n_fft = config.n_fft
        self.hop_length = config.hop_length
        self.win_length = config.win_length
        self.max_audio_length = config.max_audio_length
        
        # DETECT MODEL TYPE
        self.is_english_only = '.en' in getattr(config, 'model_size', '')
        
        # CONDITIONAL TOKENIZER
        if self.is_english_only:
            # English-only model
            self.tokenizer = get_tokenizer(
                multilingual=False,  
                language=None,       
                task=None,          
                include_diseases=True
            )
            print("Dataset using English-only tokenizer")
        else:
            # Multilingual model (existing)
            self.tokenizer = get_tokenizer(
                multilingual=True,
                num_languages=99,
                language="en",
                task="transcribe",
                include_diseases=True
            )
        
        # Disease mapping
        self.disease_mapping = config.class_to_disease
        
        print(f"Multi-task Dataset loaded: {len(self.df)} samples")
        print(f"Model type: {'English-only' if self.is_english_only else 'Multilingual'}")
        print(f"EOT token: {self.tokenizer.eot}")  # Print EOT token untuk debugging
        print(f"SOT token: {self.tokenizer.sot}")  # Print SOT token untuk debugging
        print(f"Disease tokens available: {list(self.tokenizer.disease_tokens.keys())}")
        
        # Print class distribution
        class_counts = self.df['class'].value_counts().sort_index()
        print("Class distribution:")
        for class_id, count in class_counts.items():
            disease_name = self.disease_mapping.get(class_id, f"unknown_{class_id}")
            print(f"  {disease_name}/{class_id}: {count} samples ({count/len(self.df)*100:.1f}%)")

    def __len__(self):
        return len(self.df)
    
    def load_audio_librosa(self, audio_path):
        """Load audio using librosa with Whisper-compatible preprocessing"""
        try:
            # Load audio with librosa
            audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            
            # Normalize audio
            audio = librosa.util.normalize(audio)
            
            # Pad or trim to max length (30 seconds)
            if len(audio) > self.max_audio_length:
                audio = audio[:self.max_audio_length]
            else:
                # Pad with zeros
                pad_length = self.max_audio_length - len(audio)
                audio = np.pad(audio, (0, pad_length), mode='constant', constant_values=0)
            
            return audio.astype(np.float32)
            
        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}")
            # Return silence if loading fails
            return np.zeros(self.max_audio_length, dtype=np.float32)
    
    def audio_to_mel_spectrogram(self, audio):
        """Convert audio to mel spectrogram using librosa (Whisper-compatible)"""
        try:
            # Compute mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=self.sample_rate,
                n_mels=self.n_mels,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                fmin=0,
                fmax=self.sample_rate // 2,
                power=2.0
            )
            
            # Convert to log scale
            mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Normalize to range similar to Whisper
            mel_spec = (mel_spec + 80) / 80  # Approximate Whisper normalization
            
            # Ensure consistent shape (80, 3000) for 30-second audio
            target_frames = 3000
            if mel_spec.shape[1] > target_frames:
                mel_spec = mel_spec[:, :target_frames]
            elif mel_spec.shape[1] < target_frames:
                pad_width = target_frames - mel_spec.shape[1]
                mel_spec = np.pad(mel_spec, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
            
            return mel_spec.astype(np.float32)
            
        except Exception as e:
            print(f"Error converting to mel spectrogram: {e}")
            # Return zero mel spectrogram
            return np.zeros((self.n_mels, 3000), dtype=np.float32)
    
    def get_disease_token_id(self, class_id):
        """Get disease token ID for the class"""
        disease_name = self.disease_mapping[class_id]
        return self.tokenizer.disease_tokens.get(disease_name, self.tokenizer.eot)
    
    def create_sequence_with_disease_context(self, text, class_id):
        """
        Create sequence dengan disease context sesuai arsitektur:
        [SOT] [EN] [DISEASE] [TRANSCRIBE] [TEXT] [EOT]
        """
        # Disease token
        disease_token_id = self.get_disease_token_id(class_id)
        
        # Build sequence sesuai arsitektur model B
        sequence = [self.tokenizer.sot]
    
        if not self.is_english_only:
            sequence.extend([
                self.tokenizer.language_token,
                disease_token_id,
                self.tokenizer.transcribe
            ])
        else:
            sequence.append(disease_token_id)
        
        # Add text tokens
        text_tokens = self.tokenizer.encode(text)
        sequence.extend(text_tokens)
        sequence.append(self.tokenizer.eot)
        
        return sequence
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        try:
            # Load dan process audio dengan librosa
            audio_path = row['file']
            audio = self.load_audio_librosa(audio_path)
            
            # Convert ke mel spectrogram
            mel = self.audio_to_mel_spectrogram(audio)
            mel = torch.tensor(mel, dtype=torch.float32)
            
            # Get text dan class
            text = row['text']
            class_id = int(row['class'])
            
            # Create sequence dengan disease context
            token_sequence = self.create_sequence_with_disease_context(text, class_id)
            input_tokens = torch.tensor(token_sequence[:-1], dtype=torch.long)  # Remove last EOT
            target_tokens = torch.tensor(token_sequence[1:], dtype=torch.long)   # Remove first SOT
            
            return {
                'mel': mel,
                'input_tokens': input_tokens,    # For decoder input
                'target_tokens': target_tokens,  # For loss computation
                'class': torch.tensor(class_id, dtype=torch.long),
                'text': text,
                'path': audio_path,
            }
            
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            
            # Return dummy data - FIXED: consistent return format
            dummy_mel = torch.zeros((self.n_mels, 3000), dtype=torch.float32)
            dummy_sequence = self.create_sequence_with_disease_context("", 0)
            dummy_input_tokens = torch.tensor(dummy_sequence[:-1], dtype=torch.long)
            dummy_target_tokens = torch.tensor(dummy_sequence[1:], dtype=torch.long)
            
    def get_collate_fn(self):
        """Return a collate function that has access to this dataset's tokenizer"""
        def collate_fn(batch):
            """Use tokenizer to collate batch with proper padding"""
            mels = torch.stack([item['mel'] for item in batch])
            classes = torch.stack([item['class'] for item in batch])
            texts = [item['text'] for item in batch]
            paths = [item['path'] for item in batch]
            
            # Pad sequences with -100 (standard for language modeling)
            input_tokens = [item['input_tokens'] for item in batch]
            target_tokens = [item['target_tokens'] for item in batch]
            
            # Calculate max length for padding
            max_len = max(max(len(inp), len(tgt)) for inp, tgt in zip(input_tokens, target_tokens))
            
            # Use the correct EOT token from this dataset's tokenizer
            eot_token = self.tokenizer.eot
            #print(eot_token, "THis Token")
            
            padded_inputs = []
            padded_targets = []
            
            for inp, tgt in zip(input_tokens, target_tokens):
                # Use the correct EOT token for padding
                inp_padded = torch.cat([inp, torch.full((max_len - len(inp),), eot_token, dtype=inp.dtype)])
                # Pad target with -100 (for loss computation)
                tgt_padded = torch.cat([tgt, torch.full((max_len - len(tgt),), -100, dtype=tgt.dtype)])
                
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