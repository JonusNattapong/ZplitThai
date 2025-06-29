"""
Enhanced Thai ML-based Word Tokenizer Training Script
BiLSTM+CRF model for Thai word segmentation

Features:
- Improved data preprocessing with multiple corpus support
- Better model architecture with CRF layer
- Comprehensive evaluation metrics
- Training resume capability
- Configurable hyperparameters
- Progress tracking and logging
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm
import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import pickle

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import PyThaiNLP for preprocessing
try:
    from pythainlp.tokenize import word_tokenize
    from pythainlp.util import isthai
    PYTHAINLP_AVAILABLE = True
except ImportError:
    logger.warning("PyThaiNLP not available. Limited preprocessing functionality.")
    PYTHAINLP_AVAILABLE = False

class ThaiWordSegDataset(Dataset):
    """Dataset class for Thai word segmentation with IOB tagging"""
    
    def __init__(self, data_path: str, char2idx: Dict = None, tag2idx: Dict = None, 
                 max_length: int = 512, show_progress: bool = True):
        self.samples = []
        self.max_length = max_length
        
        # Load data
        self._load_data(data_path, show_progress)
        
        # Build vocabularies if not provided
        if char2idx is None or tag2idx is None:
            self.char2idx, self.tag2idx, self.idx2tag = self._build_vocab()
        else:
            self.char2idx = char2idx
            self.tag2idx = tag2idx
            self.idx2tag = {v: k for k, v in tag2idx.items()}
        
        logger.info(f"Dataset loaded: {len(self.samples)} samples")
        logger.info(f"Vocabulary size: {len(self.char2idx)} characters, {len(self.tag2idx)} tags")
    
    def _load_data(self, data_path: str, show_progress: bool):
        """Load IOB formatted data from file"""
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        # Count lines for progress bar
        with open(data_path, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)
        
        with open(data_path, 'r', encoding='utf-8') as f:
            chars, labels = [], []
            iterator = tqdm(f, total=total_lines, desc='Loading data', ncols=80) if show_progress else f
            
            for line in iterator:
                line = line.strip()
                if not line:  # Empty line indicates end of sentence
                    if chars and len(chars) <= self.max_length:
                        self.samples.append((chars[:], labels[:]))
                    chars, labels = [], []
                    continue
                
                try:
                    parts = line.split()
                    if len(parts) >= 2:
                        char, label = parts[0], parts[1]
                        chars.append(char)
                        labels.append(label)
                    else:
                        logger.warning(f"Skipping malformed line: {line}")
                except Exception as e:
                    logger.warning(f"Error processing line '{line}': {e}")
            
            # Add last sample if exists
            if chars and len(chars) <= self.max_length:
                self.samples.append((chars, labels))
    
    def _build_vocab(self):
        """Build character and tag vocabularies"""
        chars = set()
        tags = set()
        
        for char_seq, label_seq in self.samples:
            chars.update(char_seq)
            tags.update(label_seq)
        
        # Create mappings
        char2idx = {'<PAD>': 0, '<UNK>': 1}
        char2idx.update({c: i + 2 for i, c in enumerate(sorted(chars))})
        
        tag2idx = {t: i for i, t in enumerate(sorted(tags))}
        idx2tag = {i: t for t, i in tag2idx.items()}
        
        return char2idx, tag2idx, idx2tag
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        chars, labels = self.samples[idx]
        
        # Convert to indices
        char_indices = [self.char2idx.get(c, self.char2idx['<UNK>']) for c in chars]
        label_indices = [self.tag2idx[l] for l in labels]
        
        return torch.tensor(char_indices), torch.tensor(label_indices)

class BiLSTMCRF(nn.Module):
    """BiLSTM-CRF model for sequence labeling"""
    
    def __init__(self, vocab_size: int, tagset_size: int, embedding_dim: int = 128, 
                 hidden_dim: int = 256, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tagset_size = tagset_size
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        
        # BiLSTM layer
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim // 2, num_layers=num_layers,
            bidirectional=True, batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        
        # Linear layer to project to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        
        # CRF layer
        self.crf = CRF(tagset_size)
    
    def forward(self, sentences, tags=None, mask=None):
        # Get embeddings
        embeds = self.embedding(sentences)
        embeds = self.dropout(embeds)
        
        # Pass through BiLSTM
        lstm_out, _ = self.lstm(embeds)
        lstm_out = self.dropout(lstm_out)
        
        # Project to tag space
        emissions = self.hidden2tag(lstm_out)
        
        if tags is not None:
            # Training mode: return negative log likelihood (scalar)
            return -self.crf(emissions, tags, mask=mask).mean()
        else:
            # Inference mode: return best path
            return self.crf.decode(emissions, mask=mask)

class CRF(nn.Module):
    """Conditional Random Field layer"""
    
    def __init__(self, num_tags: int):
        super().__init__()
        self.num_tags = num_tags
        
        # Transition parameters
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        
        # Initialize transitions
        self.transitions.data[0, :] = -10000  # Never transition to PAD
        self.transitions.data[:, 0] = -10000  # Never transition from PAD
    
    def forward(self, emissions, tags, mask=None):
        """Compute log likelihood of tags given emissions"""
        return self._compute_log_likelihood(emissions, tags, mask)
    
    def decode(self, emissions, mask=None):
        """Find the best path using Viterbi algorithm"""
        return self._viterbi_decode(emissions, mask)
    
    def _compute_log_likelihood(self, emissions, tags, mask):
        batch_size, seq_length = tags.shape
        
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.bool)
        
        # Compute score of given sequence
        score = self._compute_score(emissions, tags, mask)
        
        # Compute partition function
        partition = self._compute_partition(emissions, mask)
        
        return score - partition
    
    def _compute_score(self, emissions, tags, mask):
        batch_size, seq_length = tags.shape
        score = torch.zeros(batch_size, device=emissions.device)
        
        # Add emission scores
        for i in range(seq_length):
            mask_i = mask[:, i]
            emit_score = emissions[:, i].gather(1, tags[:, i].unsqueeze(1)).squeeze(1)
            score += emit_score * mask_i
        
        # Add transition scores
        for i in range(1, seq_length):
            mask_i = mask[:, i]
            trans_score = self.transitions[tags[:, i-1], tags[:, i]]
            score += trans_score * mask_i
        
        return score
    
    def _compute_partition(self, emissions, mask):
        batch_size, seq_length, num_tags = emissions.shape
        
        # Initialize forward variables
        forward_var = emissions[:, 0]  # [batch_size, num_tags]
        
        for i in range(1, seq_length):
            # Broadcast to compute all possible transitions
            emit_score = emissions[:, i].unsqueeze(1)  # [batch_size, 1, num_tags]
            trans_score = self.transitions.unsqueeze(0)  # [1, num_tags, num_tags]
            next_tag_var = forward_var.unsqueeze(2) + trans_score + emit_score
            
            # Apply mask
            mask_i = mask[:, i].unsqueeze(1).unsqueeze(2)
            next_tag_var = torch.where(mask_i, next_tag_var, forward_var.unsqueeze(2))
            
            forward_var = torch.logsumexp(next_tag_var, dim=1)
        
        return torch.logsumexp(forward_var, dim=1)
    
    def _viterbi_decode(self, emissions, mask):
        batch_size, seq_length, num_tags = emissions.shape
        
        # Initialize
        viterbi_vars = emissions[:, 0]  # [batch_size, num_tags]
        path_indices = []
        
        # Forward
        for i in range(1, seq_length):
            next_tag_var = viterbi_vars.unsqueeze(2) + self.transitions.unsqueeze(0)
            best_tag_ids = torch.argmax(next_tag_var, dim=1)
            path_indices.append(best_tag_ids)
            viterbi_vars = torch.max(next_tag_var, dim=1)[0] + emissions[:, i]
        
        # Backward
        best_paths = []
        for batch_idx in range(batch_size):
            # Find best last tag
            best_last_tag = torch.argmax(viterbi_vars[batch_idx])
            best_path = [best_last_tag.item()]
            
            # Backtrack
            for i in range(len(path_indices) - 1, -1, -1):
                best_last_tag = path_indices[i][batch_idx][best_last_tag]
                best_path.append(best_last_tag.item())
            
            best_path.reverse()
            best_paths.append(best_path)
        
        return best_paths

class DataPreprocessor:
    """Data preprocessing utilities"""
    
    @staticmethod
    def merge_text_files(input_files: List[str], output_file: str) -> bool:
        """Merge multiple text files into one"""
        try:
            with open(output_file, 'w', encoding='utf-8') as fout:
                for input_file in input_files:
                    if not os.path.exists(input_file):
                        logger.warning(f'File not found: {input_file}')
                        continue
                    
                    with open(input_file, 'r', encoding='utf-8') as fin:
                        content = fin.read()
                        if content and not content.endswith('\n'):
                            content += '\n'
                        fout.write(content)
                    
                    logger.info(f'Merged: {input_file}')
            
            logger.info(f'All files merged to: {output_file}')
            return True
        except Exception as e:
            logger.error(f'Error merging files: {e}')
            return False
    
    @staticmethod
    def text_to_iob(input_file: str, output_file: str) -> bool:
        """Convert plain text to IOB format using PyThaiNLP"""
        if not PYTHAINLP_AVAILABLE:
            logger.error("PyThaiNLP not available for text preprocessing")
            logger.info("Please install PyThaiNLP: pip install pythainlp")
            return False
        
        try:
            # Count lines for progress tracking
            with open(input_file, 'r', encoding='utf-8') as f:
                total_lines = sum(1 for _ in f)
            
            logger.info(f"Converting {total_lines} lines to IOB format...")
            
            with open(input_file, 'r', encoding='utf-8') as fin, \
                 open(output_file, 'w', encoding='utf-8') as fout:
                
                processed_lines = 0
                
                for line_num, line in enumerate(tqdm(fin, total=total_lines, desc="Converting to IOB"), 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        # Tokenize using PyThaiNLP
                        words = word_tokenize(line, engine="newmm")
                        
                        for word in words:
                            if word.strip():
                                chars = list(word)
                                if chars:
                                    # Check if word contains Thai characters or not, but always require chars
                                    fout.write(f"{chars[0]} B\n")
                                    for char in chars[1:]:
                                        fout.write(f"{char} I\n")
                        
                        fout.write("\n")  # Sentence boundary
                        processed_lines += 1
                        
                        # Progress update every 1000 lines
                        if processed_lines % 1000 == 0:
                            logger.debug(f"Processed {processed_lines} lines")
                    
                    except Exception as e:
                        logger.warning(f"Error processing line {line_num}: {e}")
            
            logger.info(f"IOB data saved to: {output_file} ({processed_lines} lines processed)")
            return True
        
        except Exception as e:
            logger.error(f"Error converting to IOB format: {e}")
            return False

def collate_fn(batch):
    """Custom collate function for DataLoader"""
    chars, labels = zip(*batch)
    
    # Pad sequences
    chars_padded = pad_sequence(chars, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=0)
    
    # Create mask
    mask = (chars_padded != 0)
    
    return chars_padded, labels_padded, mask

def evaluate_model(model, dataloader, device, idx2tag):
    """Evaluate model performance"""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for chars, labels, mask in tqdm(dataloader, desc="Evaluating"):
            chars, labels, mask = chars.to(device), labels.to(device), mask.to(device)
            
            # Get predictions
            predictions = model(chars, mask=mask)
            
            # Convert to lists and filter by mask
            for i, (pred_seq, label_seq, mask_seq) in enumerate(zip(predictions, labels, mask)):
                # Get actual length
                actual_length = mask_seq.sum().item()
                
                pred_tags = [idx2tag[p] for p in pred_seq[:actual_length]]
                true_tags = [idx2tag[l.item()] for l in label_seq[:actual_length]]
                
                all_predictions.extend(pred_tags)
                all_labels.extend(true_tags)
    
    # Calculate metrics
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    report = classification_report(all_labels, all_predictions, output_dict=True)
    
    return f1, report

def train_model(config):
    """Main training function"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create directories
    os.makedirs(config['model_dir'], exist_ok=True)
    os.makedirs(config['data_dir'], exist_ok=True)
    
    # Prepare data
    preprocessor = DataPreprocessor()
    # --- Check if IOB file exists ---
    iob_file = os.path.join(config['data_dir'], 'train_iob_strict.txt')
    if os.path.exists(iob_file):
        logger.info(f"Found existing IOB file: {iob_file}, skipping merge and preprocess.")
        config['train_data'] = iob_file
    else:
        # Merge corpus files if specified
        if config['corpus_files']:
            combined_corpus = os.path.join(config['data_dir'], 'combined_corpus.txt')
            if not preprocessor.merge_text_files(config['corpus_files'], combined_corpus):
                return False
            # Convert to IOB format
            if not preprocessor.text_to_iob(combined_corpus, iob_file):
                return False
            config['train_data'] = iob_file
    # --- Validation/Test set preprocess ---
    test_iob_file = None
    if config.get('test_data'):
        test_iob_file = config['test_data']
        if not os.path.exists(test_iob_file):
            # If test_data is plain text, convert to IOB
            if test_iob_file.endswith('.txt'):
                logger.info(f"Preprocessing test set: {test_iob_file}")
                if not preprocessor.text_to_iob(test_iob_file.replace('_iob.txt', '.txt'), test_iob_file):
                    logger.error(f"Failed to preprocess test set: {test_iob_file}")
                    test_iob_file = None
    # Load datasets
    train_dataset = ThaiWordSegDataset(config['train_data'], max_length=config['max_length'])
    
    # Save vocabularies
    vocab_path = os.path.join(config['model_dir'], 'vocab.pkl')
    with open(vocab_path, 'wb') as f:
        pickle.dump({
            'char2idx': train_dataset.char2idx,
            'tag2idx': train_dataset.tag2idx,
            'idx2tag': train_dataset.idx2tag
        }, f)
    
    # Create data loader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        collate_fn=collate_fn
    )
    
    # Load test data if available
    test_loader = None
    if test_iob_file and os.path.exists(test_iob_file):
        test_dataset = ThaiWordSegDataset(
            test_iob_file,
            char2idx=train_dataset.char2idx,
            tag2idx=train_dataset.tag2idx,
            max_length=config['max_length']
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            collate_fn=collate_fn
        )
    # Initialize model
    model = BiLSTMCRF(
        vocab_size=len(train_dataset.char2idx),
        tagset_size=len(train_dataset.tag2idx),
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    ).to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    # Resume training if checkpoint exists
    model_path = os.path.join(config['model_dir'], 'best_model.pth')
    start_epoch = 0
    best_f1 = 0.0
    
    if os.path.exists(model_path) and config['resume']:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_f1 = checkpoint.get('best_f1', 0.0)
        logger.info(f"Resumed training from epoch {start_epoch}")
    
    # Training loop
    for epoch in range(start_epoch, config['num_epochs']):
        model.train()
        total_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
        
        for batch_idx, (chars, labels, mask) in enumerate(progress_bar):
            chars, labels, mask = chars.to(device), labels.to(device), mask.to(device)
            
            optimizer.zero_grad()
            loss = model(chars, labels, mask)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
        
        # Evaluate on test set
        if test_loader:
            f1_score, report = evaluate_model(model, test_loader, device, train_dataset.idx2tag)
            logger.info(f"Epoch {epoch+1} - F1 Score: {f1_score:.4f}")
            
            scheduler.step(f1_score)
            
            # Save best model
            if f1_score > best_f1:
                best_f1 = f1_score
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_f1': best_f1,
                    'config': config
                }, model_path)
                logger.info(f"New best model saved with F1: {best_f1:.4f}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(config['model_dir'], 'checkpoint.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_f1': best_f1,
            'config': config
        }, checkpoint_path)
    
    logger.info("Training completed!")
    return True

def analyze_iob_dataset(filepath, allowed_tags={'B', 'I'}):
    """Check IOB dataset quality and print summary/warnings."""
    import collections
    total_lines = 0
    malformed = 0
    tag_counter = collections.Counter()
    char_counter = collections.Counter()
    lengths = []
    with open(filepath, encoding='utf-8') as f:
        sent_len = 0
        for line in f:
            line = line.strip()
            if not line:
                if sent_len > 0:
                    lengths.append(sent_len)
                sent_len = 0
                continue
            total_lines += 1
            parts = line.split()
            if len(parts) != 2:
                malformed += 1
                continue
            char, tag = parts
            tag_counter[tag] += 1
            char_counter[char] += 1
            sent_len += 1
            if tag not in allowed_tags:
                logger.warning(f"Unknown tag: {tag} in line: {line}")
    if sent_len > 0:
        lengths.append(sent_len)
    logger.info(f"[DATASET QC] Total lines: {total_lines}, malformed: {malformed}")
    logger.info(f"[DATASET QC] Tag counts: {dict(tag_counter)}")
    logger.info(f"[DATASET QC] Char vocab size: {len(char_counter)}")
    logger.info(f"[DATASET QC] Sentence length: min={min(lengths) if lengths else 0}, max={max(lengths) if lengths else 0}, avg={sum(lengths)/len(lengths) if lengths else 0:.2f}")
    if malformed > 0:
        logger.warning(f"[DATASET QC] Found {malformed} malformed lines in {filepath}")
    # Show 5 random samples
    import random
    logger.info("[DATASET QC] Random samples:")
    with open(filepath, encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()]
        for sample in random.sample(lines, min(5, len(lines))):
            logger.info(f"  {sample}")

def main():
    parser = argparse.ArgumentParser(description='Train Thai ML Tokenizer')
    parser.add_argument('--config', type=str, help='Config file path')
    parser.add_argument('--corpus-files', nargs='+', help='Corpus files to merge')
    parser.add_argument('--train-data', type=str, help='Training data path')
    parser.add_argument('--test-data', type=str, help='Test data path')
    parser.add_argument('--model-dir', type=str, default='model', help='Model directory')
    parser.add_argument('--data-dir', type=str, default='data', help='Data directory')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--num-epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--resume', action='store_true', help='Resume training')
    
    args = parser.parse_args()
    # Recommended starting points
    # Default configuration
    config = {
        'model_dir': args.model_dir,
        'data_dir': args.data_dir,
        'batch_size': 32,
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'embedding_dim': 128,      # Try 64, 128, 256
        'hidden_dim': 256,         # Try 128, 256, 512
        'num_layers': 2,           # Try 1, 2, 3
        'dropout': 0.3,            # Try 0.1, 0.3, 0.5
        'max_length': 512,
        'resume': args.resume,
        'corpus_files': args.corpus_files,
        'train_data': args.train_data,
        'test_data': args.test_data
    }
    
    # Load config file if provided
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            file_config = json.load(f)
        config.update(file_config)
    
    # Default corpus files if not specified
    if not config['corpus_files'] and not config['train_data']:
        config['corpus_files'] = [
            'data/thai_corpus1.txt',
            'data/thai_corpus2.txt',
            'data/thai_corpus3.txt',
            'data/thai_corpus4.txt',
            'data/thai_corpus5.txt'
        ]
    
    # Start training
    success = train_model(config)
    
    if success:
        logger.info("Training completed successfully!")
    else:
        logger.error("Training failed!")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())