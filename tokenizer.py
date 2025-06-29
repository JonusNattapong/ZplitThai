import os
import torch
import torch.nn as nn
from typing import List, Optional, Tuple

class CRF(nn.Module):
    """Conditional Random Field layer for sequence labeling"""
    def __init__(self, num_tags: int):
        super().__init__()
        self.num_tags = num_tags
        # Transition parameters: transitions[i][j] = score of transitioning from tag i to tag j
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        # Start and end transitions
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions = nn.Parameter(torch.randn(num_tags))
    
    def forward(self, emissions: torch.Tensor, tags: Optional[torch.Tensor] = None, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass for CRF layer"""
        if tags is not None:
            return -self._compute_log_likelihood(emissions, tags, mask)
        else:
            return self._viterbi_decode(emissions, mask)
    
    def _compute_log_likelihood(self, emissions: torch.Tensor, tags: torch.Tensor, 
                               mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute log likelihood for training"""
        batch_size, seq_len = tags.shape
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.bool)
        
        # Compute score for the given tag sequence
        score = torch.zeros(batch_size, device=emissions.device)
        
        # Start transition scores
        score += self.start_transitions[tags[:, 0]]
        
        # Emission scores and transition scores
        for i in range(seq_len):
            score += emissions[:, i].gather(1, tags[:, i].unsqueeze(1)).squeeze(1) * mask[:, i]
            if i < seq_len - 1:
                transition_score = self.transitions[tags[:, i], tags[:, i + 1]]
                score += transition_score * mask[:, i + 1]
        
        # End transition scores
        last_tag_indices = mask.sum(1) - 1
        score += self.end_transitions[tags.gather(1, last_tag_indices.unsqueeze(1)).squeeze(1)]
        
        # Compute partition function (normalization)
        partition = self._compute_partition_function(emissions, mask)
        
        return score - partition
    
    def _compute_partition_function(self, emissions: torch.Tensor, 
                                   mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute partition function using forward algorithm"""
        batch_size, seq_len, num_tags = emissions.shape
        if mask is None:
            mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=emissions.device)
        
        # Initialize forward variables
        forward_var = self.start_transitions.unsqueeze(0) + emissions[:, 0]
        
        for i in range(1, seq_len):
            # Broadcast for transition computation
            emit_score = emissions[:, i].unsqueeze(1)  # [batch, 1, num_tags]
            trans_score = self.transitions.unsqueeze(0)  # [1, num_tags, num_tags]
            next_tag_var = forward_var.unsqueeze(2) + trans_score + emit_score
            
            # Log-sum-exp to get forward variable
            forward_var = torch.logsumexp(next_tag_var, dim=1)
            
            # Apply mask
            forward_var = forward_var * mask[:, i].unsqueeze(1) + \
                         forward_var * (~mask[:, i]).unsqueeze(1)
        
        # Add end transitions
        terminal_var = forward_var + self.end_transitions.unsqueeze(0)
        return torch.logsumexp(terminal_var, dim=1)
    
    def _viterbi_decode(self, emissions: torch.Tensor, 
                       mask: Optional[torch.Tensor] = None) -> List[List[int]]:
        """Viterbi decoding for inference"""
        batch_size, seq_len, num_tags = emissions.shape
        if mask is None:
            mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=emissions.device)
        
        # Initialize
        viterbi_vars = self.start_transitions.unsqueeze(0) + emissions[:, 0]
        path_indices = []
        
        # Forward pass
        for i in range(1, seq_len):
            next_tag_var = viterbi_vars.unsqueeze(2) + self.transitions.unsqueeze(0)
            best_tag_ids = torch.argmax(next_tag_var, dim=1)
            path_indices.append(best_tag_ids)
            viterbi_vars = torch.max(next_tag_var, dim=1)[0] + emissions[:, i]
        
        # Add end transitions
        terminal_var = viterbi_vars + self.end_transitions.unsqueeze(0)
        best_paths = []
        
        # Backward pass to get best paths
        for b in range(batch_size):
            seq_len_b = mask[b].sum().item()
            best_last_tag = torch.argmax(terminal_var[b]).item()
            best_path = [best_last_tag]
            
            for i in range(len(path_indices) - 1, -1, -1):
                if i + 1 < seq_len_b:
                    best_last_tag = path_indices[i][b][best_last_tag].item()
                    best_path.append(best_last_tag)
            
            best_path.reverse()
            best_paths.append(best_path[:seq_len_b])
        
        return best_paths

class BiLSTMCRFTagger(nn.Module):
    """Enhanced BiLSTM-CRF model for Thai word segmentation"""
    def __init__(self, vocab_size: int, tagset_size: int, emb_dim: int = 128, 
                 hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        
        # Enhanced BiLSTM with multiple layers
        self.lstm = nn.LSTM(
            emb_dim, hidden_dim // 2, 
            num_layers=num_layers,
            bidirectional=True, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_dim, tagset_size)
        self.crf = CRF(tagset_size)
        
    def forward(self, x: torch.Tensor, tags: Optional[torch.Tensor] = None, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Embedding and LSTM
        x = self.embedding(x)
        x = self.dropout(x)
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        
        # Linear projection to tag space
        emissions = self.fc(lstm_out)
        
        # CRF layer
        return self.crf(emissions, tags, mask)

class EnhancedThaiMLTokenizer:
    """Enhanced Thai ML tokenizer with better error handling and features"""
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'cpu'):
        self.model_path = model_path or os.path.join('model', 'bilstm_crf.pth')
        self.device = torch.device(device)
        self.model = None
        self.char2idx = None
        self.idx2tag = None
        self.tag2idx = None
        self.max_length = 512  # Maximum sequence length
        self._load_model()

    def _load_model(self) -> bool:
        """Load the trained model with better error handling"""
        if not os.path.exists(self.model_path):
            print(f"Model not found at {self.model_path}")
            print("Available fallback: character-level splitting")
            return False
        
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Validate checkpoint structure
            required_keys = ['char2idx', 'tag2idx', 'idx2tag', 'model']
            if not all(key in checkpoint for key in required_keys):
                print(f"Invalid checkpoint format. Missing keys: {set(required_keys) - set(checkpoint.keys())}")
                return False
            
            # Initialize model
            vocab_size = len(checkpoint['char2idx'])
            tagset_size = len(checkpoint['tag2idx'])
            
            self.model = BiLSTMCRFTagger(vocab_size, tagset_size)
            self.model.load_state_dict(checkpoint['model'])
            self.model.to(self.device)
            self.model.eval()
            
            # Load vocabularies
            self.char2idx = checkpoint['char2idx']
            self.idx2tag = checkpoint['idx2tag']
            self.tag2idx = checkpoint['tag2idx']
            
            print(f"Model loaded successfully. Vocab size: {vocab_size}, Tag size: {tagset_size}")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def _prepare_input(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare input tensor with proper handling of unknown characters"""
        # Handle long sequences
        if len(text) > self.max_length:
            text = text[:self.max_length]
        
        # Convert characters to indices
        char_indices = []
        for char in text:
            idx = self.char2idx.get(char, self.char2idx.get('<UNK>', 0))
            char_indices.append(idx)
        
        # Create tensors
        x = torch.tensor(char_indices, device=self.device).unsqueeze(0)
        mask = torch.ones(len(char_indices), device=self.device).unsqueeze(0)
        
        return x, mask

    def _tags_to_words(self, text: str, tags: List[int]) -> List[str]:
        """Convert BIO tags to word list with improved handling"""
        if not tags or len(tags) != len(text):
            return list(text)  # Fallback to character splitting
        
        words = []
        current_word = ''
        
        for i, (char, tag_idx) in enumerate(zip(text, tags)):
            tag = self.idx2tag.get(tag_idx, 'O')
            
            if tag.startswith('B-') or (tag == 'O' and current_word):
                # Start new word
                if current_word:
                    words.append(current_word)
                current_word = char
            elif tag.startswith('I-') or tag == 'O':
                # Continue current word
                current_word += char
            else:
                # Handle unexpected tags
                current_word += char
        
        # Add the last word
        if current_word:
            words.append(current_word)
        
        return words

    def word_tokenize(self, text: str) -> List[str]:
        """Tokenize Thai text into words"""
        if not text.strip():
            return []
        
        # Fallback for missing model
        if self.model is None:
            return self._character_tokenize(text)
        
        try:
            # Prepare input
            x, mask = self._prepare_input(text)
            
            # Get predictions
            with torch.no_grad():
                if hasattr(self.model, 'crf'):
                    # CRF model returns best paths
                    best_paths = self.model(x, mask=mask)
                    if best_paths and len(best_paths) > 0:
                        tags = best_paths[0]
                    else:
                        raise ValueError("No valid paths returned from CRF")
                else:
                    # Standard model
                    logits = self.model(x)
                    tags = logits.argmax(-1).squeeze(0).tolist()
            
            # Convert tags to words
            words = self._tags_to_words(text, tags)
            return words
            
        except Exception as e:
            print(f"Error during tokenization: {e}")
            return self._character_tokenize(text)

    def _character_tokenize(self, text: str) -> List[str]:
        """Simple character-level tokenization as fallback"""
        return list(text)

    def batch_tokenize(self, texts: List[str]) -> List[List[str]]:
        """Tokenize multiple texts efficiently"""
        return [self.word_tokenize(text) for text in texts]

    def get_model_info(self) -> dict:
        """Get information about the loaded model"""
        if self.model is None:
            return {"status": "No model loaded", "fallback": "character-level"}
        
        return {
            "status": "Model loaded",
            "vocab_size": len(self.char2idx) if self.char2idx else 0,
            "tag_size": len(self.idx2tag) if self.idx2tag else 0,
            "device": str(self.device),
            "max_length": self.max_length,
            "model_path": self.model_path
        }

class ThaiMLTokenizer(EnhancedThaiMLTokenizer):
    """Alias for backward compatibility with old code."""
    pass

# Example usage
if __name__ == "__main__":
    # Initialize tokenizer
    tokenizer = EnhancedThaiMLTokenizer()
    
    # Test tokenization
    test_text = "สวัสดีครับผมชื่อโจ"
    words = tokenizer.word_tokenize(test_text)
    print(f"Text: {test_text}")
    print(f"Words: {words}")
    print(f"Model info: {tokenizer.get_model_info()}")