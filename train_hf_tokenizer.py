# -*- coding: utf-8 -*-
"""
Train a fast Thai tokenizer using HuggingFace tokenizers (Rust backend)
- BPE (subword) model, ready for LLM/NLP/transformers
- Save in HuggingFace-compatible format (tokenizer.json, vocab.json, merges.txt, tokenizer_config.json)
- Enhanced with better Thai language support and error handling
"""

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, normalizers
import os
import json
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === CONFIG ===
CORPUS_PATH = "data/combined_thai_corpus.txt"  # Plain text, one sentence per line
VOCAB_SIZE = 16000
TOKENIZER_DIR = "Bitthaitokenizer"
MIN_FREQUENCY = 2  # Minimum frequency for BPE merges

# Special tokens optimized for Thai and modern LLMs
SPECIAL_TOKENS = [
    "<pad>", "<unk>", "<s>", "</s>", "<mask>",
    "<cls>", "<sep>", "<usr>", "<sys>"  # Additional tokens for chat/instruction models
]

def validate_corpus(corpus_path):
    """Validate that the corpus file exists and is readable"""
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")
    
    # Check file size and encoding
    try:
        with open(corpus_path, 'r', encoding='utf-8') as f:
            first_lines = [f.readline().strip() for _ in range(5)]
            line_count = sum(1 for _ in f) + 5
        
        logger.info(f"Corpus validation successful:")
        logger.info(f"  - File: {corpus_path}")
        logger.info(f"  - Estimated lines: {line_count}")
        logger.info(f"  - Sample lines: {first_lines[:3]}")
        
        return True
    except UnicodeDecodeError:
        raise ValueError(f"Corpus file must be UTF-8 encoded: {corpus_path}")

def create_thai_tokenizer():
    """Create and configure the Thai tokenizer"""
    
    # Initialize BPE model
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    
    # Use only NFKC normalization (do NOT use StripAccents for Thai)
    tokenizer.normalizer = normalizers.NFKC()
    
    # Pre-tokenizer: handle whitespace and punctuation
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Whitespace(),
        pre_tokenizers.Punctuation()
    ])
    
    # BPE trainer with optimized settings for Thai
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQUENCY,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    
    # Decoder
    tokenizer.decoder = decoders.BPEDecoder()
    
    return tokenizer, trainer

def save_tokenizer_files(tokenizer, save_dir):
    """Save tokenizer in multiple formats for compatibility"""
    
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    # Main tokenizer file
    tokenizer.save(str(save_path / "tokenizer.json"))
    logger.info(f"Saved tokenizer.json")
    
    # Vocabulary file
    vocab = tokenizer.get_vocab()
    with open(save_path / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved vocab.json ({len(vocab)} tokens)")
    
    # Tokenizer configuration
    config = {
        "unk_token": {"content": "<unk>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": True},
        "pad_token": {"content": "<pad>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": True},
        "bos_token": {"content": "<s>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": True},
        "eos_token": {"content": "</s>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": True},
        "mask_token": {"content": "<mask>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": True},
        "cls_token": {"content": "<cls>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": True},
        "sep_token": {"content": "<sep>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": True},
        "model_max_length": 512,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "auto_map": {
            "AutoTokenizer": ["transformers.models.gpt2.tokenization_gpt2_fast.GPT2TokenizerFast", None]
        }
    }
    
    with open(save_path / "tokenizer_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved tokenizer_config.json")

def test_tokenizer(tokenizer, test_sentences=None):
    """Test the trained tokenizer with sample Thai sentences"""
    
    if test_sentences is None:
        test_sentences = [
            "สวัสดีครับ ผมชื่อ จอห์น",
            "วันนี้อากาศดีมาก",
            "ขอบคุณสำหรับความช่วยเหลือ",
            "Hello, this is mixed Thai-English text สวัสดี"
        ]
    
    logger.info("Testing tokenizer:")
    for sentence in test_sentences:
        encoded = tokenizer.encode(sentence)
        decoded = tokenizer.decode(encoded.ids)
        logger.info(f"  Original: {sentence}")
        logger.info(f"  Tokens: {encoded.tokens}")
        logger.info(f"  Decoded: {decoded}")
        logger.info(f"  Token count: {len(encoded.tokens)}")
        logger.info("  " + "-" * 50)

def main():
    """Main training pipeline"""
    
    try:
        logger.info("Starting Thai tokenizer training...")
        
        # Validate corpus
        validate_corpus(CORPUS_PATH)
        
        # Create tokenizer
        tokenizer, trainer = create_thai_tokenizer()
        
        # Train tokenizer
        logger.info(f"Training tokenizer on {CORPUS_PATH}...")
        logger.info(f"Target vocabulary size: {VOCAB_SIZE}")
        
        tokenizer.train([CORPUS_PATH], trainer=trainer)
        
        # Save tokenizer
        save_tokenizer_files(tokenizer, TOKENIZER_DIR)
        
        # Test tokenizer
        test_tokenizer(tokenizer)
        
        logger.info(f"✅ Tokenizer training completed successfully!")
        logger.info(f"📁 Files saved to: {TOKENIZER_DIR}/")
        logger.info(f"🔤 Vocabulary size: {len(tokenizer.get_vocab())}")
        
        return tokenizer
        
    except Exception as e:
        logger.error(f"❌ Training failed: {str(e)}")
        raise

# === OPTIONAL: Advanced Thai Pre-tokenizer ===
def create_thai_pretokenizer():
    """
    Custom Thai pre-tokenizer using PyThaiNLP
    Uncomment and modify if you want Thai word-level tokenization
    """
    try:
        from pythainlp.tokenize import word_tokenize
        
        class ThaiPreTokenizer:
            def pre_tokenize(self, pretok):
                splits = []
                text = pretok.normalized.str
                offset = 0
                
                # Use PyThaiNLP for Thai word segmentation
                words = word_tokenize(text, engine="newmm")
                
                for word in words:
                    if word.strip():  # Skip empty words
                        start = text.find(word, offset)
                        if start != -1:
                            splits.append((word, (start, start + len(word))))
                            offset = start + len(word)
                
                pretok.split(splits)
        
        return ThaiPreTokenizer()
        
    except ImportError:
        logger.warning("PyThaiNLP not installed. Using default pre-tokenizer.")
        logger.warning("Install with: pip install pythainlp")
        return None

if __name__ == "__main__":
    # Uncomment to use Thai-specific pre-tokenizer
    # thai_pretokenizer = create_thai_pretokenizer()
    # if thai_pretokenizer:
    #     logger.info("Using Thai-specific pre-tokenizer")
    
    tokenizer = main()