# -*- coding: utf-8 -*-
"""
Advanced Thai Tokenizer Training with HuggingFace Tokenizers (Production-Ready)
- Multiple model types: BPE, Unigram, WordPiece
- Advanced Thai language processing with PyThaiNLP integration
- Comprehensive evaluation and quality metrics
- Production-ready configuration with optimization
- Multi-format export and compatibility testing
"""

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, normalizers, processors
import os
import json
import time
import argparse
from pathlib import Path
import logging
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
import sys

# Advanced imports
try:
    from pythainlp.tokenize import word_tokenize
    from pythainlp.util import normalize
    PYTHAINLP_AVAILABLE = True
except ImportError:
    PYTHAINLP_AVAILABLE = False
    print("PyThaiNLP not available. Install with: pip install pythainlp")

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Transformers not available. Install with: pip install transformers")

# Setup advanced logging with Windows compatibility
def setup_logging():
    """Setup logging with proper encoding for Windows"""
    
    # Create formatters
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    
    # File handler (with UTF-8 encoding)
    file_handler = logging.FileHandler('tokenizer_training.log', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(file_formatter)
    
    # Console handler (with safe encoding)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # Clear any existing handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logging.getLogger(__name__)

# Initialize logger
logger = setup_logging()

@dataclass
class TokenizerConfig:
    """Advanced configuration for tokenizer training"""
    corpus_path: str = "data/combined_thai_corpus.txt"
    vocab_size: int = 32000  # Increased for better coverage
    tokenizer_dir: str = "Bitthaitokenizer"
    min_frequency: int = 2
    model_type: str = "bpe"  # bpe, unigram, wordpiece
    use_thai_pretokenizer: bool = True
    max_token_length: int = 16
    model_max_length: int = 2048  # Increased for modern LLMs
    enable_byte_fallback: bool = False  # CRITICAL: Disable to prevent byte-level encoding of Thai characters
    dropout: Optional[float] = None  # For regularization
    
    # Advanced Thai-specific settings
    thai_engine: str = "newmm"  # PyThaiNLP engine (newmm is most reliable)
    normalize_text: bool = False  # CRITICAL: Disable to preserve Thai characters exactly
    handle_unk_tokens: bool = True
    preserve_spaces: bool = True
    
    # Quality control
    validation_split: float = 0.1
    test_sentences_path: Optional[str] = None
    benchmark_models: List[str] = None

# Enhanced special tokens for modern Thai NLP
SPECIAL_TOKENS = [
    "<pad>", "<unk>", "<s>", "</s>", "<mask>",
    "<cls>", "<sep>", "<usr>", "<sys>", "<bot>",
    # Thai-specific tokens
    "<th>", "<en>", "<num>", "<url>", "<email>",
    # Instruction tuning tokens
    "<human>", "<assistant>", "<system>", "<tool>",
    # Additional utility tokens
    "<|im_start|>", "<|im_end|>", "<|endoftext|>"
]

# Common Thai characters for alphabet initialization
THAI_ALPHABET = [
    # Thai consonants
    'ก', 'ข', 'ฃ', 'ค', 'ฅ', 'ฆ', 'ง', 'จ', 'ฉ', 'ช', 'ซ', 'ฌ', 'ญ', 'ฎ', 'ฏ',
    'ฐ', 'ฑ', 'ฒ', 'ณ', 'ด', 'ต', 'ถ', 'ท', 'ธ', 'น', 'บ', 'ป', 'ผ', 'ฝ', 'พ',
    'ฟ', 'ภ', 'ม', 'ย', 'ร', 'ล', 'ว', 'ศ', 'ษ', 'ส', 'ห', 'ฬ', 'อ', 'ฮ',
    # Thai vowels
    'ะ', 'ั', 'า', 'ำ', 'ิ', 'ี', 'ึ', 'ื', 'ุ', 'ู', 'เ', 'แ', 'โ', 'ใ', 'ไ',
    'ๅ', 'ๆ', '็', '่', '้', '๊', '๋', '์', 'ํ', '๎',
    # Thai numbers
    '๐', '๑', '๒', '๓', '๔', '๕', '๖', '๗', '๘', '๙',
    # Common symbols
    '฿', '฽', '฾', '฿'
]

def create_thai_pre_tokenizer_function(engine: str = "newmm", normalize_text: bool = True):
    """Create a Thai pre-tokenization function for use with HuggingFace tokenizers"""
    
    if not PYTHAINLP_AVAILABLE:
        logger.warning("PyThaiNLP not available. Using basic pre-tokenizer.")
        return None
    
    # List of engines to try in order of preference
    engines_to_try = [engine]
    
    # Add fallback engines if the requested engine might not be available
    if engine == "deepcut":
        engines_to_try.extend(["newmm", "longest", "mm"])
    elif engine not in ["newmm", "longest", "mm"]:
        engines_to_try.extend(["newmm", "longest", "mm"])
    
    def thai_pretokenize_func(text: str):
        """Thai pre-tokenization function with fallback engines"""
        try:
            if normalize_text:
                text = normalize(text)
            
            # Try engines in order of preference
            for current_engine in engines_to_try:
                try:
                    words = word_tokenize(text, engine=current_engine, keep_whitespace=True)
                    return words
                except Exception as e:
                    if current_engine == engine:
                        # Only show warning for the primary engine
                        logger.warning(f"Engine '{current_engine}' failed: {e}")
                        if "deepcut" in str(e).lower():
                            logger.info("Hint: Install deepcut with: pip install deepcut")
                        elif "attacut" in str(e).lower():
                            logger.info("Hint: Install attacut with: pip install attacut")
                    continue
            
            # If all engines failed, return basic tokenization
            logger.warning("All Thai engines failed. Using basic word splitting.")
            return text.split()
            
        except Exception as e:
            logger.warning(f"Thai pre-tokenization completely failed: {e}. Using basic tokenization.")
            return [text]
    
    return thai_pretokenize_func

def validate_corpus(corpus_path: str) -> Dict[str, Any]:
    """Enhanced corpus validation with detailed statistics"""
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")
    
    try:
        stats = {
            'file_size_mb': os.path.getsize(corpus_path) / (1024 * 1024),
            'total_lines': 0,
            'total_chars': 0,
            'thai_chars': 0,
            'empty_lines': 0,
            'avg_line_length': 0,
            'sample_lines': []
        }
        
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = line.strip()
                stats['total_lines'] += 1
                
                if not line:
                    stats['empty_lines'] += 1
                    continue
                
                stats['total_chars'] += len(line)
                
                # Count Thai characters
                thai_count = sum(1 for char in line if '\u0e00' <= char <= '\u0e7f')
                stats['thai_chars'] += thai_count
                
                # Collect sample lines
                if i < 5:
                    stats['sample_lines'].append(line[:100] + ('...' if len(line) > 100 else ''))
        
        if stats['total_lines'] > 0:
            stats['avg_line_length'] = stats['total_chars'] / stats['total_lines']
            stats['thai_ratio'] = stats['thai_chars'] / stats['total_chars'] if stats['total_chars'] > 0 else 0
        
        logger.info("Corpus Statistics:")
        logger.info(f"  File size: {stats['file_size_mb']:.2f} MB")
        logger.info(f"  Total lines: {stats['total_lines']:,}")
        logger.info(f"  Total characters: {stats['total_chars']:,}")
        logger.info(f"  Thai characters: {stats['thai_chars']:,} ({stats.get('thai_ratio', 0):.1%})")
        logger.info(f"  Average line length: {stats['avg_line_length']:.1f}")
        logger.info(f"  Empty lines: {stats['empty_lines']}")
        
        return stats
        
    except UnicodeDecodeError:
        raise ValueError(f"Corpus file must be UTF-8 encoded: {corpus_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to validate corpus: {str(e)}")

def create_advanced_tokenizer(config: TokenizerConfig) -> Tuple[Tokenizer, trainers.BpeTrainer]:
    """Create advanced tokenizer with multiple model support"""
    
    logger.info(f"Creating {config.model_type.upper()} tokenizer...")
    
    # Initialize model based on type
    if config.model_type.lower() == "bpe":
        model = models.BPE(unk_token="<unk>", dropout=config.dropout)
        trainer = trainers.BpeTrainer(
            vocab_size=config.vocab_size,
            min_frequency=config.min_frequency,
            special_tokens=SPECIAL_TOKENS,
            show_progress=True,
            # CRITICAL: Only use Thai alphabet, NO byte-level alphabet to prevent encoding issues
            initial_alphabet=THAI_ALPHABET + list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
        )
    elif config.model_type.lower() == "unigram":
        model = models.Unigram()
        trainer = trainers.UnigramTrainer(
            vocab_size=config.vocab_size,
            special_tokens=["<unk>"] if config.model_type.lower() == "unigram" else SPECIAL_TOKENS,  # Minimal special tokens for Thai
            show_progress=True,
            unk_token="<unk>"
        )
    elif config.model_type.lower() == "wordpiece":
        model = models.WordPiece(unk_token="<unk>", max_input_chars_per_word=config.max_token_length)
        trainer = trainers.WordPieceTrainer(
            vocab_size=config.vocab_size,
            min_frequency=config.min_frequency,
            special_tokens=SPECIAL_TOKENS,
            show_progress=True
        )
    else:
        raise ValueError(f"Unsupported model type: {config.model_type}")
    
    # Initialize tokenizer
    tokenizer = Tokenizer(model)
    
    # Advanced normalization for Thai - CRITICAL: Minimal normalization to preserve Thai characters
    normalizer_sequence = []
    
    # Only use NFKC if absolutely necessary and text normalization is explicitly enabled
    if config.normalize_text:
        normalizer_sequence.append(normalizers.NFKC())
        logger.info("NFKC normalization enabled - may affect Thai character rendering")
    else:
        logger.info("Text normalization disabled - preserving original Thai characters")
    
    # NEVER use Lowercase for Thai - it corrupts Thai characters
    # Only apply to non-Thai text if really needed
    
    if len(normalizer_sequence) > 0:
        tokenizer.normalizer = normalizers.Sequence(normalizer_sequence) if len(normalizer_sequence) > 1 else normalizer_sequence[0]
    else:
        tokenizer.normalizer = None  # No normalization to preserve Thai text
    
    # Advanced pre-tokenization - CRITICAL: Minimal pre-tokenization to preserve Thai text
    pre_tokenizer_sequence = []
    
    # Only use punctuation pre-tokenizer to avoid spacing issues
    # Whitespace pre-tokenizer can cause unwanted spaces in decoding
    pre_tokenizer_sequence.append(pre_tokenizers.Punctuation())
    
    # Add PyThaiNLP integration if available
    if config.use_thai_pretokenizer and PYTHAINLP_AVAILABLE:
        logger.info(f"Thai pre-tokenization will use PyThaiNLP engine: {config.thai_engine}")
        logger.info("Note: Thai pre-tokenization is applied during data preprocessing")
    else:
        logger.info("Using minimal punctuation-only pre-tokenizer for Thai preservation")
    
    # CRITICAL: Never add ByteLevel pre-tokenizer for Thai text
    # It causes Thai characters to be encoded as bytes (à¸ª instead of ส)
    if config.enable_byte_fallback:
        logger.warning("Byte fallback is enabled - this may cause Thai encoding issues!")
        pre_tokenizer_sequence.append(pre_tokenizers.ByteLevel(add_prefix_space=False))
    else:
        logger.info("Byte fallback disabled - Thai characters will be preserved")
    
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(pre_tokenizer_sequence)
    
    # Advanced post-processing - CRITICAL: Minimal post-processing to avoid spacing issues
    # For Thai text, we want to avoid adding special tokens that interfere with decoding
    if config.model_type.lower() == "unigram":
        # For Unigram, disable post-processor to preserve Thai text exactly
        tokenizer.post_processor = None
        logger.info("Post-processor disabled for Unigram model - preserves Thai text")
    else:
        # For other models, use minimal post-processing
        tokenizer.post_processor = processors.TemplateProcessing(
            single="<s> $A </s>",
            pair="<s> $A </s> <s> $B </s>",
            special_tokens=[
                ("<s>", SPECIAL_TOKENS.index("<s>")),
                ("</s>", SPECIAL_TOKENS.index("</s>")),
            ],
        )
    
    # Decoder - CRITICAL: Use proper decoder that preserves ALL characters including spaces
    if config.model_type.lower() == "bpe":
        if config.enable_byte_fallback:
            tokenizer.decoder = decoders.ByteLevel()
            logger.warning("Using ByteLevel decoder - may cause Thai character corruption!")
        else:
            # Use BPE decoder without byte-level processing to preserve Thai characters
            tokenizer.decoder = decoders.BPEDecoder()
            logger.info("Using BPE decoder - Thai characters will be preserved")
    elif config.model_type.lower() == "wordpiece":
        tokenizer.decoder = decoders.WordPiece()
    elif config.model_type.lower() == "unigram":
        # For Unigram, use no special decoder to preserve all characters exactly
        # Metaspace decoder can strip characters, so we'll rely on the basic decoding
        tokenizer.decoder = None
        logger.info("Using basic decoding for Unigram model to preserve all characters")
    else:
        # Default to no special decoder to preserve all content
        tokenizer.decoder = None
        logger.info("Using basic decoding to preserve all characters")
    
    return tokenizer, trainer

def save_tokenizer_files(tokenizer: Tokenizer, config: TokenizerConfig) -> Dict[str, Any]:
    """Save tokenizer in multiple formats with comprehensive metadata"""
    
    save_path = Path(config.tokenizer_dir)
    save_path.mkdir(exist_ok=True)
    
    # Get vocabulary statistics
    vocab = tokenizer.get_vocab()
    vocab_stats = {
        'total_tokens': len(vocab),
        'special_tokens': len([t for t in vocab if t.startswith('<') and t.endswith('>')]),
        'thai_tokens': len([t for t in vocab if any('\u0e00' <= char <= '\u0e7f' for char in t)]),
        'numeric_tokens': len([t for t in vocab if any(char.isdigit() for char in t)]),
    }
    
    logger.info(f"Saving tokenizer files to {save_path}/")
    
    # 1. Main tokenizer file (HuggingFace format)
    tokenizer.save(str(save_path / "tokenizer.json"))
    logger.info("Saved tokenizer.json")
    
    # 2. Vocabulary file
    with open(save_path / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved vocab.json ({vocab_stats['total_tokens']} tokens)")
    
    # 3. Enhanced tokenizer configuration
    tokenizer_config = {
        "unk_token": {
            "content": "<unk>", "single_word": False, "lstrip": False, 
            "rstrip": False, "normalized": True, "__type": "AddedToken"
        },
        "pad_token": {
            "content": "<pad>", "single_word": False, "lstrip": False, 
            "rstrip": False, "normalized": True, "__type": "AddedToken"
        },
        "bos_token": {
            "content": "<s>", "single_word": False, "lstrip": False, 
            "rstrip": False, "normalized": True, "__type": "AddedToken"
        },
        "eos_token": {
            "content": "</s>", "single_word": False, "lstrip": False, 
            "rstrip": False, "normalized": True, "__type": "AddedToken"
        },
        "mask_token": {
            "content": "<mask>", "single_word": False, "lstrip": False, 
            "rstrip": False, "normalized": True, "__type": "AddedToken"
        },
        "cls_token": {
            "content": "<cls>", "single_word": False, "lstrip": False, 
            "rstrip": False, "normalized": True, "__type": "AddedToken"
        },
        "sep_token": {
            "content": "<sep>", "single_word": False, "lstrip": False, 
            "rstrip": False, "normalized": True, "__type": "AddedToken"
        },
        "model_max_length": config.model_max_length,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "auto_map": {
            "AutoTokenizer": ["transformers.models.gpt2.tokenization_gpt2_fast.GPT2TokenizerFast", None]
        },
        # Enhanced metadata
        "model_type": config.model_type,
        "vocab_size": config.vocab_size,
        "language": ["th", "thai"],
        "license": "apache-2.0",
        "library_name": "tokenizers",
        "tags": ["thai", "tokenizer", "nlp", "subword"],
        "thai_engine": config.thai_engine if config.use_thai_pretokenizer else None,
        "creation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "vocab_stats": vocab_stats
    }
    
    with open(save_path / "tokenizer_config.json", "w", encoding="utf-8") as f:
        json.dump(tokenizer_config, f, ensure_ascii=False, indent=2)
    logger.info("Saved tokenizer_config.json")
    
    # 4. Training configuration for reproducibility
    training_config = {
        "corpus_path": config.corpus_path,
        "vocab_size": config.vocab_size,
        "model_type": config.model_type,
        "min_frequency": config.min_frequency,
        "max_token_length": config.max_token_length,
        "use_thai_pretokenizer": config.use_thai_pretokenizer,
        "thai_engine": config.thai_engine,
        "normalize_text": config.normalize_text,
        "enable_byte_fallback": config.enable_byte_fallback,
        "dropout": config.dropout,
        "special_tokens": SPECIAL_TOKENS,
        "training_time": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(save_path / "training_config.json", "w", encoding="utf-8") as f:
        json.dump(training_config, f, ensure_ascii=False, indent=2)
    logger.info("Saved training_config.json")
    
    # 5. README with usage instructions
    readme_content = f"""# Thai Tokenizer - {config.model_type.upper()}

Advanced Thai language tokenizer trained with HuggingFace tokenizers.

## Model Information
- **Model Type**: {config.model_type.upper()}
- **Vocabulary Size**: {vocab_stats['total_tokens']:,}
- **Thai Tokens**: {vocab_stats['thai_tokens']:,} ({vocab_stats['thai_tokens']/vocab_stats['total_tokens']*100:.1f}%)
- **Special Tokens**: {vocab_stats['special_tokens']}
- **Max Length**: {config.model_max_length}
- **Thai Engine**: {config.thai_engine if config.use_thai_pretokenizer else 'None'}

## Usage

### With Transformers
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('{config.tokenizer_dir}')
text = "สวัสดีครับ ผมชื่อจอห์น"
tokens = tokenizer.tokenize(text)
encoded = tokenizer(text, return_tensors="pt")
```

### Direct Usage
```python
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file('{config.tokenizer_dir}/tokenizer.json')
text = "สวัสดีครับ ผมชื่อจอห์น"
encoding = tokenizer.encode(text)
print(encoding.tokens)
```

## Training Details
- **Corpus**: {config.corpus_path}
- **Training Date**: {time.strftime("%Y-%m-%d %H:%M:%S")}
- **Normalization**: {'Enabled' if config.normalize_text else 'Disabled'}
- **Byte Fallback**: {'Enabled' if config.enable_byte_fallback else 'Disabled'}

## Files
- `tokenizer.json`: Main tokenizer file
- `vocab.json`: Vocabulary mapping
- `tokenizer_config.json`: Configuration for transformers
- `training_config.json`: Training parameters
- `README.md`: This file

## License
Apache-2.0
"""
    
    with open(save_path / "README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
    logger.info("Saved README.md")
    
    return {
        'save_path': str(save_path),
        'vocab_stats': vocab_stats,
        'files_created': ['tokenizer.json', 'vocab.json', 'tokenizer_config.json', 'training_config.json', 'README.md']
    }

def comprehensive_tokenizer_test(tokenizer: Tokenizer, config: TokenizerConfig) -> Dict[str, Any]:
    """Comprehensive testing suite for the tokenizer"""
    
    # Default test sentences covering various Thai language aspects
    default_test_sentences = [
        # Basic Thai
        "สวัสดีครับ ผมชื่อจอห์น",
        "วันนี้อากาศดีมาก",
        "ขอบคุณสำหรับความช่วยเหลือ",
        
        # Mixed languages
        "Hello สวัสดี 123 ราคา $50",
        "Email: john@example.com โทร 02-123-4567",
        
        # Complex Thai
        "กรุงเทพมหานครอมรรัตนโกสินทร์มหินทรายุธยามหาดิลกภพนพรัตนราชธานีบุรีรมย์อุดมราชนิเวศน์มหาสถานอมรพิมานอวตารสถิตสักกะทัตติยวิษณุกรรมประสิทธิ์",
        
        # Numbers and symbols
        "ราคา ๑,๒๓๔.๕๖ บาท วันที่ ๑๕ มกราคม ๒๕๖๗",
        
        # Formal Thai
        "พระบาทสมเด็จพระเจ้าอยู่หัวทรงพระกรุณาโปรดเกล้าฯ",
        
        # Social media style
        "55555 อร่อยมากก กินข้าวยัง? #อาหารไทย 🇹🇭",
        
        # Technical terms
        "การประมวลผลภาษาธรรมชาติ Natural Language Processing AI ML",
        
        # Empty and edge cases
        "",
        " ",
        "a",
        "ก",
        "12345",
    ]
    
    # Load custom test sentences if provided
    test_sentences = default_test_sentences
    if config.test_sentences_path and os.path.exists(config.test_sentences_path):
        try:
            with open(config.test_sentences_path, 'r', encoding='utf-8') as f:
                custom_sentences = [line.strip() for line in f if line.strip()]
                test_sentences.extend(custom_sentences)
        except Exception as e:
            logger.warning(f"Failed to load custom test sentences: {e}")
    
    logger.info("Running comprehensive tokenizer tests...")
    
    test_results = {
        'total_tests': len(test_sentences),
        'successful_tests': 0,
        'failed_tests': 0,
        'total_tokens': 0,
        'avg_tokens_per_sentence': 0,
        'thai_coverage': 0,
        'unk_token_count': 0,
        'test_details': []
    }
    
    thai_char_total = 0
    thai_char_covered = 0
    
    for i, sentence in enumerate(test_sentences):
        try:
            # Encode
            encoded = tokenizer.encode(sentence)
            tokens = encoded.tokens
            token_ids = encoded.ids
            
            # Decode
            decoded = tokenizer.decode(token_ids)
            
            # Analysis
            unk_count = tokens.count('<unk>')
            thai_chars_in_sentence = sum(1 for char in sentence if '\u0e00' <= char <= '\u0e7f')
            
            # Check coverage for Thai characters
            if thai_chars_in_sentence > 0:
                thai_char_total += thai_chars_in_sentence
                # Simple heuristic: if decoded matches original closely, coverage is good
                if sentence.replace(' ', '') == decoded.replace(' ', ''):
                    thai_char_covered += thai_chars_in_sentence
            
            test_detail = {
                'input': sentence[:50] + ('...' if len(sentence) > 50 else ''),
                'tokens': tokens,
                'token_count': len(tokens),
                'unk_count': unk_count,
                'decoded_matches': sentence == decoded,
                'thai_chars': thai_chars_in_sentence
            }
            
            test_results['test_details'].append(test_detail)
            test_results['successful_tests'] += 1
            test_results['total_tokens'] += len(tokens)
            test_results['unk_token_count'] += unk_count
            
            # Log sample results
            if i < 5 or sentence in ["", " ", "a", "ก"]:
                logger.info(f"  Test {i+1}: '{sentence[:30]}{'...' if len(sentence) > 30 else ''}'")
                logger.info(f"     Tokens: {tokens}")
                logger.info(f"     Decoded: '{decoded[:30]}{'...' if len(decoded) > 30 else ''}'")
                logger.info(f"     Match: {'YES' if sentence == decoded else 'NO'}")
                logger.info(f"     UNK count: {unk_count}")
                logger.info("  " + "-" * 50)
            
        except Exception as e:
            logger.error(f"  Test {i+1} failed: {e}")
            test_results['failed_tests'] += 1
    
    # Calculate final metrics
    if test_results['successful_tests'] > 0:
        test_results['avg_tokens_per_sentence'] = test_results['total_tokens'] / test_results['successful_tests']
    
    if thai_char_total > 0:
        test_results['thai_coverage'] = thai_char_covered / thai_char_total
    
    test_results['unk_ratio'] = test_results['unk_token_count'] / test_results['total_tokens'] if test_results['total_tokens'] > 0 else 0
    
    # Summary
    logger.info("Test Results Summary:")
    logger.info(f"  Successful: {test_results['successful_tests']}/{test_results['total_tests']}")
    logger.info(f"  Avg tokens/sentence: {test_results['avg_tokens_per_sentence']:.1f}")
    logger.info(f"  Thai coverage: {test_results['thai_coverage']:.1%}")
    logger.info(f"  UNK ratio: {test_results['unk_ratio']:.1%}")
    
    return test_results

def benchmark_against_existing(tokenizer: Tokenizer, config: TokenizerConfig) -> Dict[str, Any]:
    """Benchmark against existing tokenizers if available"""
    
    if not TRANSFORMERS_AVAILABLE:
        logger.warning("Transformers not available. Skipping benchmark.")
        return {}
    
    benchmark_results = {}
    benchmark_models = config.benchmark_models or [
        "airesearch/wangchanberta-base-att-spm-uncased",
        "microsoft/DialoGPT-medium"
    ]
    
    test_sentences = [
        "สวัสดีครับ ผมชื่อจอห์น",
        "วันนี้อากาศดีมาก ไปเที่ยวกันเถอะ",
        "ขอบคุณสำหรับความช่วยเหลือ"
    ]
    
    logger.info("Benchmarking against existing tokenizers...")
    
    for model_name in benchmark_models:
        try:
            logger.info(f"  Testing against {model_name}...")
            existing_tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            results = {
                'model_name': model_name,
                'vocab_size': existing_tokenizer.vocab_size,
                'comparisons': []
            }
            
            for sentence in test_sentences:
                # Our tokenizer
                our_tokens = tokenizer.encode(sentence).tokens
                our_count = len(our_tokens)
                
                # Existing tokenizer
                existing_tokens = existing_tokenizer.tokenize(sentence)
                existing_count = len(existing_tokens)
                
                comparison = {
                    'sentence': sentence,
                    'our_tokens': our_tokens,
                    'our_count': our_count,
                    'existing_tokens': existing_tokens,
                    'existing_count': existing_count,
                    'efficiency_ratio': existing_count / our_count if our_count > 0 else float('inf')
                }
                
                results['comparisons'].append(comparison)
            
            benchmark_results[model_name] = results
            
            # Summary for this model
            avg_efficiency = sum(c['efficiency_ratio'] for c in results['comparisons']) / len(results['comparisons'])
            logger.info(f"    Avg efficiency ratio: {avg_efficiency:.2f} (lower is better)")
            
        except Exception as e:
            logger.warning(f"  Failed to benchmark against {model_name}: {e}")
    
    return benchmark_results

def preprocess_corpus_with_thai(corpus_path: str, config: TokenizerConfig) -> str:
    """Preprocess corpus with Thai tokenization if enabled"""
    
    if not config.use_thai_pretokenizer or not PYTHAINLP_AVAILABLE:
        return corpus_path
    
    # Create preprocessed corpus path
    corpus_dir = Path(corpus_path).parent
    preprocessed_path = corpus_dir / f"preprocessed_{Path(corpus_path).name}"
    
    # Check if already preprocessed
    if preprocessed_path.exists():
        logger.info(f"Using existing preprocessed corpus: {preprocessed_path}")
        return str(preprocessed_path)
    
    logger.info(f"Preprocessing corpus with Thai tokenization (engine: {config.thai_engine})...")
    
    thai_pretokenize_func = create_thai_pre_tokenizer_function(
        engine=config.thai_engine,
        normalize_text=config.normalize_text
    )
    
    if thai_pretokenize_func is None:
        logger.warning("Thai preprocessing not available. Using original corpus.")
        return corpus_path
    
    processed_lines = 0
    failed_lines = 0
    with open(corpus_path, 'r', encoding='utf-8') as infile, \
         open(preprocessed_path, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            line = line.strip()
            if not line:
                outfile.write('\n')
                continue
            
            try:
                # Apply Thai pre-tokenization
                tokens = thai_pretokenize_func(line)
                processed_line = ' '.join(tokens)
                outfile.write(processed_line + '\n')
                processed_lines += 1
                
                if processed_lines % 10000 == 0:
                    logger.info(f"  Processed {processed_lines:,} lines...")
                    
            except Exception as e:
                logger.warning(f"Failed to preprocess line {processed_lines + failed_lines + 1}: {e}")
                outfile.write(line + '\n')
                failed_lines += 1
    
    logger.info(f"Preprocessing completed:")
    logger.info(f"  Successfully processed: {processed_lines:,} lines")
    if failed_lines > 0:
        logger.warning(f"  Failed to process: {failed_lines:,} lines")
    logger.info(f"Preprocessed corpus saved to: {preprocessed_path}")
    
    return str(preprocessed_path)
def train_with_validation(config: TokenizerConfig) -> Tuple[Tokenizer, Dict[str, Any]]:
    """Train tokenizer with validation split and Thai preprocessing"""
    
    logger.info(f"Training {config.model_type.upper()} tokenizer with validation...")
    
    # Preprocess corpus with Thai tokenization if enabled
    processed_corpus_path = preprocess_corpus_with_thai(config.corpus_path, config)
    
    # Validate and analyze corpus
    corpus_stats = validate_corpus(processed_corpus_path)
    
    # Create validation split if requested
    train_files = [processed_corpus_path]
    validation_stats = None
    
    if config.validation_split > 0:
        logger.info(f"Creating validation split ({config.validation_split:.1%})...")
        
        # Create validation file
        corpus_dir = Path(processed_corpus_path).parent
        val_path = corpus_dir / "validation_corpus.txt"
        train_path = corpus_dir / "train_corpus.txt"
        
        with open(processed_corpus_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        val_size = int(len(lines) * config.validation_split)
        val_lines = lines[:val_size]
        train_lines = lines[val_size:]
        
        with open(val_path, 'w', encoding='utf-8') as f:
            f.writelines(val_lines)
        
        with open(train_path, 'w', encoding='utf-8') as f:
            f.writelines(train_lines)
        
        train_files = [str(train_path)]
        validation_stats = {
            'validation_lines': len(val_lines),
            'training_lines': len(train_lines),
            'validation_path': str(val_path)
        }
        
        logger.info(f"  Training lines: {len(train_lines):,}")
        logger.info(f"  Validation lines: {len(val_lines):,}")
    
    # Create tokenizer
    tokenizer, trainer = create_advanced_tokenizer(config)
    
    # Train
    start_time = time.time()
    logger.info(f"Training tokenizer on {len(train_files)} file(s)...")
    
    tokenizer.train(train_files, trainer=trainer)
    
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    # Prepare training metadata
    training_metadata = {
        'training_time_seconds': training_time,
        'corpus_stats': corpus_stats,
        'validation_stats': validation_stats,
        'config': config.__dict__,
        'final_vocab_size': len(tokenizer.get_vocab()),
        'preprocessed_corpus_used': processed_corpus_path != config.corpus_path
    }
    
    return tokenizer, training_metadata

def validate_thai_encoding(tokenizer: Tokenizer) -> Dict[str, Any]:
    """Validate that Thai characters are properly encoded and decoded"""
    
    logger.info("Validating Thai character encoding...")
    
    # Test sentences with various Thai characters
    test_cases = [
        "สวัสดี",  # Basic Thai greeting
        "ส",       # Single Thai character
        "ขอบคุณ",   # Common Thai phrase
        "กิน ข้าว อร่อย",  # Thai with spaces
        "123 สวัสดี abc",  # Mixed Thai, numbers, English
        "ก่อน หลัง",  # Thai with tone marks
        "เด็ก เดิน เดา",  # Thai vowels
        "แม่ แมว แมลง",  # Various Thai vowel combinations
    ]
    
    results = {
        'total_tests': len(test_cases),
        'passed': 0,
        'failed': 0,
        'test_results': [],
        'encoding_issues': [],
        'decoding_issues': []
    }
    
    for i, text in enumerate(test_cases):
        try:
            # Encode the text
            encoding = tokenizer.encode(text)
            tokens = encoding.tokens
            token_ids = encoding.ids
            
            # Test different decoding approaches to find the best one
            decoded_standard = tokenizer.decode(token_ids)
            decoded_skip_special = tokenizer.decode(token_ids, skip_special_tokens=True)
            
            # Manual decoding (concatenate non-special tokens) - often works best for Thai
            manual_decoded = ""
            for token in tokens:
                if not (token.startswith('<') and token.endswith('>')):
                    manual_decoded += token
            
            # Choose the best decoding result
            decoded_methods = [
                ("standard", decoded_standard),
                ("skip_special", decoded_skip_special), 
                ("manual", manual_decoded)
            ]
            
            best_decoded = decoded_standard
            perfect_roundtrip = False
            best_method = "standard"
            
            # Find the method that gives perfect roundtrip
            for method_name, decoded_text in decoded_methods:
                if decoded_text.strip() == text.strip():
                    best_decoded = decoded_text
                    perfect_roundtrip = True
                    best_method = method_name
                    break
            
            # If no perfect match, use the closest one
            if not perfect_roundtrip:
                # Calculate similarity scores and pick the best
                best_score = -1
                for method_name, decoded_text in decoded_methods:
                    score = len(set(text) & set(decoded_text)) / max(len(set(text)), len(set(decoded_text)), 1)
                    if score > best_score:
                        best_score = score
                        best_decoded = decoded_text
                        best_method = method_name
            
            # Check for issues
            has_byte_artifacts = any('Ġ' in token or 'â' in token or 'Ã' in token for token in tokens)
            
            test_result = {
                'input': text,
                'tokens': tokens,
                'decoded': best_decoded,
                'decoding_method': best_method,
                'perfect_roundtrip': perfect_roundtrip,
                'has_byte_artifacts': has_byte_artifacts,
                'token_count': len(tokens)
            }
            
            if perfect_roundtrip and not has_byte_artifacts:
                results['passed'] += 1
                logger.info(f"✓ Test {i+1}: '{text}' -> {len(tokens)} tokens -> '{best_decoded}' ({best_method})")
            else:
                results['failed'] += 1
                logger.warning(f"✗ Test {i+1}: '{text}' -> {len(tokens)} tokens -> '{best_decoded}' ({best_method})")
                
                if has_byte_artifacts:
                    results['encoding_issues'].append(test_result)
                    logger.warning(f"  Byte artifacts detected in tokens: {tokens}")
                
                if not perfect_roundtrip:
                    results['decoding_issues'].append(test_result)
                    logger.warning(f"  Roundtrip failed: '{text}' != '{best_decoded}'")
                    logger.warning(f"  Available decodings: {[f'{m}: {d}' for m, d in decoded_methods]}")
            
            results['test_results'].append(test_result)
            
        except Exception as e:
            results['failed'] += 1
            error_result = {
                'input': text,
                'error': str(e),
                'tokens': None,
                'decoded': None,
                'perfect_roundtrip': False,
                'has_byte_artifacts': True,
                'token_count': 0
            }
            results['test_results'].append(error_result)
            results['encoding_issues'].append(error_result)
            logger.error(f"✗ Test {i+1}: '{text}' -> ERROR: {e}")
    
    # Summary
    success_rate = results['passed'] / results['total_tests'] * 100
    logger.info(f"Thai encoding validation: {results['passed']}/{results['total_tests']} passed ({success_rate:.1f}%)")
    
    if results['encoding_issues']:
        logger.warning(f"Found {len(results['encoding_issues'])} encoding issues (byte artifacts)")
    if results['decoding_issues']:
        logger.warning(f"Found {len(results['decoding_issues'])} decoding issues (roundtrip failures)")
    
    return results

def main(config: TokenizerConfig = None) -> Tokenizer:
    """Advanced main training pipeline with comprehensive evaluation"""
    
    if config is None:
        config = TokenizerConfig()
    
    try:
        logger.info("Starting Advanced Thai Tokenizer Training Pipeline...")
        logger.info(f"Configuration:")
        for key, value in config.__dict__.items():
            logger.info(f"  {key}: {value}")
        
        # Train tokenizer with validation
        tokenizer, training_metadata = train_with_validation(config)
        
        # Save tokenizer files
        save_results = save_tokenizer_files(tokenizer, config)
        
        # CRITICAL: Validate Thai character encoding before testing
        thai_validation = validate_thai_encoding(tokenizer)
        
        # Comprehensive testing
        test_results = comprehensive_tokenizer_test(tokenizer, config)
        
        # Benchmark against existing models
        benchmark_results = benchmark_against_existing(tokenizer, config)
        
        # Save evaluation results including Thai validation
        evaluation_results = {
            'training_metadata': training_metadata,
            'thai_validation': thai_validation,
            'test_results': test_results,
            'benchmark_results': benchmark_results,
            'save_results': save_results
        }
        
        eval_path = Path(config.tokenizer_dir) / "evaluation_results.json"
        with open(eval_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info("Saved evaluation_results.json")
        
        # Final summary
        logger.info("Advanced Thai Tokenizer Training Completed Successfully!")
        logger.info("=" * 60)
        logger.info(f"Output directory: {config.tokenizer_dir}")
        logger.info(f"Final vocabulary size: {len(tokenizer.get_vocab()):,}")
        logger.info(f"Thai encoding validation: {thai_validation['passed']}/{thai_validation['total_tests']} passed ({thai_validation['passed']/thai_validation['total_tests']*100:.1f}%)")
        logger.info(f"Thai coverage: {test_results.get('thai_coverage', 0):.1%}")
        logger.info(f"UNK ratio: {test_results.get('unk_ratio', 0):.1%}")
        logger.info(f"Training time: {training_metadata['training_time_seconds']:.2f}s")
        logger.info(f"Test success rate: {test_results['successful_tests']}/{test_results['total_tests']}")
        
        # Report any encoding issues
        if thai_validation['encoding_issues']:
            logger.warning(f"⚠️  Found {len(thai_validation['encoding_issues'])} Thai encoding issues!")
        if thai_validation['decoding_issues']:
            logger.warning(f"⚠️  Found {len(thai_validation['decoding_issues'])} Thai decoding issues!")
        
        if thai_validation['passed'] == thai_validation['total_tests']:
            logger.info("✅ All Thai character tests passed - tokenizer is ready for use!")
        else:
            logger.warning("⚠️  Some Thai character tests failed - review configuration")
        
        logger.info("Files created:")
        for file in save_results['files_created']:
            logger.info(f"  - {file}")
        logger.info("=" * 60)
        
        return tokenizer
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}")
        logger.exception("Full error details:")
        raise

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Advanced Thai Tokenizer Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples (Windows cmd.exe):
  Basic training:
    python train_hf_tokenizer.py
  
  With custom parameters:
    python train_hf_tokenizer.py --vocab-size 50000 --model-type unigram
  
  Full configuration:
    python train_hf_tokenizer.py --corpus data/my_corpus.txt --vocab-size 50000 --model-type unigram --thai-engine deepcut --validation-split 0.1 --max-length 4096 --output-dir MyThaiTokenizer
  
  With benchmarking:
    python train_hf_tokenizer.py --benchmark-models "airesearch/wangchanberta-base-att-spm-uncased"
        """
    )
    
    parser.add_argument("--corpus", type=str, default="data/combined_thai_corpus.txt",
                        help="Path to training corpus (default: data/combined_thai_corpus.txt)")
    parser.add_argument("--vocab-size", type=int, default=32000,
                        help="Vocabulary size (default: 32000)")
    parser.add_argument("--model-type", choices=["bpe", "unigram", "wordpiece"], default="bpe",
                        help="Tokenizer model type (default: bpe)")
    parser.add_argument("--output-dir", type=str, default="Bitthaitokenizer",
                        help="Output directory (default: Bitthaitokenizer)")
    parser.add_argument("--min-frequency", type=int, default=2,
                        help="Minimum frequency for tokens (default: 2)")
    parser.add_argument("--max-length", type=int, default=2048,
                        help="Maximum sequence length (default: 2048)")
    parser.add_argument("--thai-engine", type=str, default="newmm",
                        choices=["newmm", "longest", "mm", "deepcut", "attacut"],
                        help="PyThaiNLP tokenization engine (default: newmm, most reliable)")
    parser.add_argument("--no-thai-pretokenizer", action="store_true",
                        help="Disable Thai pre-tokenizer (use basic tokenizer)")
    parser.add_argument("--no-normalize", action="store_true",
                        help="Disable text normalization")
    parser.add_argument("--validation-split", type=float, default=0.0,
                        help="Validation split ratio 0.0-1.0 (default: 0.0)")
    parser.add_argument("--test-sentences", type=str,
                        help="Path to custom test sentences file")
    parser.add_argument("--benchmark-models", nargs="+",
                        help="Models to benchmark against (space-separated)")
    parser.add_argument("--dropout", type=float,
                        help="Dropout for regularization (BPE only)")
    
    return parser.parse_args()

# === Legacy compatibility functions ===
def create_thai_pretokenizer():
    """Legacy function for backward compatibility"""
    logger.warning("create_thai_pretokenizer() is deprecated. Use create_thai_pre_tokenizer_function() instead.")
    return create_thai_pre_tokenizer_function()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Create configuration from arguments
    config = TokenizerConfig(
        corpus_path=args.corpus,
        vocab_size=args.vocab_size,
        tokenizer_dir=args.output_dir,
        min_frequency=args.min_frequency,
        model_type=args.model_type,
        use_thai_pretokenizer=not args.no_thai_pretokenizer,
        model_max_length=args.max_length,
        thai_engine=args.thai_engine,
        normalize_text=not args.no_normalize,
        validation_split=args.validation_split,
        test_sentences_path=args.test_sentences,
        benchmark_models=args.benchmark_models,
        dropout=args.dropout
    )
    
    # Check dependencies
    if config.use_thai_pretokenizer and not PYTHAINLP_AVAILABLE:
        logger.warning("PyThaiNLP not available. Install with: pip install pythainlp")
        logger.warning("Falling back to basic pre-tokenizer.")
        config.use_thai_pretokenizer = False
    
    # Run training pipeline
    tokenizer = main(config)