# Thai Tokenizer - UNIGRAM

Advanced Thai language tokenizer trained with HuggingFace tokenizers.

## Model Information
- **Model Type**: UNIGRAM
- **Vocabulary Size**: 2,010
- **Thai Tokens**: 1,984 (98.7%)
- **Special Tokens**: 22
- **Max Length**: 2048
- **Thai Engine**: newmm

## Usage

### With Transformers
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('FixedThaiTokenizer')
text = "สวัสดีครับ ผมชื่อจอห์น"
tokens = tokenizer.tokenize(text)
encoded = tokenizer(text, return_tensors="pt")
```

### Direct Usage
```python
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file('FixedThaiTokenizer/tokenizer.json')
text = "สวัสดีครับ ผมชื่อจอห์น"
encoding = tokenizer.encode(text)
print(encoding.tokens)
```

## Training Details
- **Corpus**: data/combined_thai_corpus.txt
- **Training Date**: 2025-07-02 20:39:30
- **Normalization**: Enabled
- **Byte Fallback**: Disabled

## Files
- `tokenizer.json`: Main tokenizer file
- `vocab.json`: Vocabulary mapping
- `tokenizer_config.json`: Configuration for transformers
- `training_config.json`: Training parameters
- `README.md`: This file

## License
Apache-2.0
