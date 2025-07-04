# Advanced Thai Tokenizer V2

## Overview
Advanced Thai language tokenizer with improved handling of Thai text, mixed content, and modern vocabulary.

## Performance
- Overall Accuracy: 24/24 (100.0%)
- Vocabulary Size: 1,223 tokens
- Average Compression: 2.91 chars/token

## Key Features
- ✅ No Thai character corruption
- ✅ Handles mixed Thai-English content
- ✅ Modern vocabulary (internet, technology terms)
- ✅ Efficient compression
- ✅ Clean decoding without artifacts

## Quick Start
```python
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("tokenizer.json")
text = "สวัสดีครับ วันนี้อากาศดีมาก"
encoding = tokenizer.encode(text)

# Best decoding method
decoded = "".join(token for token in encoding.tokens 
                 if not (token.startswith('<') and token.endswith('>')))
```

## Files
- `tokenizer.json` - Main tokenizer file
- `vocab.json` - Vocabulary mapping
- `metadata.json` - Performance and configuration details
- `usage_examples.json` - Code examples
- `README.md` - This file

Created: July 2025
