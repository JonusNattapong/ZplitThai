#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Focused test to debug Thai tokenizer decoding issues
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from train_hf_tokenizer import TokenizerConfig, create_advanced_tokenizer
from tokenizers import Tokenizer

def debug_tokenizer_decoding():
    """Debug the specific decoding issues we're seeing"""
    
    print("🔍 Debugging Thai Tokenizer Decoding...")
    print("=" * 50)
    
    # Create minimal config
    config = TokenizerConfig(
        vocab_size=1000,
        model_type="unigram",
        enable_byte_fallback=False,
        normalize_text=False,
        use_thai_pretokenizer=False
    )
    
    # Create tokenizer
    tokenizer, trainer = create_advanced_tokenizer(config)
    
    # Train on minimal data
    test_corpus = [
        "สวัสดี", "ขอบคุณ", "ส", "ก", "123", "abc", "hello",
        "กิน", "ข้าว", "อร่อย", "แม่", "แมว", "แมลง",
        "เด็ก", "เดิน", "เดา", "ก่อน", "หลัง",
        " ", "  ", "กิน ข้าว", "123 abc", "สวัสดี 123"
    ]
    
    print("🏋️ Training on test corpus...")
    tokenizer.train_from_iterator(test_corpus, trainer)
    print("✅ Training completed!")
    
    # Test cases that were failing
    test_cases = [
        "สวัสดี",           # Working
        "ส",               # Working  
        "ขอบคุณ",           # Working
        "กิน ข้าว อร่อย",    # Failing - spaces lost
        "123 สวัสดี abc",   # Failing - non-Thai lost
        "ก่อน หลัง",        # Failing - spaces lost
    ]
    
    print("\n🧪 Testing problematic cases:")
    print()
    
    for i, text in enumerate(test_cases):
        print(f"Test {i+1}: '{text}'")
        
        try:
            # Encode
            encoding = tokenizer.encode(text)
            tokens = encoding.tokens
            token_ids = encoding.ids
            
            print(f"  Tokens: {tokens}")
            print(f"  Token IDs: {token_ids}")
            
            # Try different decoding methods
            decoded1 = tokenizer.decode(token_ids)
            decoded2 = tokenizer.decode(token_ids, skip_special_tokens=True)
            
            print(f"  Decoded (with special): '{decoded1}'")
            print(f"  Decoded (no special):   '{decoded2}'")
            
            # Check which one is better
            match1 = (decoded1.strip() == text.strip())
            match2 = (decoded2.strip() == text.strip())
            
            print(f"  Match with special: {match1}")
            print(f"  Match no special:   {match2}")
            
            if match1 or match2:
                print(f"  ✅ SUCCESS (best: {'with special' if match1 else 'no special'})")
            else:
                print(f"  ❌ FAILED")
                
                # Debug: Show character-by-character comparison
                best_decoded = decoded2  # Usually no special tokens is better
                print(f"  Debug comparison:")
                print(f"    Original: {[c for c in text]}")
                print(f"    Decoded:  {[c for c in best_decoded]}")
                
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
        
        print()
    
    print("🔍 Additional debugging:")
    print()
    
    # Check vocabulary
    vocab = tokenizer.get_vocab()
    print(f"Vocabulary size: {len(vocab)}")
    
    # Check special tokens
    special_tokens_in_vocab = [token for token in vocab if token.startswith('<') and token.endswith('>')]
    print(f"Special tokens: {special_tokens_in_vocab}")
    
    # Check space handling
    space_tokens = [token for token in vocab if ' ' in token]
    print(f"Space-containing tokens: {space_tokens[:10]}...")
    
    # Check Thai characters
    thai_tokens = [token for token in vocab if any('\u0e00' <= char <= '\u0e7f' for char in token)]
    print(f"Thai tokens: {len(thai_tokens)} (sample: {thai_tokens[:10]})")
    
    return True

if __name__ == "__main__":
    debug_tokenizer_decoding()
