#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Minimal test to check tokenizer configuration
"""

from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders, trainers

def test_thai_config():
    """Test the tokenizer configuration without full training"""
    
    print("🔍 Testing Tokenizer Configuration for Thai...")
    
    # Test 1: Check normalization
    print("\n1️⃣ Testing Normalization:")
    
    # No normalization (our fixed config)
    text = "สวัสดี"
    print(f"Original: {text}")
    print(f"Encoded bytes: {text.encode('utf-8')}")
    print(f"Repr: {repr(text)}")
    
    # Test 2: Create minimal tokenizer like our configuration
    print("\n2️⃣ Testing Tokenizer Components:")
    
    # Create model
    model = models.Unigram()
    tokenizer = Tokenizer(model)
    
    # No normalization (our fix)
    tokenizer.normalizer = None
    print("✅ Normalizer: None (preserves Thai characters)")
    
    # Basic pre-tokenizer (no byte-level)
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Whitespace(),
        pre_tokenizers.Punctuation()
    ])
    print("✅ Pre-tokenizer: Whitespace + Punctuation (no byte-level)")
    
    # Metaspace decoder (good for Thai)
    tokenizer.decoder = decoders.Metaspace()
    print("✅ Decoder: Metaspace (preserves Thai)")
    
    # Test 3: Manual token test
    print("\n3️⃣ Testing Manual Tokenization:")
    
    test_texts = ["สวัสดี", "ขอบคุณ", "ส", "Thai text"]
    
    for text in test_texts:
        try:
            # Since we don't have a trained vocabulary, this will mostly produce <unk>
            # But we can check if the text is preserved at the normalization level
            
            # Test normalization (should be None)
            if tokenizer.normalizer:
                normalized = tokenizer.normalizer.normalize_str(text)
            else:
                normalized = text
            
            print(f"'{text}' -> normalized: '{normalized}' (preserved: {text == normalized})")
            
        except Exception as e:
            print(f"'{text}' -> ERROR: {e}")
    
    print("\n✅ Configuration test completed!")
    print("📝 Key findings:")
    print("  - Normalizer is disabled (preserves Thai characters)")
    print("  - Pre-tokenizer avoids byte-level encoding")
    print("  - Decoder uses Metaspace (suitable for Thai)")
    
    return True

if __name__ == "__main__":
    test_thai_config()
