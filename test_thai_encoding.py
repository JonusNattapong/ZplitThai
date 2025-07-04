#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick test script to validate Thai character encoding fixes
"""

import sys
import os
from pathlib import Path

# Add the project directory to the path
sys.path.append(str(Path(__file__).parent))

# Import our training module
from train_hf_tokenizer import TokenizerConfig, create_advanced_tokenizer, validate_thai_encoding
from tokenizers import Tokenizer

def quick_thai_test():
    """Quick test of Thai character handling"""
    
    print("🔍 Testing Thai Character Encoding Fixes...")
    print("=" * 50)
    
    # Create a minimal test configuration
    config = TokenizerConfig(
        corpus_path="data/combined_thai_corpus.txt",
        vocab_size=1000,  # Small vocab for quick test
        tokenizer_dir="test_output",
        model_type="unigram",
        enable_byte_fallback=False,  # CRITICAL: Disabled
        normalize_text=False,        # CRITICAL: Disabled
        use_thai_pretokenizer=False  # Disable for simplicity
    )
    
    print(f"Configuration:")
    print(f"  - Model type: {config.model_type}")
    print(f"  - Byte fallback: {config.enable_byte_fallback}")
    print(f"  - Text normalization: {config.normalize_text}")
    print(f"  - Thai pre-tokenizer: {config.use_thai_pretokenizer}")
    print()
    
    try:
        # Create tokenizer
        print("🔧 Creating tokenizer...")
        tokenizer, trainer = create_advanced_tokenizer(config)
        
        # Quick training with minimal data
        print("🏋️ Quick training...")
        
        # Create minimal training data
        test_corpus = [
            "สวัสดีครับ",
            "ขอบคุณมาก", 
            "สวัสดี โลก",
            "ผม ชื่อ จอห์น",
            "วันนี้ อากาศ ดี"
        ]
        
        # Train on minimal data
        tokenizer.train_from_iterator(test_corpus, trainer)
        
        print("✅ Training completed!")
        print()
        
        # Test Thai character handling
        print("🧪 Testing Thai character encoding...")
        validation_results = validate_thai_encoding(tokenizer)
        
        print()
        print("📊 Results Summary:")
        print(f"  - Tests passed: {validation_results['passed']}/{validation_results['total_tests']}")
        print(f"  - Success rate: {validation_results['passed']/validation_results['total_tests']*100:.1f}%")
        print(f"  - Encoding issues: {len(validation_results['encoding_issues'])}")
        print(f"  - Decoding issues: {len(validation_results['decoding_issues'])}")
        print()
        
        if validation_results['passed'] == validation_results['total_tests']:
            print("🎉 SUCCESS: All Thai character tests passed!")
            print("✅ Tokenizer configuration is working correctly!")
        else:
            print("⚠️  WARNING: Some tests failed!")
            print("❌ Configuration needs further adjustment")
            
            # Show failed test details
            for test in validation_results['test_results']:
                if not test['perfect_roundtrip'] or test['has_byte_artifacts']:
                    print(f"❌ Failed: '{test['input']}' -> {test['tokens']} -> '{test['decoded']}'")
        
        print("=" * 50)
        return validation_results['passed'] == validation_results['total_tests']
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = quick_thai_test()
    sys.exit(0 if success else 1)
