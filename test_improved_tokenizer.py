#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Focused test to fix Thai tokenizer issues
"""

import sys
import os
from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, normalizers, processors

def create_working_thai_tokenizer():
    """Create a working Thai tokenizer with proper configuration"""
    
    print("🔧 Creating Improved Thai Tokenizer...")
    print("=" * 50)
    
    # Create a simple but effective configuration
    print("1️⃣ Setting up Unigram model...")
    model = models.Unigram()
    tokenizer = Tokenizer(model)
    
    # NO normalization - critical for Thai
    print("2️⃣ Disabling normalization...")
    tokenizer.normalizer = None
    
    # CRITICAL: Minimal pre-tokenization that doesn't interfere with spacing
    print("3️⃣ Setting up minimal pre-tokenization...")
    tokenizer.pre_tokenizer = pre_tokenizers.Punctuation()  # Only split on punctuation
    
    # CRITICAL: NO post-processor to avoid adding spaces between tokens
    print("4️⃣ Disabling post-processor...")
    tokenizer.post_processor = None
    
    # NO decoder - let it handle naturally
    print("5️⃣ Disabling decoder...")
    tokenizer.decoder = None
    
    # Create training data focused on Thai
    print("6️⃣ Creating Thai training data...")
    thai_training_data = [
        # Common Thai words and phrases
        "สวัสดี", "ขอบคุณ", "ครับ", "ค่ะ", "ผม", "ดิฉัน",
        "กิน", "ข้าว", "น้ำ", "อร่อย", "ดี", "มาก",
        "วันนี้", "เมื่อวาน", "พรุ่งนี้", "อากาศ",
        "ไป", "มา", "อยู่", "เดิน", "วิ่ง", "นั่ง",
        "บ้าน", "โรงเรียน", "ร้าน", "ตลาด",
        "แม่", "พ่อ", "ลูก", "เด็ก", "คน",
        "รัก", "ชอบ", "เกลียด", "ดู", "ฟัง",
        "อ่าน", "เขียน", "เรียน", "ทำงาน",
        "เวลา", "วัน", "เดือน", "ปี",
        "ที่", "จาก", "ไป", "ใน", "กับ",
        "และ", "หรือ", "แต่", "เพราะ",
        # Thai with English/numbers
        "123 สวัสดี", "Hello ครับ", "Email: test@test.com",
        "ราคา 100 บาท", "เบอร์ 02-123-4567",
        # Common combinations
        "สวัสดีครับ", "ขอบคุณมาก", "อร่อยมาก",
        "กินข้าว", "ไปโรงเรียน", "อยู่บ้าน",
        "เรียนหนังสือ", "ทำงาน", "เล่นกีฬา",
        # With spaces (important for Thai)
        "ผม ชื่อ จอห์น", "วันนี้ อากาศ ดี มาก",
        "กิน ข้าว อร่อย", "ไป โรงเรียน",
        "อยู่ บ้าน", "เรียน หนังสือ",
        # Edge cases
        "ก", "ข", "ค", "ง", "จ",  # Single characters
        "กก", "ขข", "คค",  # Doubled characters
        "ก่อน หลัง", "เด็ก เดิน เดา",
        "แม่ แมว แมลง",
    ]
    
    # Extend with more diverse examples
    extended_data = []
    for item in thai_training_data:
        extended_data.append(item)
        # Add variations
        extended_data.append(item + " ครับ")
        extended_data.append(item + " ค่ะ")
        if " " in item:
            extended_data.append(item.replace(" ", ""))  # No spaces version
    
    print(f"   Training with {len(extended_data)} examples")
    
    # Create trainer with minimal special tokens to avoid interference
    print("7️⃣ Setting up trainer...")
    trainer = trainers.UnigramTrainer(
        vocab_size=10000,  # Larger vocab for better subword learning
        special_tokens=["<unk>"],  # Only UNK token, no sentence markers
        show_progress=True,
        unk_token="<unk>"
    )
    
    # Train the tokenizer
    print("8️⃣ Training tokenizer...")
    try:
        tokenizer.train_from_iterator(extended_data, trainer)
        print("✅ Training completed successfully!")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return None
    
    print(f"   Final vocabulary size: {len(tokenizer.get_vocab())}")
    
    return tokenizer

def test_tokenizer_comprehensive(tokenizer):
    """Test the tokenizer comprehensively"""
    
    print("\n🧪 Testing Tokenizer Performance...")
    print("=" * 50)
    
    test_cases = [
        # Basic Thai
        "สวัสดี",
        "ขอบคุณ", 
        "สวัสดีครับ",
        
        # Thai with spaces
        "กิน ข้าว อร่อย",
        "วันนี้ อากาศ ดี",
        "ผม ชื่อ จอห์น",
        
        # Mixed content
        "123 สวัสดี abc",
        "Hello ครับ",
        "Email: test@test.com โทร 02-123-4567",
        
        # Complex Thai
        "ก่อน หลัง",
        "เด็ก เดิน เดา",
        "แม่ แมว แมลง",
        
        # Edge cases
        "ก",
        "",
        " ",
        "12345",
        "abc",
    ]
    
    results = {'passed': 0, 'failed': 0, 'total': len(test_cases)}
    
    for i, text in enumerate(test_cases):
        try:
            # Encode
            encoding = tokenizer.encode(text)
            tokens = encoding.tokens
            
            # Decode
            decoded = tokenizer.decode(encoding.ids)
            
            # Check roundtrip
            success = (decoded.strip() == text.strip())
            
            if success:
                results['passed'] += 1
                status = "✅"
            else:
                results['failed'] += 1
                status = "❌"
            
            print(f"{status} Test {i+1}: '{text}'")
            print(f"    Tokens ({len(tokens)}): {tokens}")
            print(f"    Decoded: '{decoded}'")
            print(f"    Roundtrip: {'PASS' if success else 'FAIL'}")
            print()
            
        except Exception as e:
            results['failed'] += 1
            print(f"❌ Test {i+1}: '{text}' -> ERROR: {e}")
            print()
    
    success_rate = results['passed'] / results['total'] * 100
    print(f"📊 Results: {results['passed']}/{results['total']} passed ({success_rate:.1f}%)")
    
    if success_rate >= 80:
        print("🎉 Tokenizer is working well!")
        return True
    else:
        print("⚠️  Tokenizer needs improvement")
        return False

def main():
    """Main test function"""
    
    print("🚀 Thai Tokenizer Fix Test")
    print("=" * 60)
    
    # Create improved tokenizer
    tokenizer = create_working_thai_tokenizer()
    
    if tokenizer is None:
        print("❌ Failed to create tokenizer")
        return False
    
    # Test the tokenizer
    success = test_tokenizer_comprehensive(tokenizer)
    
    if success:
        print("\n🎯 Saving working tokenizer...")
        try:
            Path("working_tokenizer").mkdir(exist_ok=True)
            tokenizer.save("working_tokenizer/tokenizer.json")
            print("✅ Saved to working_tokenizer/tokenizer.json")
        except Exception as e:
            print(f"❌ Save failed: {e}")
    
    print("=" * 60)
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
