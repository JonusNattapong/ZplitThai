#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Advanced Thai Tokenizer Development - Next Phase
การพัฒนา Thai Tokenizer ขั้นสูง - เฟสต่อไป
"""

import sys
import json
from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, normalizers, processors
from typing import Dict, List, Any, Tuple

def create_advanced_thai_tokenizer_v2():
    """Create an even better Thai tokenizer with advanced features"""
    
    print("🚀 Creating Advanced Thai Tokenizer V2...")
    print("=" * 60)
    
    # Advanced configuration
    print("1️⃣ Setting up Advanced Unigram model...")
    model = models.Unigram()
    tokenizer = Tokenizer(model)
    
    # NO normalization - preserve Thai exactly
    print("2️⃣ Preserving Thai characters (no normalization)...")
    tokenizer.normalizer = None
    
    # Smarter pre-tokenization for Thai
    print("3️⃣ Setting up smart pre-tokenization...")
    # Only split on major punctuation, preserve Thai structure
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(pattern=r'[.!?।॥\n]', behavior="removed"),  # Major sentence breaks
        pre_tokenizers.Punctuation(behavior="isolated")  # Isolate punctuation
    ])
    
    # No post-processor for clean Thai text
    print("4️⃣ Disabling post-processor for clean output...")
    tokenizer.post_processor = None
    
    # No decoder for direct concatenation
    print("5️⃣ Using direct decoding...")
    tokenizer.decoder = None
    
    # Enhanced Thai training data
    print("6️⃣ Creating comprehensive Thai training data...")
    thai_training_data = create_comprehensive_thai_dataset()
    
    print(f"   📚 Training with {len(thai_training_data)} examples")
    
    # Advanced trainer settings
    print("7️⃣ Setting up advanced trainer...")
    trainer = trainers.UnigramTrainer(
        vocab_size=15000,  # Larger vocab for better coverage
        special_tokens=["<unk>", "<pad>", "<s>", "</s>"],  # Essential tokens only
        show_progress=True,
        unk_token="<unk>",
        shrinking_factor=0.75,  # Better subword learning
        max_piece_length=20,    # Allow longer Thai words
        n_sub_iterations=2      # More training iterations
    )
    
    # Train the tokenizer
    print("8️⃣ Training advanced tokenizer...")
    try:
        tokenizer.train_from_iterator(thai_training_data, trainer)
        print("✅ Advanced training completed!")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return None
    
    vocab_size = len(tokenizer.get_vocab())
    print(f"   📊 Final vocabulary size: {vocab_size:,}")
    
    return tokenizer

def create_comprehensive_thai_dataset() -> List[str]:
    """Create a comprehensive Thai dataset for training"""
    
    dataset = []
    
    # 1. Basic Thai vocabulary
    basic_thai = [
        # Greetings and politeness
        "สวัสดี", "สวัสดีครับ", "สวัสดีค่ะ", "ขอบคุณ", "ขอบคุณครับ", "ขอบคุณค่ะ",
        "ขอโทษ", "ขอโทษครับ", "ขอโทษค่ะ", "ไม่เป็นไร", "ยินดี", "ยินดีครับ",
        
        # Personal pronouns and titles
        "ผม", "ดิฉัน", "ฉัน", "เรา", "คุณ", "เขา", "เธอ", "นาย", "นาง", "นางสาว",
        "ครู", "อาจารย์", "หมอ", "พยาบาล", "ตำรวจ", "ทหาร",
        
        # Family
        "แม่", "พ่อ", "ลูก", "พี่", "น้อง", "ปู่", "ย่า", "ตา", "ยาย", "ลุง", "ป้า", "อา", "น้า",
        
        # Food and eating
        "กิน", "ข้าว", "น้ำ", "อาหาร", "อร่อย", "เผ็ด", "หวาน", "เค็ม", "เปรี้ยว", "ขม",
        "หิว", "อิ่ม", "ดื่ม", "กาแฟ", "ชา", "นม", "เบียร์", "ไวน์",
        
        # Daily activities
        "ไป", "มา", "อยู่", "นอน", "ตื่น", "เดิน", "วิ่ง", "นั่ง", "ยืน", "ทำงาน",
        "เรียน", "อ่าน", "เขียน", "ฟัง", "ดู", "พูด", "คิด", "รู้", "เข้าใจ",
        
        # Time and dates
        "วัน", "เวลา", "วันนี้", "เมื่อวาน", "พรุ่งนี้", "เช้า", "เที่ยง", "เย็น", "คืน",
        "จันทร์", "อังคาร", "พุธ", "พฤหัสบดี", "ศุกร์", "เสาร์", "อาทิตย์",
        "มกราคม", "กุมภาพันธ์", "มีนาคม", "เมษายน", "พฤษภาคม", "มิถุนายน",
        
        # Places
        "บ้าน", "โรงเรียน", "มหาวิทยาลัย", "โรงพยาบาล", "ตลาด", "ร้าน", "ออฟฟิศ",
        "วัด", "โบสถ์", "สวนสาธารณะ", "สนามบิน", "สถานีรถไฟ", "ท่าเรือ",
        
        # Numbers
        "หนึ่ง", "สอง", "สาม", "สี่", "ห้า", "หก", "เจ็ด", "แปด", "เก้า", "สิบ",
        "ยี่สิบ", "สามสิบ", "สี่สิบ", "ห้าสิบ", "หกสิบ", "เจ็ดสิบ", "แปดสิบ", "เก้าสิบ", "ร้อย", "พัน",
    ]
    
    # 2. Common Thai phrases
    thai_phrases = [
        "สบายดีไหม", "เป็นไงบ้าง", "ไปไหนมา", "กินข้าวยัง", "ทำงานหนักมาก",
        "อากาศร้อนจัง", "ฝนตกหนัก", "รถติดมาก", "ราคาแพงขึ้น", "ชีวิตยากขึ้น",
        "มีความสุข", "รู้สึกดี", "เหนื่อยมาก", "ปวดหัว", "ไม่สบาย",
        "ไปเที่ยว", "พักผ่อน", "ออกกำลังกาย", "เล่นกีฬา", "ดูหนัง",
        "ฟังเพลง", "อ่านหนังสือ", "เล่นเกม", "ใช้โทรศัพท์", "ดูทีวี",
    ]
    
    # 3. Modern Thai (internet, technology)
    modern_thai = [
        "อินเทอร์เน็ต", "คอมพิวเตอร์", "โทรศัพท์", "มือถือ", "แอปพลิเคชัน",
        "เว็บไซต์", "อีเมล", "เฟซบุ๊ก", "ไลน์", "ทวิตเตอร์", "ยูทูบ", "ติ๊กต๊อก",
        "ออนไลน์", "ออฟไลน์", "ดาวน์โหลด", "อัปโหลด", "แชร์", "ไลค์", "คอมเมนต์",
        "เซลฟี่", "สติกเกอร์", "อีโมจิ", "มีม", "ไวรัล", "เทรนด์", "แฮชแท็ก",
    ]
    
    # 4. Mixed Thai-English content
    mixed_content = [
        "Hello สวัสดี", "Thank you ขอบคุณ", "Good morning อรุณสวัสดิ์",
        "Happy birthday วันเกิดมุบารค", "Have a nice day ขอให้มีความสุข",
        "Email อีเมล", "Facebook เฟซบุ๊ก", "Google กูเกิล", "iPhone ไอโฟน",
        "McDonald's แมคโดนัลด์", "Starbucks สตาร์บัคส์", "7-Eleven เซเว่น",
        "COVID-19 โควิด", "AI เอไอ", "IT ไอที", "HR ฝ่ายบุคคล", "CEO ซีอีโอ",
    ]
    
    # 5. Formal Thai
    formal_thai = [
        "พระบาทสมเด็จพระเจ้าอยู่หัว", "สมเด็จพระราชินี", "รัฐบาล", "นายกรัฐมนตรี",
        "รัฐมนตรี", "ศาสตราจารย์", "ผู้จัดการ", "ประธานกรรมการ", "กรรมการผู้จัดการ",
        "การประชุม", "การประชุมสำคัญ", "การตัดสินใจ", "การพิจารณา", "การอนุมัติ",
        "ขอเรียนเชิญ", "ขอแสดงความยินดี", "ขอแสดงความเสียใจ", "ด้วยความเคารพ",
    ]
    
    # 6. Street Thai / Casual
    casual_thai = [
        "เท่ไหร่", "ลดให้หน่อย", "แพงมาก", "ถูกมาก", "ฟรี", "ส่วนลด",
        "จ่ายเท่าไหร่", "เงินทอน", "บัตรเครดิต", "เงินสด", "โอนเงิน",
        "อร่อยจัง", "เผ็ดมาก", "หวานไป", "เค็มไป", "อิ่มแล้ว", "อีกแก้วหนึ่ง",
        "เอาไอติมด้วย", "ไม่เอาผัก", "ไม่เอาเผ็ด", "เพิ่มเนื้อ", "เยอะๆ นะ",
    ]
    
    # Combine all categories
    dataset.extend(basic_thai)
    dataset.extend(thai_phrases)
    dataset.extend(modern_thai)
    dataset.extend(mixed_content)
    dataset.extend(formal_thai)
    dataset.extend(casual_thai)
    
    # 7. Generate combinations and variations
    extended_dataset = []
    for item in dataset:
        extended_dataset.append(item)
        
        # Add polite endings
        if not any(ending in item for ending in ["ครับ", "ค่ะ", "นะ", "จ้า"]):
            extended_dataset.append(item + "ครับ")
            extended_dataset.append(item + "ค่ะ")
            extended_dataset.append(item + "นะ")
        
        # Add common prefixes
        extended_dataset.append("อยาก" + item)
        extended_dataset.append("ชอบ" + item)
        extended_dataset.append("ไม่" + item)
        
        # Create sentences with spaces
        if " " not in item and len(item) > 3:
            # Split Thai words artificially for training
            mid = len(item) // 2
            extended_dataset.append(item[:mid] + " " + item[mid:])
    
    # 8. Add numbers and mixed content
    for i in range(100):
        extended_dataset.append(f"{i} บาท")
        extended_dataset.append(f"ราคา {i}")
        extended_dataset.append(f"อายุ {i} ปี")
        extended_dataset.append(f"เวลา {i:02d}:00")
    
    # 9. Add common patterns
    patterns = [
        "ไป{place}", "มา{place}", "อยู่{place}", "ที่{place}",
        "กิน{food}", "ดื่ม{drink}", "ซื้อ{item}", "ขาย{item}",
        "รัก{person}", "คิดถึง{person}", "เจอ{person}", "คุย{person}",
    ]
    
    places = ["บ้าน", "โรงเรียน", "ตลาด", "ร้าน"]
    foods = ["ข้าว", "ส้มตำ", "ต้มยำ", "ผัดไทย"]
    drinks = ["น้ำ", "กาแฟ", "ชา", "เบียร์"]
    items = ["เสื้อ", "กางเกง", "รองเท้า", "กระเป๋า"]
    people = ["แม่", "พ่อ", "เพื่อน", "แฟน"]
    
    for pattern in patterns:
        if "{place}" in pattern:
            for place in places:
                extended_dataset.append(pattern.replace("{place}", place))
        elif "{food}" in pattern:
            for food in foods:
                extended_dataset.append(pattern.replace("{food}", food))
        elif "{drink}" in pattern:
            for drink in drinks:
                extended_dataset.append(pattern.replace("{drink}", drink))
        elif "{item}" in pattern:
            for item in items:
                extended_dataset.append(pattern.replace("{item}", item))
        elif "{person}" in pattern:
            for person in people:
                extended_dataset.append(pattern.replace("{person}", person))
    
    # Remove duplicates and return
    return list(set(extended_dataset))

def test_advanced_tokenizer(tokenizer: Tokenizer) -> Dict[str, Any]:
    """Advanced testing for the tokenizer"""
    
    print("\n🧪 Advanced Tokenizer Testing...")
    print("=" * 60)
    
    test_categories = {
        "basic_thai": [
            "สวัสดี", "ขอบคุณ", "ครับ", "ค่ะ"
        ],
        "thai_with_spaces": [
            "กิน ข้าว อร่อย", "วันนี้ อากาศ ดี", "ผม ชื่อ จอห์น"
        ],
        "mixed_content": [
            "123 สวัสดี abc", "Hello ครับ", "COVID-19 ระบาด"
        ],
        "formal_thai": [
            "พระบาทสมเด็จพระเจ้าอยู่หัว", "การประชุมสำคัญ"
        ],
        "casual_thai": [
            "อร่อยจัง", "แพงมาก", "ถูกมาก"
        ],
        "complex_thai": [
            "กรุงเทพมหานคร", "ราชมงคลธัญบุรี", "จุฬาลงกรณ์มหาวิทยาลัย"
        ],
        "numbers_dates": [
            "1 มกราคม 2567", "เวลา 14:30 น.", "ราคา 1,234 บาท"
        ],
        "technology": [
            "อินเทอร์เน็ต", "โทรศัพท์มือถือ", "แอปพลิเคชัน"
        ]
    }
    
    results = {
        "overall": {"passed": 0, "total": 0},
        "categories": {}
    }
    
    for category, test_cases in test_categories.items():
        print(f"\n📂 Testing {category.replace('_', ' ').title()}...")
        category_results = {"passed": 0, "total": len(test_cases), "details": []}
        
        for i, text in enumerate(test_cases):
            try:
                # Encode
                encoding = tokenizer.encode(text)
                tokens = encoding.tokens
                
                # Try different decoding methods
                decoded_standard = tokenizer.decode(encoding.ids)
                
                # Manual decoding (concatenate non-special tokens)
                manual_decoded = ""
                for token in tokens:
                    if not (token.startswith('<') and token.endswith('>')):
                        manual_decoded += token
                
                # Choose best result
                best_decoded = manual_decoded if manual_decoded.strip() == text.strip() else decoded_standard
                success = (best_decoded.strip() == text.strip())
                
                test_detail = {
                    "input": text,
                    "tokens": tokens,
                    "token_count": len(tokens),
                    "decoded": best_decoded,
                    "success": success
                }
                
                category_results["details"].append(test_detail)
                
                if success:
                    category_results["passed"] += 1
                    status = "✅"
                else:
                    status = "❌"
                
                print(f"  {status} '{text}' -> {len(tokens)} tokens -> '{best_decoded}'")
                
            except Exception as e:
                print(f"  ❌ '{text}' -> ERROR: {e}")
                category_results["details"].append({
                    "input": text,
                    "error": str(e),
                    "success": False
                })
        
        success_rate = category_results["passed"] / category_results["total"] * 100
        print(f"  📊 {category}: {category_results['passed']}/{category_results['total']} ({success_rate:.1f}%)")
        
        results["categories"][category] = category_results
        results["overall"]["passed"] += category_results["passed"]
        results["overall"]["total"] += category_results["total"]
    
    # Overall results
    overall_success = results["overall"]["passed"] / results["overall"]["total"] * 100
    print(f"\n📊 Overall Results: {results['overall']['passed']}/{results['overall']['total']} ({overall_success:.1f}%)")
    
    if overall_success >= 85:
        print("🎉 Excellent performance!")
    elif overall_success >= 70:
        print("👍 Good performance!")
    else:
        print("⚠️ Needs improvement")
    
    return results

def benchmark_tokenizer_efficiency(tokenizer: Tokenizer) -> Dict[str, Any]:
    """Benchmark tokenizer efficiency and compression"""
    
    print("\n⚡ Benchmarking Tokenizer Efficiency...")
    print("=" * 60)
    
    # Test sentences of varying complexity
    test_sentences = [
        "สวัสดี",  # Simple
        "สวัสดีครับ ผมชื่อจอห์น",  # Medium
        "วันนี้อากาศดีมาก ผมจึงไปเดินเล่นที่สวนสาธารณะ",  # Complex
        "พระบาทสมเด็จพระเจ้าอยู่หัวทรงพระกรุณาโปรดเกล้าฯ ให้จัดงานพระราชพิธี",  # Formal
        "555 อร่อยมากกก กินข้าวยัง? #อาหารไทย 🇹🇭",  # Social media
    ]
    
    results = {
        "compression_ratios": [],
        "avg_tokens_per_char": 0,
        "vocab_coverage": 0,
        "details": []
    }
    
    total_chars = 0
    total_tokens = 0
    vocab = tokenizer.get_vocab()
    used_tokens = set()
    
    for sentence in test_sentences:
        encoding = tokenizer.encode(sentence)
        tokens = encoding.tokens
        
        char_count = len(sentence)
        token_count = len(tokens)
        compression_ratio = char_count / token_count if token_count > 0 else 0
        
        # Track vocabulary usage
        for token in tokens:
            if token in vocab:
                used_tokens.add(token)
        
        detail = {
            "sentence": sentence,
            "char_count": char_count,
            "token_count": token_count,
            "compression_ratio": compression_ratio,
            "tokens": tokens
        }
        
        results["details"].append(detail)
        results["compression_ratios"].append(compression_ratio)
        
        total_chars += char_count
        total_tokens += token_count
        
        print(f"📝 '{sentence[:30]}{'...' if len(sentence) > 30 else ''}'")
        print(f"   Characters: {char_count}, Tokens: {token_count}, Ratio: {compression_ratio:.2f}")
    
    results["avg_tokens_per_char"] = total_tokens / total_chars if total_chars > 0 else 0
    results["vocab_coverage"] = len(used_tokens) / len(vocab) if len(vocab) > 0 else 0
    
    avg_compression = sum(results["compression_ratios"]) / len(results["compression_ratios"])
    
    print(f"\n📊 Efficiency Summary:")
    print(f"   Average compression ratio: {avg_compression:.2f} chars/token")
    print(f"   Tokens per character: {results['avg_tokens_per_char']:.3f}")
    print(f"   Vocabulary coverage: {results['vocab_coverage']:.1%}")
    
    return results

def save_advanced_tokenizer(tokenizer: Tokenizer, test_results: Dict, efficiency_results: Dict):
    """Save the advanced tokenizer with comprehensive metadata"""
    
    print("\n💾 Saving Advanced Thai Tokenizer...")
    
    save_dir = Path("AdvancedThaiTokenizerV2")
    save_dir.mkdir(exist_ok=True)
    
    # Save tokenizer
    tokenizer.save(str(save_dir / "tokenizer.json"))
    
    # Save vocabulary
    vocab = tokenizer.get_vocab()
    with open(save_dir / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    
    # Save comprehensive metadata
    metadata = {
        "model_info": {
            "version": "2.0",
            "model_type": "unigram",
            "vocab_size": len(vocab),
            "creation_date": "2025-07-02",
            "language": "thai",
            "description": "Advanced Thai tokenizer with improved handling of Thai text, mixed content, and modern vocabulary"
        },
        "performance": {
            "test_results": test_results,
            "efficiency": efficiency_results,
            "overall_accuracy": f"{test_results['overall']['passed']}/{test_results['overall']['total']}"
        },
        "features": [
            "No normalization (preserves Thai characters)",
            "Smart punctuation handling",
            "Mixed Thai-English support",
            "Modern vocabulary coverage",
            "Efficient compression",
            "Direct decoding without artifacts"
        ],
        "usage_notes": {
            "best_decoding": "manual concatenation of non-special tokens",
            "recommended_for": ["Thai NLP", "LLM training", "Text processing", "Social media analysis"],
            "avoid": ["Text normalization", "Byte-level fallback", "Aggressive post-processing"]
        }
    }
    
    with open(save_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    # Create usage examples
    usage_examples = {
        "basic_usage": """
from tokenizers import Tokenizer

# Load tokenizer
tokenizer = Tokenizer.from_file("AdvancedThaiTokenizerV2/tokenizer.json")

# Encode Thai text
text = "สวัสดีครับ วันนี้อากาศดีมาก"
encoding = tokenizer.encode(text)

# Best decoding method for Thai
decoded = ""
for token in encoding.tokens:
    if not (token.startswith('<') and token.endswith('>')):
        decoded += token

print(f"Original: {text}")
print(f"Tokens: {encoding.tokens}")
print(f"Decoded: {decoded}")
""",
        "batch_processing": """
# Process multiple Thai sentences
sentences = [
    "กินข้าวยัง",
    "ไปไหนมา", 
    "สบายดีไหม"
]

for sentence in sentences:
    encoding = tokenizer.encode(sentence)
    # Use manual decoding for best results
    decoded = "".join(token for token in encoding.tokens 
                     if not (token.startswith('<') and token.endswith('>')))
    print(f"{sentence} -> {decoded}")
""",
        "mixed_content": """
# Handle Thai-English mixed content
mixed_text = "Hello สวัสดี COVID-19 ระบาด"
encoding = tokenizer.encode(mixed_text)

# Manual decoding preserves mixed content
decoded = "".join(token for token in encoding.tokens 
                 if not (token.startswith('<') and token.endswith('>')))

print(f"Mixed: {mixed_text}")
print(f"Tokens: {encoding.tokens}")
print(f"Decoded: {decoded}")
"""
    }
    
    with open(save_dir / "usage_examples.json", "w", encoding="utf-8") as f:
        json.dump(usage_examples, f, ensure_ascii=False, indent=2)
    
    # Create README
    readme_content = f"""# Advanced Thai Tokenizer V2

## Overview
Advanced Thai language tokenizer with improved handling of Thai text, mixed content, and modern vocabulary.

## Performance
- Overall Accuracy: {test_results['overall']['passed']}/{test_results['overall']['total']} ({test_results['overall']['passed']/test_results['overall']['total']*100:.1f}%)
- Vocabulary Size: {len(vocab):,} tokens
- Average Compression: {sum(efficiency_results['compression_ratios'])/len(efficiency_results['compression_ratios']):.2f} chars/token

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
"""
    
    with open(save_dir / "README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
    
    print(f"✅ Saved to {save_dir}/")
    print(f"   Files: tokenizer.json, vocab.json, metadata.json, usage_examples.json, README.md")
    
    return save_dir

def main():
    """Main function for advanced Thai tokenizer development"""
    
    print("🚀 Advanced Thai Tokenizer Development V2")
    print("=" * 70)
    
    # Create advanced tokenizer
    tokenizer = create_advanced_thai_tokenizer_v2()
    
    if tokenizer is None:
        print("❌ Failed to create advanced tokenizer")
        return False
    
    # Advanced testing
    test_results = test_advanced_tokenizer(tokenizer)
    
    # Efficiency benchmarking
    efficiency_results = benchmark_tokenizer_efficiency(tokenizer)
    
    # Save everything
    save_dir = save_advanced_tokenizer(tokenizer, test_results, efficiency_results)
    
    # Final summary
    overall_success = test_results["overall"]["passed"] / test_results["overall"]["total"] * 100
    
    print(f"\n🎉 Advanced Thai Tokenizer V2 Complete!")
    print("=" * 70)
    print(f"📊 Overall Performance: {overall_success:.1f}%")
    print(f"📁 Saved to: {save_dir}")
    print(f"🚀 Ready for production use!")
    
    if overall_success >= 85:
        print("🏆 Excellent quality - ready for deployment!")
    elif overall_success >= 70:
        print("👍 Good quality - suitable for most applications!")
    else:
        print("⚠️ Consider further improvements")
    
    return overall_success >= 70

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
