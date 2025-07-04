#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compare and benchmark different Thai tokenizer versions
เปรียบเทียบและวัดประสิทธิภาพ tokenizer รุ่นต่างๆ
"""

import json
from pathlib import Path
from tokenizers import Tokenizer
from transformers import AutoTokenizer
import time
import sys

def load_tokenizers():
    """Load all available tokenizer versions"""
    tokenizers = {}
    
    # Load Advanced V2
    adv_v2_path = Path("AdvancedThaiTokenizerV2/tokenizer.json")
    if adv_v2_path.exists():
        tokenizers["Advanced V2"] = Tokenizer.from_file(str(adv_v2_path))
        print("✅ Loaded Advanced V2")
    
    # Load Final Thai Tokenizer
    final_path = Path("FinalThaiTokenizer")
    if final_path.exists():
        try:
            tokenizers["Final"] = AutoTokenizer.from_pretrained(str(final_path))
            print("✅ Loaded Final Thai Tokenizer")
        except Exception as e:
            print(f"❌ Failed to load Final: {e}")
    
    # Load Fixed V2
    fixed_v2_path = Path("FixedThaiTokenizerV2")
    if fixed_v2_path.exists():
        try:
            tokenizers["Fixed V2"] = AutoTokenizer.from_pretrained(str(fixed_v2_path))
            print("✅ Loaded Fixed V2")
        except Exception as e:
            print(f"❌ Failed to load Fixed V2: {e}")
    
    return tokenizers

def test_real_world_content():
    """Test with real-world mixed Thai-English content"""
    test_cases = [
        # Social media posts
        "วันนี้ไป Starbucks กับเพื่อน ดื่ม Green Tea Latte อร่อยมาก! 😊 #coffee #thailand",
        
        # News headlines
        "Apple เปิดตัว iPhone 15 Pro Max ราคาเริ่มต้น 44,900 บาท ขายในไทย 22 กันยายน 2567",
        
        # Business communication
        "Meeting วันจันทร์ เวลา 14:00 น. ที่ห้องประชุม A พูดคุยเรื่อง Q4 Strategy และ Budget 2025",
        
        # Technical documentation
        "การติดตั้ง Python 3.11 บน Windows 10/11 โดยใช้ pip install pandas numpy matplotlib",
        
        # Food reviews
        "ร้าน McDonald's สาขาใหม่ที่ Siam Paragon มี Big Mac Set ราคา 189 บาท รสชาติดีเหมือนเดิม",
        
        # Travel content
        "Flight TG123 จาก Bangkok (BKK) ไป Tokyo (NRT) เวลา 23:55 น. ระยะเวลาบิน 6 ชั่วโมง 30 นาที",
        
        # Academic text
        "งานวิจัย Machine Learning สำหรับ Natural Language Processing ในภาษาไทย โดย Prof. สมชาย",
        
        # Gaming
        "เล่น FIFA 24 กับเพื่อน ทีม Real Madrid vs Barcelona สกอร์ 3-2 มันส์มาก!",
        
        # Shopping
        "ซื้อ MacBook Pro 16\" M3 Max ราคา 89,900 บาท ผ่อน 0% 10 เดือน ที่ Apple Store Central World",
        
        # Government
        "กระทรวงสาธารณสุข ออกประกาศ COVID-19 ระลอกใหม่ ให้ประชาชนใส่หน้ากากอนามัยในที่แออัด"
    ]
    
    return test_cases

def decode_tokens_v2(tokenizer, encoding):
    """Manual decoding for Advanced V2 tokenizer"""
    if hasattr(encoding, 'tokens'):
        # Filter out special tokens
        filtered_tokens = [token for token in encoding.tokens 
                          if not (token.startswith('<') and token.endswith('>'))]
        return "".join(filtered_tokens)
    return str(encoding)

def benchmark_tokenizer(name, tokenizer, test_cases):
    """Benchmark a tokenizer on test cases"""
    results = {
        "name": name,
        "total_time": 0,
        "cases": [],
        "stats": {
            "avg_tokens": 0,
            "avg_ratio": 0,
            "total_chars": 0,
            "total_tokens": 0
        }
    }
    
    print(f"\n🧪 Testing {name}...")
    print("=" * 50)
    
    start_time = time.time()
    
    for i, text in enumerate(test_cases, 1):
        case_start = time.time()
        
        try:
            if name == "Advanced V2":
                # Use tokenizers library
                encoding = tokenizer.encode(text)
                tokens = encoding.tokens
                decoded = decode_tokens_v2(tokenizer, encoding)
                token_count = len(tokens)
            else:
                # Use HuggingFace tokenizer
                encoding = tokenizer.encode(text, add_special_tokens=False)
                tokens = tokenizer.convert_ids_to_tokens(encoding)
                decoded = tokenizer.decode(encoding, skip_special_tokens=True)
                token_count = len(tokens)
            
            case_time = time.time() - case_start
            char_count = len(text)
            ratio = char_count / token_count if token_count > 0 else 0
            
            # Check for Thai encoding issues
            has_issues = any(char in decoded for char in ['à¸', 'à¹', 'àº'])
            
            case_result = {
                "id": i,
                "text": text[:50] + "..." if len(text) > 50 else text,
                "char_count": char_count,
                "token_count": token_count,
                "ratio": ratio,
                "time": case_time,
                "has_encoding_issues": has_issues,
                "decoded_matches": decoded.strip() == text.strip()
            }
            
            results["cases"].append(case_result)
            results["stats"]["total_chars"] += char_count
            results["stats"]["total_tokens"] += token_count
            
            status = "✅" if not has_issues and decoded.strip() == text.strip() else "❌"
            print(f"  {status} Case {i}: {char_count} chars → {token_count} tokens (ratio: {ratio:.2f})")
            
        except Exception as e:
            print(f"  ❌ Case {i}: Error - {str(e)[:50]}")
            case_result = {
                "id": i,
                "text": text[:50] + "..." if len(text) > 50 else text,
                "error": str(e)
            }
            results["cases"].append(case_result)
    
    results["total_time"] = time.time() - start_time
    
    # Calculate averages
    valid_cases = [c for c in results["cases"] if "error" not in c]
    if valid_cases:
        results["stats"]["avg_tokens"] = results["stats"]["total_tokens"] / len(valid_cases)
        results["stats"]["avg_ratio"] = results["stats"]["total_chars"] / results["stats"]["total_tokens"]
        results["stats"]["success_rate"] = len([c for c in valid_cases if not c.get("has_encoding_issues", True)]) / len(valid_cases) * 100
    
    print(f"📊 {name} Summary:")
    print(f"   Time: {results['total_time']:.3f}s")
    print(f"   Avg tokens: {results['stats']['avg_tokens']:.1f}")
    print(f"   Compression ratio: {results['stats']['avg_ratio']:.2f} chars/token")
    print(f"   Success rate: {results['stats'].get('success_rate', 0):.1f}%")
    
    return results

def main():
    print("🔬 Thai Tokenizer Comparison & Benchmark")
    print("=" * 60)
    
    # Load tokenizers
    tokenizers = load_tokenizers()
    if not tokenizers:
        print("❌ No tokenizers found!")
        return
    
    # Prepare test cases
    test_cases = test_real_world_content()
    print(f"📝 Testing with {len(test_cases)} real-world cases...")
    
    # Benchmark each tokenizer
    all_results = {}
    for name, tokenizer in tokenizers.items():
        try:
            results = benchmark_tokenizer(name, tokenizer, test_cases)
            all_results[name] = results
        except Exception as e:
            print(f"❌ Failed to benchmark {name}: {e}")
    
    # Compare results
    print("\n🏆 Final Comparison")
    print("=" * 60)
    
    comparison_data = []
    for name, results in all_results.items():
        stats = results["stats"]
        comparison_data.append({
            "name": name,
            "avg_ratio": stats.get("avg_ratio", 0),
            "success_rate": stats.get("success_rate", 0),
            "time": results["total_time"]
        })
    
    # Sort by success rate, then by compression ratio
    comparison_data.sort(key=lambda x: (-x["success_rate"], -x["avg_ratio"]))
    
    print("Rank | Tokenizer     | Success% | Compression | Time")
    print("-" * 55)
    for i, data in enumerate(comparison_data, 1):
        print(f"{i:4d} | {data['name']:<12} | {data['success_rate']:7.1f}% | {data['avg_ratio']:10.2f} | {data['time']:.3f}s")
    
    # Save benchmark results
    output_file = "tokenizer_benchmark_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Results saved to {output_file}")
    
    # Winner announcement
    if comparison_data:
        winner = comparison_data[0]
        print(f"\n🥇 Winner: {winner['name']}")
        print(f"   Success Rate: {winner['success_rate']:.1f}%")
        print(f"   Compression: {winner['avg_ratio']:.2f} chars/token")
        print(f"   Speed: {winner['time']:.3f}s")

if __name__ == "__main__":
    main()
