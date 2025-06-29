import re
import logging
from pathlib import Path
from collections import Counter
from typing import Set, List, Tuple, Optional
import unicodedata

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IOBDatasetCleaner:
    """Enhanced IOB dataset cleaner with comprehensive statistics and validation"""
    
    def __init__(self, allowed_tags: Set[str] = {'B', 'I'}):
        self.allowed_tags = allowed_tags
        # Extended Thai Unicode ranges
        self.thai_patterns = {
            'basic': re.compile(r'^[\u0E00-\u0E7F]$'),  # Basic Thai block
            'extended': re.compile(r'^[\u0E00-\u0E7F\u0E80-\u0EFF]$'),  # Include Lao for some mixed texts
            'strict_thai': re.compile(r'^[\u0E01-\u0E3A\u0E3F-\u0E5B]$'),  # Only Thai letters and digits
        }
        # Thai character categories
        self.thai_consonants = re.compile(r'^[\u0E01-\u0E2E]$')
        self.thai_vowels = re.compile(r'^[\u0E30-\u0E3A\u0E40-\u0E4E]$')
        self.thai_tones = re.compile(r'^[\u0E48-\u0E4B]$')
        self.thai_digits = re.compile(r'^[\u0E50-\u0E59]$')
        
        # Statistics tracking
        self.stats = {
            'total_lines': 0,
            'blank_lines': 0,
            'malformed_lines': 0,
            'non_thai_chars': 0,
            'invalid_tags': 0,
            'duplicate_blanks_removed': 0,
            'cleaned_lines': 0,
            'sentences': 0,
            'char_distribution': Counter(),
            'tag_distribution': Counter(),
            'rejected_chars': Counter(),
            'rejected_tags': Counter()
        }
    
    def normalize_unicode(self, text: str) -> str:
        """Normalize Unicode text using NFC normalization"""
        return unicodedata.normalize('NFC', text)
    
    def is_thai_char(self, char: str, mode: str = 'basic') -> bool:
        """Check if character is Thai with different strictness levels"""
        if mode not in self.thai_patterns:
            mode = 'basic'
        return self.thai_patterns[mode].match(char) is not None
    
    def validate_iob_sequence(self, tags: List[str]) -> bool:
        """Validate IOB tag sequence for consistency"""
        if not tags:
            return True
        
        # Check for invalid I at start or after non-B
        for i, tag in enumerate(tags):
            if tag == 'I':
                # I must follow B or another I
                if i == 0 or tags[i-1] not in ['B', 'I']:
                    return False
        return True
    
    def clean_dataset(self, 
                     input_path: str, 
                     output_path: str,
                     thai_mode: str = 'basic',
                     validate_sequences: bool = False,
                     min_sentence_length: int = 1,
                     max_sentence_length: int = 512,
                     preserve_whitespace_chars: bool = False) -> bool:
        """
        Clean IOB dataset with comprehensive options
        
        Args:
            input_path: Input file path
            output_path: Output file path  
            thai_mode: Thai character validation mode ('basic', 'extended', 'strict_thai')
            validate_sequences: Whether to validate IOB sequence consistency
            min_sentence_length: Minimum sentence length to keep
            max_sentence_length: Maximum sentence length to keep
            preserve_whitespace_chars: Whether to keep whitespace characters
        """
        try:
            input_file = Path(input_path)
            output_file = Path(output_path)
            
            if not input_file.exists():
                logger.error(f"Input file not found: {input_path}")
                return False
            
            # Create output directory if needed
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Cleaning IOB dataset: {input_path} -> {output_path}")
            logger.info(f"Thai mode: {thai_mode}, Validate sequences: {validate_sequences}")
            
            cleaned_lines = []
            current_sentence = []
            current_tags = []
            
            with open(input_path, 'r', encoding='utf-8') as fin:
                for line_num, line in enumerate(fin, 1):
                    self.stats['total_lines'] += 1
                    line = line.strip()
                    
                    # Handle blank lines (sentence boundaries)
                    if not line:
                        self.stats['blank_lines'] += 1
                        
                        # Process accumulated sentence
                        if current_sentence:
                            if self._process_sentence(current_sentence, current_tags, 
                                                   cleaned_lines, validate_sequences,
                                                   min_sentence_length, max_sentence_length):
                                self.stats['sentences'] += 1
                            current_sentence = []
                            current_tags = []
                        continue
                    
                    # Parse line
                    parts = line.split()
                    if len(parts) != 2:
                        self.stats['malformed_lines'] += 1
                        logger.debug(f"Malformed line {line_num}: {line}")
                        continue
                    
                    char, tag = parts
                    
                    # Normalize Unicode
                    char = self.normalize_unicode(char)
                    
                    # Handle whitespace characters
                    if char.isspace():
                        if preserve_whitespace_chars:
                            char = ' '  # Normalize all whitespace to space
                        else:
                            continue  # Skip whitespace
                    
                    # Validate Thai character
                    if not self.is_thai_char(char, thai_mode) and not char.isspace():
                        self.stats['non_thai_chars'] += 1
                        self.stats['rejected_chars'][char] += 1
                        continue
                    
                    # Validate tag
                    if tag not in self.allowed_tags:
                        self.stats['invalid_tags'] += 1
                        self.stats['rejected_tags'][tag] += 1
                        continue
                    
                    # Add to current sentence
                    current_sentence.append(char)
                    current_tags.append(tag)
                    self.stats['char_distribution'][char] += 1
                    self.stats['tag_distribution'][tag] += 1
            
            # Process last sentence if exists
            if current_sentence:
                if self._process_sentence(current_sentence, current_tags, 
                                       cleaned_lines, validate_sequences,
                                       min_sentence_length, max_sentence_length):
                    self.stats['sentences'] += 1
            
            # Remove consecutive blank lines
            final_lines = self._remove_consecutive_blanks(cleaned_lines)
            
            # Write cleaned data
            with open(output_path, 'w', encoding='utf-8') as fout:
                for line in final_lines:
                    fout.write(line + '\n')
            
            self.stats['cleaned_lines'] = len([l for l in final_lines if l.strip()])
            
            # Print statistics
            self._print_statistics(input_path, output_path)
            
            logger.info(f"Successfully cleaned dataset: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error cleaning dataset: {e}")
            return False
    
    def _process_sentence(self, chars: List[str], tags: List[str], 
                         output_lines: List[str], validate_sequences: bool,
                         min_length: int, max_length: int) -> bool:
        """Process a single sentence"""
        
        # Check sentence length
        if len(chars) < min_length or len(chars) > max_length:
            return False
        
        # Validate IOB sequence if requested
        if validate_sequences and not self.validate_iob_sequence(tags):
            return False
        
        # Add sentence to output
        for char, tag in zip(chars, tags):
            output_lines.append(f"{char} {tag}")
        output_lines.append('')  # Sentence boundary
        
        return True
    
    def _remove_consecutive_blanks(self, lines: List[str]) -> List[str]:
        """Remove consecutive blank lines"""
        result = []
        prev_blank = False
        
        for line in lines:
            if line == '':
                if not prev_blank:
                    result.append('')
                else:
                    self.stats['duplicate_blanks_removed'] += 1
                prev_blank = True
            else:
                result.append(line)
                prev_blank = False
        
        # Remove trailing blank line
        while result and result[-1] == '':
            result.pop()
            
        return result
    
    def _print_statistics(self, input_path: str, output_path: str):
        """Print cleaning statistics"""
        print("\n" + "="*60)
        print("IOB DATASET CLEANING STATISTICS")
        print("="*60)
        print(f"Input file:  {input_path}")
        print(f"Output file: {output_path}")
        print("-"*60)
        print(f"Total lines processed:     {self.stats['total_lines']:,}")
        print(f"Blank lines:               {self.stats['blank_lines']:,}")
        print(f"Malformed lines removed:   {self.stats['malformed_lines']:,}")
        print(f"Non-Thai chars removed:    {self.stats['non_thai_chars']:,}")
        print(f"Invalid tags removed:      {self.stats['invalid_tags']:,}")
        print(f"Duplicate blanks removed:  {self.stats['duplicate_blanks_removed']:,}")
        print(f"Final cleaned lines:       {self.stats['cleaned_lines']:,}")
        print(f"Sentences:                 {self.stats['sentences']:,}")
        
        if self.stats['cleaned_lines'] > 0:
            retention_rate = (self.stats['cleaned_lines'] / 
                            (self.stats['total_lines'] - self.stats['blank_lines'])) * 100
            print(f"Data retention rate:       {retention_rate:.1f}%")
        
        print("\nTag distribution:")
        for tag, count in sorted(self.stats['tag_distribution'].items()):
            print(f"  {tag}: {count:,}")
        
        print(f"\nCharacter vocabulary size: {len(self.stats['char_distribution'])}")
        
        # Show most common rejected characters/tags
        if self.stats['rejected_chars']:
            print("\nMost rejected characters:")
            for char, count in self.stats['rejected_chars'].most_common(10):
                print(f"  '{char}': {count}")
        
        if self.stats['rejected_tags']:
            print("\nRejected tags:")
            for tag, count in self.stats['rejected_tags'].most_common():
                print(f"  '{tag}': {count}")
        
        print("="*60)

def analyze_thai_text_distribution(file_path: str):
    """Analyze Thai character distribution in dataset"""
    char_types = {'consonants': 0, 'vowels': 0, 'tones': 0, 'digits': 0, 'other': 0}
    cleaner = IOBDatasetCleaner()
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                continue
            char = parts[0]
            
            if cleaner.thai_consonants.match(char):
                char_types['consonants'] += 1
            elif cleaner.thai_vowels.match(char):
                char_types['vowels'] += 1
            elif cleaner.thai_tones.match(char):
                char_types['tones'] += 1
            elif cleaner.thai_digits.match(char):
                char_types['digits'] += 1
            else:
                char_types['other'] += 1
    
    print("\nThai Character Type Distribution:")
    total = sum(char_types.values())
    for char_type, count in char_types.items():
        percentage = (count / total * 100) if total > 0 else 0
        print(f"  {char_type.capitalize()}: {count:,} ({percentage:.1f}%)")

# Enhanced usage examples
def main():
    # Basic cleaning
    cleaner = IOBDatasetCleaner(allowed_tags={'B', 'I'})
    
    # Clean with basic settings
    cleaner.clean_dataset(
        'data/train_iob.txt', 
        'data/train_iob_clean.txt'
    )
    
    # Clean with strict validation
    cleaner_strict = IOBDatasetCleaner(allowed_tags={'B', 'I'})
    cleaner_strict.clean_dataset(
        'data/train_iob.txt',
        'data/train_iob_strict.txt',
        thai_mode='strict_thai',
        validate_sequences=True,
        min_sentence_length=3,
        max_sentence_length=256,
        preserve_whitespace_chars=False
    )
    
    # Analyze character distribution
    analyze_thai_text_distribution('data/train_iob_clean.txt')

if __name__ == '__main__':
    main()