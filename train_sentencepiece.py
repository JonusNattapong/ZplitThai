# train_sentencepiece.py
"""
Train SentencePiece tokenizer for Thai text.
- ใช้กับไฟล์ corpus ที่รวมข้อความไทย (เช่น data/thai_corpus.txt)
- ได้ไฟล์ model/thai_spm.model, model/thai_spm.vocab
"""
import sentencepiece as spm
import os
import csv

# เพิ่มบรรทัดนี้เพื่อรองรับ field ขนาดใหญ่
csv.field_size_limit(10**7)

def csv_to_txt(input_csv, output_txt, text_column_name='text'):
    """แปลงไฟล์ .csv (header: id,url,title,text) เป็นไฟล์ .txt (หนึ่งบรรทัดต่อหนึ่งประโยค โดยใช้คอลัมน์ 'text')"""
    with open(input_csv, encoding='utf-8') as fin, open(output_txt, 'w', encoding='utf-8') as fout:
        reader = csv.DictReader(fin)
        for row in reader:
            text = row.get(text_column_name, '').strip()
            if text:
                fout.write(text + '\n')
    print(f'Extracted text from {input_csv} to {output_txt} (column: {text_column_name})')

def csv_to_txt_multi(input_csv_list, output_txt, text_column_name='text'):
    """รวมไฟล์ .csv หลายไฟล์ (header: id,url,title,text) โดยดึงคอลัมน์ 'text' รวมเป็นไฟล์ .txt เดียว"""
    with open(output_txt, 'w', encoding='utf-8') as fout:
        for input_csv in input_csv_list:
            if not os.path.exists(input_csv):
                print(f'File not found: {input_csv}')
                continue
            with open(input_csv, encoding='utf-8') as fin:
                reader = csv.DictReader(fin)
                for row in reader:
                    text = row.get(text_column_name, '').strip()
                    if text:
                        fout.write(text + '\n')
            print(f'Extracted text from {input_csv} (column: {text_column_name})')
    print(f'All CSVs merged to {output_txt}')

def train_sentencepiece(input_file, model_prefix, vocab_size=8000, model_type='unigram'):
    spm.SentencePieceTrainer.Train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=0.9995,  # ลด character coverage เพื่อรองรับตัวอักษรน้อยลง
        input_sentence_size=1000000,
        shuffle_input_sentence=True
    )
    print(f"Model saved to {model_prefix}.model and {model_prefix}.vocab")

def merge_txt_files(input_files, output_file):
    """รวมข้อความจากหลายไฟล์ .txt เป็นไฟล์เดียว"""
    with open(output_file, 'w', encoding='utf-8') as fout:
        for input_file in input_files:
            if not os.path.exists(input_file):
                print(f'File not found: {input_file}')
                continue
            with open(input_file, 'r', encoding='utf-8') as fin:
                for line in fin:
                    fout.write(line)
    print(f'Merged files into {output_file}')

if __name__ == '__main__':
    os.makedirs('model', exist_ok=True)

    # รวมไฟล์ thai_corpus1.txt ถึง thai_corpus5.txt
    corpus_files = [
        'data/thai_corpus1.txt',
        'data/thai_corpus2.txt',
        'data/thai_corpus3.txt',
        'data/thai_corpus4.txt',
        'data/thai_corpus5.txt'
    ]
    combined_corpus_path = 'data/combined_thai_corpus.txt'

    with open(combined_corpus_path, 'w', encoding='utf-8') as fout:
        for corpus_file in corpus_files:
            if not os.path.exists(corpus_file):
                print(f'File not found: {corpus_file}')
                continue
            with open(corpus_file, 'r', encoding='utf-8') as fin:
                fout.write(fin.read())

    # Train SentencePiece tokenizer
    train_sentencepiece(combined_corpus_path, 'model/BitthaiSPM', vocab_size=8000, model_type='unigram')
