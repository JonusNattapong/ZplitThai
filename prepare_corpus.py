# prepare_corpus.py
"""
สคริปต์ช่วยเตรียมไฟล์ corpus สำหรับ train SentencePiece หรือ ML-based tokenizer
- รวมไฟล์ข้อความหลายไฟล์เป็นไฟล์เดียว
- ลบอักขระพิเศษ/clean ข้อความ
- สามารถดึงข้อความจากไฟล์ .txt, .csv, หรือ json ได้
"""
import os
import glob
import csv
import re
import pythainlp

# เพิ่มบรรทัดนี้เพื่อรองรับ field ขนาดใหญ่
csv.field_size_limit(10**7)

def clean_text(text):
    """ลบอักขระพิเศษและจัดเรียงข้อความเป็นคำๆ โดยกรองเฉพาะคำภาษาไทย"""
    text = re.sub(r'[^\u0E00-\u0E7F\s]', '', text)  # ลบอักขระที่ไม่ใช่ภาษาไทยและช่องว่าง
    text = re.sub(r'\s+', ' ', text).strip()  # ลบช่องว่างเกินและ trim
    words = pythainlp.word_tokenize(text)  # ตัดคำภาษาไทย
    words = [word for word in words if re.match(r'[\u0E00-\u0E7F]', word)]  # กรองเฉพาะคำภาษาไทย
    return '\n'.join(words)  # เริ่มบรรทัดใหม่หลังแต่ละคำ

def merge_txt_files(input_dir, output_file):
    with open(output_file, 'w', encoding='utf-8') as fout:
        for fname in glob.glob(os.path.join(input_dir, '*.txt')):
            with open(fname, encoding='utf-8') as fin:
                for line in fin:
                    line = clean_text(line)
                    if line:
                        fout.write(line + '\n')
    print(f'Merged corpus saved to {output_file}')

def merge_csv_file(input_csv, output_file, text_column_name='text'):
    """รวมข้อความจากไฟล์ .csv (เลือกคอลัมน์ 'text') เป็นไฟล์ .txt"""
    with open(output_file, 'w', encoding='utf-8') as fout:
        with open(input_csv, encoding='utf-8') as fin:
            reader = csv.DictReader(fin)
            total_rows = sum(1 for _ in fin)  # นับจำนวนแถวทั้งหมด
            fin.seek(0)  # กลับไปที่จุดเริ่มต้นของไฟล์
            reader = csv.DictReader(fin)

            for row_num, row in enumerate(reader, 1):
                text = row.get(text_column_name, '').strip()
                text = clean_text(text)
                if text:
                    fout.write(text + '\n')

                # แสดงความคืบหน้าทุก 10,000 แถว
                if row_num % 10000 == 0:
                    print(f"Processed {row_num}/{total_rows} rows...")

    print(f'Merged CSV corpus saved to {output_file}')

if __name__ == '__main__':

    # ตัวอย่างการใช้งาน: รวมข้อความจากไฟล์ .csv เป็น thai_corpus.txt
    # merge_csv_file('data/thwiki_articles_20250629.csv', 'data/thai_corpus1.txt', text_column_name='text')
    # merge_csv_file('data/sql-console-for-zombitx64-opensubtitles-english-thai.csv', 'data/thai_corpus2.txt', text_column_name='text')
    # merge_csv_file('data/sql-console-for-zombitx64-opensubtitles-tw-corpus-thai-to-chinese.csv', 'data/thai_corpus3.txt', text_column_name='text')
    # merge_csv_file('data/sql-console-for-zombitx64-multiccaligned-tw-corpus-thai-to-chinese.csv', 'data/thai_corpus4.txt', text_column_name='text')
    # merge_csv_file('data/data/sql-console-for-zombitx64-ted2020-tw-corpus-thai-to-chinese.csv.csv', 'data/thai_corpus5.txt', text_column_name='text')
    # ตัวอย่างการใช้งาน: รวมไฟล์ .txt ทั้งหมดใน data/ เป็น thai_corpus.txt
    merge_txt_files('data', 'data/thai_corpus.txt')
