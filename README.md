# my_thai_tokenizer

ML-based Thai word tokenizer (template)

## โครงสร้างโปรเจกต์

```text
my_thai_tokenizer/
├── __init__.py
├── tokenizer.py         # core logic (ML-based)
├── example.py           # ตัวอย่างการใช้งาน
├── data/                # สำหรับเก็บ dataset (เช่นไฟล์ .txt, .csv, .tsv)
└── model/               # สำหรับเก็บ model ที่ train แล้ว
```

## วิธีใช้งานครบจบ (Workflow)

1. เตรียมไฟล์ข้อความภาษาไทย (เช่น .txt หลายไฟล์ใน data/)
2. รวมไฟล์ทั้งหมดเป็นไฟล์เดียว (thai_corpus.txt):

```bash
python prepare_corpus.py
```

3. ติดตั้ง dependencies:

```bash
pip install sentencepiece
```

4. Train SentencePiece tokenizer:

```bash
python train_sentencepiece.py
```

5. ใช้งาน tokenizer ที่ train เอง:

```python
from sp_tokenizer import ThaiSentencePieceTokenizer

tokenizer = ThaiSentencePieceTokenizer('model/thai_spm.model')
text = "ฉันรักประเทศไทยหมูกรอบอร่อย"
tokens = tokenizer.word_tokenize(text)
print(tokens)
```

- สามารถนำ model นี้ไปใช้กับ Huggingface Transformers ได้ทันที
- รองรับข้อความไทยทุกแบบ ไม่ต้องพึ่ง wordlist

## การสร้าง tokenizer ภาษาไทยด้วย SentencePiece

SentencePiece เป็น unsupervised tokenizer ที่เหมาะกับภาษาไทยมาก (ใช้ใน WangchanBERTa, T5, ฯลฯ)

### ขั้นตอน

1. เตรียมไฟล์ corpus ข้อความภาษาไทย (เช่น data/thai_corpus.txt)
2. ติดตั้ง sentencepiece

```bash
pip install sentencepiece
```

3. Train tokenizer (Unigram หรือ BPE)

```bash
python -m sentencepiece --input=data/thai_corpus.txt --model_prefix=model/thai_spm --vocab_size=32000 --model_type=unigram
```

- `--input` : ไฟล์ข้อความภาษาไทย (หนึ่งบรรทัดต่อหนึ่งประโยค)
- `--model_prefix` : prefix ของไฟล์ model ที่จะได้ (เช่น model/thai_spm.model, model/thai_spm.vocab)
- `--vocab_size` : ขนาด vocabulary (เช่น 16000, 32000)
- `--model_type` : เลือก 'unigram' (เหมาะกับภาษาไทย) หรือ 'bpe'

4. ใช้งาน tokenizer ใน Python

```python
import sentencepiece as spm
sp = spm.SentencePieceProcessor()
sp.load('model/thai_spm.model')
text = "ฉันรักประเทศไทยหมูกรอบอร่อย"
tokens = sp.encode(text, out_type=str)
print(tokens)
```

## ตัวอย่างไฟล์ corpus

ไฟล์ตัวอย่าง (data/thai_corpus_example.txt):

```
ฉันรักประเทศไทยมาก
หมูกรอบอร่อยที่สุด
วันนี้อากาศดี
ไปเที่ยวเชียงใหม่กับเพื่อน
ภาษาไทยไม่มีการเว้นวรรคระหว่างคำ
```

## สคริปต์ช่วยเตรียม corpus

ใช้ `prepare_corpus.py` เพื่อรวมไฟล์ .txt หลายไฟล์ใน data/ เป็นไฟล์เดียวสำหรับ train SentencePiece:

```bash
python prepare_corpus.py
```

จะได้ไฟล์ `data/thai_corpus.txt` สำหรับ train tokenizer

## Integration กับ PyThaiNLP/transformers

### ใช้ tokenizer ที่ train ด้วย SentencePiece กับ PyThaiNLP

```python
import sentencepiece as spm
sp = spm.SentencePieceProcessor()
sp.load('model/thai_spm.model')
text = "ฉันรักประเทศไทยหมูกรอบอร่อย"
tokens = sp.encode(text, out_type=str)
print(tokens)
```

### ใช้กับ Huggingface Transformers

```python
from transformers import AutoTokenizer
# โหลด tokenizer ที่ train เอง (เช่น WangchanBERTa-style)
tokenizer = AutoTokenizer.from_pretrained('path/to/model_dir')
text = "ฉันรักประเทศไทยหมูกรอบอร่อย"
tokens = tokenizer.tokenize(text)
print(tokens)
```

- สามารถนำ model_dir ที่ได้จากการ train SentencePiece ไปใช้กับ transformers ได้ทันที (แค่มี .model, .vocab, และไฟล์ config)

### ข้อดี
- ไม่ต้องพึ่ง wordlist หรือ dictionary
- เหมาะกับภาษาไทยและภาษาอื่นที่ไม่มีการเว้นวรรค
- ใช้กับงาน NLP สมัยใหม่ (BERT, T5, etc.)

## หมายเหตุ

- คุณสามารถนำ dataset (เช่น BEST2010, LEXiTRON, หรือชุดข้อมูลอื่น) มาใส่ในโฟลเดอร์ `data/`
- สามารถนำโมเดลที่ train แล้วมาใส่ในโฟลเดอร์ `model/`
- ตัวอย่างนี้เป็น template สำหรับเริ่มต้นพัฒนา ML-based tokenizer (BiLSTM+CRF, transformer, sentencepiece ฯลฯ)

## สรุปไฟล์สำคัญ
- `data/thai_corpus_example.txt` : ตัวอย่างไฟล์ข้อความ
- `prepare_corpus.py` : รวมไฟล์ข้อความหลายไฟล์
- `train_sentencepiece.py` : train tokenizer
- `sp_tokenizer.py` : ใช้งาน tokenizer ที่ train เอง
- `model/thai_spm.model` : ไฟล์ model ที่ได้

ครบจบ พร้อมใช้งานจริงสำหรับภาษาไทย!
