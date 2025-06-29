# ZplitThai: Thai Tokenizer Toolkit

ระบบ tokenizer ภาษาไทยแบบ ML (BiLSTM-CRF) และ subword (SentencePiece, HuggingFace Tokenizers)

## โครงสร้างโปรเจกต์

```text
ZplitThai/
├── train.py                  # เทรน BiLSTM-CRF tokenizer (ML-based)
├── train_sentencepiece.py    # เทรน SentencePiece tokenizer
├── train_hf_tokenizer.py     # เทรน HuggingFace BPE tokenizer
├── sp_tokenizer.py           # ใช้งาน SentencePiece tokenizer
├── clean.py                  # ฟังก์ชัน clean/preprocess ข้อมูล
├── prepare_corpus.py         # รวมไฟล์ข้อความหลายไฟล์
├── analyze_iob_dataset.py    # QC/วิเคราะห์ dataset IOB
├── hfupload.py               # อัปโหลด tokenizer ขึ้น HuggingFace Hub
├── datasethfopload.py        # อัปโหลด dataset ขึ้น HuggingFace Hub
├── data/                     # dataset (txt, IOB, ฯลฯ)
│   ├── combined_thai_corpus.txt
│   ├── train_iob_clean.txt
│   ├── train_iob_strict.txt
│   └── ...
├── model/                    # โมเดลที่ train แล้ว (bilstm.pth, bilstm_crf.pth, thai_spm.model)
├── Bitthaitokenizer/         # HuggingFace tokenizer (tokenizer.json, vocab.json, tokenizer_config.json)
├── .gitignore
└── README.md
```

## Workflow ครบจบ (ML, Subword, HuggingFace)

### 1. เตรียมและ clean dataset

- วางไฟล์ข้อความภาษาไทย (.txt) ใน `data/`
- รวมไฟล์เป็น corpus เดียว:

```bash
python prepare_corpus.py
```

- QC/clean dataset (IOB):

```bash
python analyze_iob_dataset.py
python clean.py
```

### 2. ติดตั้ง dependencies

```bash
pip install -r requirements.txt
# หรือ
pip install sentencepiece transformers huggingface_hub pythainlp
```

### 3. เทรน tokenizer

#### (A) ML-based (BiLSTM-CRF)

```bash
python train.py
```
- รองรับ resume training, validation, F1-score, checkpoint
- โมเดลจะถูกบันทึกใน `model/`

#### (B) SentencePiece

```bash
python train_sentencepiece.py
```
- ปรับพารามิเตอร์ในสคริปต์ได้ตามต้องการ
- โมเดลจะถูกบันทึกใน `model/thai_spm.model`

#### (C) HuggingFace Tokenizer (BPE)

```bash
python train_hf_tokenizer.py
```
- ได้ไฟล์ `tokenizer.json`, `vocab.json`, `tokenizer_config.json` ใน `Bitthaitokenizer/`

### 4. อัปโหลด dataset/tokenizer ขึ้น HuggingFace Hub

- อัปโหลด tokenizer:

```bash
python hfupload.py
```
- อัปโหลด dataset:
```bash
python datasethfopload.py
```

### 5. ใช้งาน tokenizer

#### (A) SentencePiece
```python
from sp_tokenizer import ThaiSentencePieceTokenizer
sp = ThaiSentencePieceTokenizer('model/thai_spm.model')
tokens = sp.word_tokenize("ฉันรักประเทศไทยหมูกรอบอร่อย")
print(tokens)
```

#### (B) HuggingFace Tokenizer
```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('Bitthaitokenizer')
tokens = tokenizer.tokenize("ฉันรักประเทศไทยหมูกรอบอร่อย")
print(tokens)
```

#### (C) ML-based (BiLSTM-CRF)
```python
# ดูตัวอย่างใน train.py หรือ Bitthaitokenizer/README.md
```

### 6. Integration กับ PyThaiNLP/Transformers/LLM
- ใช้ tokenizer ที่ train เองกับ PyThaiNLP หรือ transformers ได้ทันที
- รองรับ LLM, Thai NLP, downstream task

## หมายเหตุ/ข้อควรทราบ
- Dataset/Tokenizer ผ่าน QC/clean แล้ว (ดู analyze_iob_dataset.py, clean.py)
- สามารถปรับแต่ง pre-tokenizer/custom logic เพิ่มเติมได้ (เช่น PyThaiNLP)
- รองรับการอัปโหลด HuggingFace Hub (dataset/tokenizer)
- ตัวอย่างไฟล์สำคัญ: train.py, train_sentencepiece.py, train_hf_tokenizer.py, sp_tokenizer.py, Bitthaitokenizer/
- README และ workflow ครบถ้วน พร้อมใช้งานจริง

## ตัวอย่างไฟล์สำคัญ
- `data/combined_thai_corpus.txt` : corpus หลัก
- `data/train_iob_clean.txt` : dataset IOB ที่ QC แล้ว
- `model/thai_spm.model` : SentencePiece model
- `Bitthaitokenizer/` : HuggingFace tokenizer
- `train.py`, `train_sentencepiece.py`, `train_hf_tokenizer.py` : สคริปต์เทรน tokenizer
- `hfupload.py`, `datasethfopload.py` : อัปโหลดขึ้น HuggingFace Hub

ครบจบ พร้อมใช้งานจริงสำหรับภาษาไทย ทั้ง ML-based และ subword tokenizer!
