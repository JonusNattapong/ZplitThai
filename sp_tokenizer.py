# sp_tokenizer.py
"""
ใช้งาน SentencePiece tokenizer ภาษาไทยที่ train เอง
"""
import sentencepiece as spm
import os

class ThaiSentencePieceTokenizer:
    def __init__(self, model_path='model/thai_spm.model'):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)

    def word_tokenize(self, text: str) -> list[str]:
        return self.sp.encode(text, out_type=str)

if __name__ == '__main__':
    tokenizer = ThaiSentencePieceTokenizer()
    text = "เฉพาะพระองค์เท่านั้นที่พวกข้าพระองค์เคารพอิบาดะฮฺ และเฉพาะพระองค์เท่านั้นที่พวกข้าพระองค์ขอความช่วยเหลือ"
    print(tokenizer.word_tokenize(text))
