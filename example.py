from tokenizer import ThaiMLTokenizer

tokenizer = ThaiMLTokenizer()
text = "ฉันรักประเทศไทยหมูกรอบอร่อย"
tokens = tokenizer.word_tokenize(text)
print("Tokenized:", tokens)
