from transformers import AutoTokenizer

# Load tokenizer from HuggingFace Hub
try:
    tokenizer = AutoTokenizer.from_pretrained("ZombitX64/Thaitokenizer")
    text = "นั่งตาก ลม"
    tokens = tokenizer.tokenize(text)
    print(f"Tokens: {tokens}")
    encoding = tokenizer(text, return_tensors=None, add_special_tokens=False)
    decoded = tokenizer.decode(encoding['input_ids'], skip_special_tokens=True)
    print(f"Original: {text}")
    print(f"Decoded: {decoded}")
except Exception as e:
    print(f"Error loading tokenizer: {e}")
