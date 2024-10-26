from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download, login


login(token="LOGIN_TOKEN")


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

# Size of the vocabulary which precludes special tokens e.g <|begin_of_text|>, etc
VOCAB_SIZE = 128000

def convert():
	with open("tokenizer.bin", "wb") as f:
		for i in range(VOCAB_SIZE):
			word = tokenizer.decode([i], clean_up_tokenization_spaces=False)
			word_byte = word.encode(encoding="utf-8")
			f.write(int.to_bytes(len(word_byte), length=4, byteorder="little"))
			f.write(word_byte)

convert()