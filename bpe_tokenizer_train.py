from BPE_Tokenizer import BPE_Tokenizer

# Train
tokenizer = BPE_Tokenizer()
with open("MahaBharat.txt", "r") as f:
    text = f.read()

tokenizer.train(text, 16000)

# Save
tokenizer.save("bpe_Tokenizer")