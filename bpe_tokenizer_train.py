from BPE_Tokenizer import BPE_Tokenizer

tokenizer = BPE_Tokenizer()
with open("MahaBharat.txt", "r") as f:
    text = f.read()

# Train
tokenizer.train(text, 16000)
# Save
tokenizer.save("bpe_Tokenizer")
