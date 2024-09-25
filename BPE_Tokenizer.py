import json

class BPE_Tokenizer():
    def __init__(self):
        super().__init__()
        self.vocab = {}
        self.bpe_merges = {}

    def get_stats(ids, counts=None):
        counts = {} if counts is None else counts
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts
    
    def merges(ids, pair, idx):
        token_list = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                token_list.append(idx)
                i += 2
            else:
                token_list.append(ids[i])
                i += 1
        return token_list
    
    def train(self, text, vocab_size, verbose=True):
        assert vocab_size >= 256
        num_merges = vocab_size - 256
        
        # convert text file to utf-8 encoding
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)
        
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for i in range(num_merges):
            stats = BPE_Tokenizer.get_stats(ids)
            top_pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = BPE_Tokenizer.merges(ids, top_pair, idx)
            self.bpe_merges[top_pair] = idx
            vocab[idx] = vocab[top_pair[0]] + vocab[top_pair[1]]
            if verbose:
                print(f"Merge {i+1}/{num_merges}: {top_pair} -> {idx} {vocab[idx]} had {stats[top_pair]} occurences.")

        self.vocab = vocab

    def encode(self, text):
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)
        while len(ids) >= 2:
            stats = BPE_Tokenizer.get_stats(ids)
            pair = {pair for pair in stats.keys() if pair in self.bpe_merges}
            if not pair:
                break
            top_pair = min(pair, key=lambda p: self.bpe_merges[p])
            idx = self.bpe_merges[pair]
            ids = BPE_Tokenizer.merges(ids, top_pair, idx)

        return ids
    
    def decode(self, ids):
        text_bytes = b"".join(self.vocab[ids] for ids in ids)
        text = text_bytes.decode("utf-8", errors="replace")
        return text
    
    def save(self, file_prefix):
        # Save the tokenizer as JSON file
        tokenizer_data = {
            "vocab": {k: v.decode("utf-8", errors="ignore") for k, v in self.vocab.items()},
            "bpe_merges": {str(k): v for k, v in self.bpe_merges.items()}
        }
        self.file_path = file_path
        file_path = f"{file_prefix}.json"
        with open(file_path, "w") as f:
            json.dump(tokenizer_data, f)
        print(f"Tokenizer saved to {file_path}")
    
    def load(self):
        # Loads the tokenizer from JSON file
        with open(self.file_path, "r") as f:
            tokenizer_data = json.load(f)
        
        self.vocab = {int(k): v.encode("utf-8") for k, v in tokenizer_data["vocab"].items()}
        self.bpe_merges = {tuple(map(int, k.strip("()").split(","))): v for k, v in tokenizer_data["bpe_merges"].items()}
        print(f"Tokenizer loaded from {self.file_path}")

    def encode_text(self, text):
        BPE_Tokenizer.load(self.file_path)
        tokens = BPE_Tokenizer.encode(text)
        return tokens
