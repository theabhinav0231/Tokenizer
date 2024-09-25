# Tokenizer

I'm training custom Tokenizers on texts of Mahabharat. The MahaBharat.txt file contains ~1.4 Million words.

First Step: To train a simple byte-pair encoding tokenizer.
The code file BPE_tokenizer.py contains all the code you need to train your own BPE Tokenizer. The tokenizer does not handle special tokens like EOS (End of Statement). 

Second Step: To train a byte-pair encoding tokenizer with capability of handling special tokens and uses Regex pattern for spilliting texts.

My project is inspired from Andrej Karpathy's GPT Tokenizer video- https://youtu.be/zduSFxRajkE?si=caNaMTd9UkOO-05S
minbpe Github link- https://github.com/karpathy/minbpe
Thanks to Andrej Karpathy for teaching great stuff!
