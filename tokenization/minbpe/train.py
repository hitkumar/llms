"""
Train our tokenizers on some data
"""
import os
import time
from src import BasicTokenizer, RegexTokenizer
from pathlib import Path

# get some text
text = open('tests/taylorswift.txt', 'r', encoding='utf-8').read()

os.makedirs('models', exist_ok=True)

t0 = time.time()

for TokenizerClass, name in zip([BasicTokenizer, RegexTokenizer], ['basic', 'regex']):
    tokenizer = TokenizerClass()
    # 256 merges
    tokenizer.train(text, 256 + 256, verbose=False)
    # save model
    model_file = Path('models', name)
    # print(model_file)
    tokenizer.save(model_file)
    t1 = time.time()
    print(f"Took {t1 - t0:.2f}s to train {name} model")

