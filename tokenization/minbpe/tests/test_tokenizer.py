from src import BasicTokenizer

import pytest
import tiktoken
import os

# ---------------------------------------------------------------
# common test data
test_strings = [
    "",
    "?",
    "hello world!!!? (안녕하세요!) lol123 😉", # fun small string
]


# ---------------------------------------------------------------

def train_tokenizer(tokenizer):
    text = open('taylorswift.txt', "r", encoding='utf-8').read()
    tokenizer.train(text, 512, verbose=True)

# tests
@pytest.mark.parametrize("tokenizer_factory", [BasicTokenizer])
@pytest.mark.parametrize("text", test_strings)
def test_encode_decode_identity(tokenizer_factory, text):
    tokenizer = tokenizer_factory()
    # TODO: Is this step necessary, karpathy@ doesn't use it?
    train_tokenizer(tokenizer)
    ids = tokenizer.encode(text)
    # TODO: how to make the logging work?
    print(ids)
    decoded = tokenizer.decode(ids)
    assert text == decoded

@pytest.mark.parametrize("tokenizer_factory", [BasicTokenizer])
def test_basic_tokenizer(tokenizer_factory):
    tokenizer = tokenizer_factory()
    tokenizer.train("How are you doing", 257)
    assert tokenizer.decode(tokenizer.encode("abcd")) == "abcd"

if __name__=="__main__":
    pytest.main()