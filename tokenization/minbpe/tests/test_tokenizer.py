from src import BasicTokenizer, RegexTokenizer

import pytest
import tiktoken
import os

# ---------------------------------------------------------------
# common test data
test_strings = [
    "",
    "?",
    "hello world!!!? (안녕하세요!) lol123 😉", # fun small string
    "FILE:taylorswift.txt", # File is handled separately in unpack.
]

def unpack(text):
    """ Writing to avoid printing the entire contents of file to console which pytest does by default"""
    if text.startswith("FILE:"):
        dirname = os.path.dirname(os.path.abspath(__file__))
        file = os.path.join(dirname, text[5:])
        contents = open(file, "r", encoding="utf-8").read()
        return contents
    else:
        return text

# ---------------------------------------------------------------

def train_tokenizer(tokenizer):
    text = open('taylorswift.txt', "r", encoding='utf-8').read()
    tokenizer.train(text, 512, verbose=False)

# tests
@pytest.mark.parametrize("tokenizer_factory", [BasicTokenizer, RegexTokenizer])
@pytest.mark.parametrize("text", test_strings)
def test_encode_decode_identity(tokenizer_factory, text):
    text = unpack(text)
    tokenizer = tokenizer_factory()
    # TODO: Is this step necessary, karpathy@ doesn't use it?
    train_tokenizer(tokenizer)
    ids = tokenizer.encode(text)
    # how to make the logging work? use -s flag while running pytest
    # print(ids)
    decoded = tokenizer.decode(ids)
    assert text == decoded

@pytest.mark.parametrize("tokenizer_factory", [BasicTokenizer, RegexTokenizer])
def test_basic_tokenizer(tokenizer_factory):
    tokenizer = tokenizer_factory()
    tokenizer.train("How are you doing", 257)
    assert tokenizer.decode(tokenizer.encode("abcd")) == "abcd"

if __name__=="__main__":
    pytest.main()