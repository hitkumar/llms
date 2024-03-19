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

special_tokens = {
    '<|endoftext|>': 100257,
    '<|fim_prefix|>': 100258,
    '<|fim_middle|>': 100259,
    '<|fim_suffix|>': 100260,
    '<|endofprompt|>': 100276
}

llama_text = """
<|endoftext|>The llama (/ˈlɑːmə/; Spanish pronunciation: [ˈʎama] or [ˈʝama]) (Lama glama) is a domesticated South American camelid, widely used as a meat and pack animal by Andean cultures since the pre-Columbian era.
Llamas are social animals and live with others as a herd. Their wool is soft and contains only a small amount of lanolin.[2] Llamas can learn simple tasks after a few repetitions. When using a pack, they can carry about 25 to 30% of their body weight for 8 to 13 km (5–8 miles).[3] The name llama (in the past also spelled "lama" or "glama") was adopted by European settlers from native Peruvians.[4]
The ancestors of llamas are thought to have originated from the Great Plains of North America about 40 million years ago, and subsequently migrated to South America about three million years ago during the Great American Interchange. By the end of the last ice age (10,000–12,000 years ago), camelids were extinct in North America.[3] As of 2007, there were over seven million llamas and alpacas in South America and over 158,000 llamas and 100,000 alpacas, descended from progenitors imported late in the 20th century, in the United States and Canada.[5]
<|fim_prefix|>In Aymara mythology, llamas are important beings. The Heavenly Llama is said to drink water from the ocean and urinates as it rains.[6] According to Aymara eschatology,<|fim_suffix|> where they come from at the end of time.[6]<|fim_middle|> llamas will return to the water springs and ponds<|endofprompt|>
""".strip()

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

@pytest.mark.parametrize("tokenizer_factory", [BasicTokenizer, RegexTokenizer])
def test_wikipedia_example(tokenizer_factory):
    """
    Unit test following the example in https://en.wikipedia.org/wiki/Byte_pair_encoding

    Running bpe on "aaabdaaabac" will result in XdXac where
    X=ZY
    Y=ab
    Z=aa

    Z = 256, Y = 257, X = 258, a = 97 (ascii)
    so output will be
    [258, 100, 258, 97, 99]
    """
    tokenizer = tokenizer_factory()
    text = "aaabdaaabac"
    tokenizer.train(text, 256 + 3)
    ids = tokenizer.encode(text)
    assert tokenizer.decode(ids) == text
    assert ids == [258, 100, 258, 97, 99]

def test_save_load_basic_tokenizer():
    # train on more complex text which also includes special tokens
    text = "How are you doing"
    tokenizer = BasicTokenizer()
    train_tokenizer(tokenizer)
    assert tokenizer.decode(tokenizer.encode(text)) == text
    ids = tokenizer.encode(text)

    # save load test
    tokenizer.save("test_tokenizer_tmp")
    tokenizer_saved = BasicTokenizer()
    tokenizer_saved.load("test_tokenizer_tmp.model")
    ids_saved = tokenizer_saved.encode(text)
    assert tokenizer_saved.decode(ids) == text
    assert tokenizer_saved.decode(ids_saved) == text
    assert ids_saved == ids

    # delete the temp files
    for file in ["test_tokenizer_tmp.vocab", "test_tokenizer_tmp.model"]:
        os.remove(file)

@pytest.mark.parametrize("special_tokens", [{}, special_tokens])
def test_save_load_regex_tokenizer(special_tokens):
    # train on more complex text which also includes special tokens
    text = llama_text
    tokenizer = RegexTokenizer()
    # 64 merges
    tokenizer.train(llama_text, 256 + 64)
    tokenizer.register_special_tokens(special_tokens)
    # basic check first
    ids = tokenizer.encode(text, "all")
    assert tokenizer.decode(ids) == text

    # now test save and load
    tokenizer.save("test_tokenizer_tmp")
    tokenizer_saved = RegexTokenizer()
    tokenizer_saved.load("test_tokenizer_tmp.model")
    ids_saved = tokenizer_saved.encode(text, "all")
    assert tokenizer_saved.decode(ids) == text
    assert tokenizer_saved.decode(ids_saved) == text
    assert ids_saved == ids

    # delete the temp files
    for file in ["test_tokenizer_tmp.vocab"]:
        os.remove(file)

if __name__=="__main__":
    pytest.main()