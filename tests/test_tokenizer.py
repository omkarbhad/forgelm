import pytest

from forgelm import BPETokenizer

CORPUS = ("the quick brown fox jumps over the lazy dog. " * 40) + "to be, or not to be: that is the question.\n" * 40


@pytest.fixture(scope="module")
def tok():
    return BPETokenizer.train(CORPUS, vocab_size=300)


@pytest.mark.parametrize("text", [
    "the quick brown fox",
    "",
    "  leading and   multiple   spaces\n\nnewlines\t",
    "snake_case and CamelCase 12345",
    "héllo wörld, 你好, 🙂 emoji",
    "unseen words like zzyzx still work",
])
def test_roundtrip_is_lossless(tok, text):
    assert tok.decode(tok.encode(text)) == text


def test_learns_merges_and_compresses(tok):
    text = "the quick brown fox jumps over the lazy dog."
    assert len(tok.encode(text)) < len(text.encode("utf-8")) / 2


def test_vocab_size(tok):
    assert tok.vocab_size == 300


def test_special_token_is_one_id(tok):
    ids = tok.encode("be<|endoftext|>to")
    assert tok.eot_id in ids and ids.count(tok.eot_id) == 1
    assert tok.decode(ids) == "be<|endoftext|>to"


def test_training_is_deterministic():
    assert BPETokenizer.train(CORPUS, 300).merges == BPETokenizer.train(CORPUS, 300).merges


def test_save_load(tok, tmp_path):
    path = tmp_path / "tok.json"
    tok.save(str(path))
    loaded = BPETokenizer.load(str(path))
    assert loaded.merges == tok.merges
    assert loaded.encode("to be or not") == tok.encode("to be or not")
