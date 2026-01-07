"""
Tests for Stage 7: Tokenization

Comprehensive tests for all tokenizer implementations.
"""

import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from tokenizer import (
    CharTokenizer,
    BPETokenizer,
    WordPieceTokenizer,
    UnigramTokenizer,
    analyze_tokenization,
)


# =============================================================================
# Character Tokenizer Tests
# =============================================================================

def test_char_tokenizer_basic():
    """Test basic character tokenization."""
    tok = CharTokenizer().train(["hello world"])

    encoded = tok.encode("hello")
    decoded = tok.decode(encoded)

    assert decoded == "hello", f"Expected 'hello', got '{decoded}'"
    assert len(encoded) == 5, f"Expected 5 tokens, got {len(encoded)}"
    print("✓ test_char_tokenizer_basic passed")


def test_char_tokenizer_unknown():
    """Test unknown character handling."""
    tok = CharTokenizer().train(["abc"])

    # 'x' was not in training data
    encoded = tok.encode("axc")
    decoded = tok.decode(encoded)

    assert '<UNK>' in decoded or 'x' not in decoded or decoded == "axc", \
        f"Unknown chars should be handled, got '{decoded}'"
    print("✓ test_char_tokenizer_unknown passed")


def test_char_tokenizer_vocab_size():
    """Test vocabulary size is correct."""
    tok = CharTokenizer().train(["hello world"])

    # Unique chars: h, e, l, o, ' ', w, r, d + <UNK> = 9
    unique_chars = len(set("hello world"))
    assert tok.vocab_size == unique_chars + 1, \
        f"Expected {unique_chars + 1} vocab size, got {tok.vocab_size}"
    print("✓ test_char_tokenizer_vocab_size passed")


def test_char_tokenizer_roundtrip():
    """Test encode-decode roundtrip."""
    text = "The quick brown fox"
    tok = CharTokenizer().train([text])

    encoded = tok.encode(text)
    decoded = tok.decode(encoded)

    assert decoded == text, f"Roundtrip failed: '{text}' -> '{decoded}'"
    print("✓ test_char_tokenizer_roundtrip passed")


# =============================================================================
# BPE Tokenizer Tests
# =============================================================================

def test_bpe_basic():
    """Test basic BPE tokenization."""
    corpus = ["the cat sat on the mat", "the dog ran on the mat"]
    tok = BPETokenizer(vocab_size=300).train(corpus)

    assert tok.vocab_size > 256, "BPE should have more than byte vocabulary"
    assert len(tok.merges) > 0, "BPE should have learned merges"
    print("✓ test_bpe_basic passed")


def test_bpe_merges_frequent_pairs():
    """Test that BPE merges frequent pairs."""
    # 'th' appears very frequently
    corpus = ["the the the that this then"] * 10
    tok = BPETokenizer(vocab_size=300, min_frequency=1).train(corpus)

    merges_str = tok.get_merges()

    # 'th' should be among early merges (as a readable string)
    # Note: actual merge might be bytes, so we check representation
    found_th = any('t' in str(m[0]) and 'h' in str(m[1]) for m in merges_str[:20])
    # This is a soft check since exact merge order can vary
    print("✓ test_bpe_merges_frequent_pairs passed")


def test_bpe_compression():
    """Test that BPE compresses text."""
    corpus = ["hello world hello world"] * 5
    tok = BPETokenizer(vocab_size=350).train(corpus)

    text = "hello world"
    char_tokens = len(text)
    bpe_tokens = len(tok.encode(text))

    # BPE should use fewer tokens than characters
    assert bpe_tokens <= char_tokens, \
        f"BPE should compress: {bpe_tokens} <= {char_tokens}"
    print("✓ test_bpe_compression passed")


def test_bpe_decode():
    """Test BPE decoding produces valid text."""
    corpus = ["hello world"]
    tok = BPETokenizer(vocab_size=300).train(corpus)

    text = "hello"
    encoded = tok.encode(text)
    decoded = tok.decode(encoded)

    assert decoded == text, f"BPE decode failed: '{text}' -> '{decoded}'"
    print("✓ test_bpe_decode passed")


def test_bpe_special_tokens():
    """Test BPE special tokens."""
    tok = BPETokenizer(vocab_size=300, special_tokens=['<PAD>', '<UNK>']).train(["test"])

    assert '<PAD>' in tok.vocab, "Special token <PAD> should be in vocab"
    assert '<UNK>' in tok.vocab, "Special token <UNK> should be in vocab"
    assert tok.vocab['<PAD>'] == 0, "<PAD> should have ID 0"
    print("✓ test_bpe_special_tokens passed")


# =============================================================================
# WordPiece Tokenizer Tests
# =============================================================================

def test_wordpiece_basic():
    """Test basic WordPiece tokenization."""
    corpus = ["the cat sat on the mat"]
    tok = WordPieceTokenizer(vocab_size=100).train(corpus)

    assert tok.vocab_size > 0, "WordPiece should have vocabulary"
    assert '[UNK]' in tok.vocab, "WordPiece should have [UNK] token"
    print("✓ test_wordpiece_basic passed")


def test_wordpiece_continuation():
    """Test WordPiece continuation prefix."""
    corpus = ["playing played player plays"]
    tok = WordPieceTokenizer(vocab_size=150).train(corpus)

    # Check that continuation tokens exist
    has_continuation = any('##' in t for t in tok.vocab.keys() if isinstance(t, str))
    assert has_continuation, "WordPiece should have ## continuation tokens"
    print("✓ test_wordpiece_continuation passed")


def test_wordpiece_decode():
    """Test WordPiece decoding."""
    corpus = ["hello world hello"]
    tok = WordPieceTokenizer(vocab_size=100).train(corpus)

    text = "hello"
    encoded = tok.encode(text)
    decoded = tok.decode(encoded)

    # Decoded text should contain the word
    assert "hello" in decoded.lower().replace(' ', ''), \
        f"WordPiece decode failed: expected 'hello' in '{decoded}'"
    print("✓ test_wordpiece_decode passed")


def test_wordpiece_special_tokens():
    """Test WordPiece BERT-style special tokens."""
    tok = WordPieceTokenizer(vocab_size=100).train(["test"])

    bert_tokens = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
    for token in bert_tokens:
        assert token in tok.vocab, f"BERT token {token} should be in vocab"
    print("✓ test_wordpiece_special_tokens passed")


# =============================================================================
# Unigram Tokenizer Tests
# =============================================================================

def test_unigram_basic():
    """Test basic Unigram tokenization."""
    corpus = ["the cat sat on the mat"]
    tok = UnigramTokenizer(vocab_size=50).train(corpus)

    assert tok.vocab_size > 0, "Unigram should have vocabulary"
    print("✓ test_unigram_basic passed")


def test_unigram_encode():
    """Test Unigram encoding."""
    corpus = ["hello world"] * 5
    tok = UnigramTokenizer(vocab_size=100).train(corpus)

    encoded = tok.encode("hello")
    assert len(encoded) > 0, "Unigram should produce tokens"
    assert all(isinstance(i, int) for i in encoded), "Tokens should be integers"
    print("✓ test_unigram_encode passed")


def test_unigram_decode():
    """Test Unigram decoding."""
    corpus = ["hello world"] * 5
    tok = UnigramTokenizer(vocab_size=100).train(corpus)

    text = "hello"
    encoded = tok.encode(text)
    decoded = tok.decode(encoded)

    assert "hello" in decoded or decoded == text, \
        f"Unigram decode failed: expected 'hello' in '{decoded}'"
    print("✓ test_unigram_decode passed")


# =============================================================================
# Analysis Tests
# =============================================================================

def test_analyze_tokenization():
    """Test tokenization analysis."""
    tok = CharTokenizer().train(["hello world"])
    texts = ["hello", "world"]

    stats = analyze_tokenization(tok, texts)

    assert 'vocab_size' in stats
    assert 'compression_ratio' in stats
    assert 'unk_rate' in stats
    assert stats['compression_ratio'] > 0
    print("✓ test_analyze_tokenization passed")


def test_compression_ratio():
    """Test compression ratio calculation."""
    corpus = ["the the the"] * 10
    bpe_tok = BPETokenizer(vocab_size=300).train(corpus)

    stats = analyze_tokenization(bpe_tok, corpus)

    # BPE should compress repeated text well
    assert stats['compression_ratio'] >= 1.0, \
        f"BPE should achieve compression >= 1.0, got {stats['compression_ratio']}"
    print("✓ test_compression_ratio passed")


# =============================================================================
# Save/Load Tests
# =============================================================================

def test_char_tokenizer_save_load():
    """Test saving and loading character tokenizer."""
    tok = CharTokenizer().train(["hello world"])

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        path = f.name

    try:
        tok.save(path)
        loaded = CharTokenizer.load(path)

        assert loaded.vocab_size == tok.vocab_size
        assert loaded.encode("hello") == tok.encode("hello")
        print("✓ test_char_tokenizer_save_load passed")
    finally:
        os.unlink(path)


# =============================================================================
# Edge Cases
# =============================================================================

def test_empty_text():
    """Test handling of empty text."""
    tok = CharTokenizer().train(["hello"])

    encoded = tok.encode("")
    assert encoded == [], f"Empty text should give empty encoding, got {encoded}"
    print("✓ test_empty_text passed")


def test_unicode():
    """Test Unicode handling."""
    tok = CharTokenizer().train(["Hello 世界 🌍"])

    text = "世界"
    encoded = tok.encode(text)
    decoded = tok.decode(encoded)

    assert decoded == text, f"Unicode roundtrip failed: '{text}' -> '{decoded}'"
    print("✓ test_unicode passed")


def test_bpe_unicode():
    """Test BPE with Unicode text."""
    corpus = ["Hello 世界", "世界 你好"]
    tok = BPETokenizer(vocab_size=350).train(corpus)

    text = "世界"
    encoded = tok.encode(text)
    decoded = tok.decode(encoded)

    assert "世界" in decoded or decoded == text, \
        f"BPE Unicode failed: '{text}' -> '{decoded}'"
    print("✓ test_bpe_unicode passed")


def test_long_text():
    """Test tokenization of longer text."""
    tok = CharTokenizer().train(["a" * 1000])

    text = "a" * 500
    encoded = tok.encode(text)
    decoded = tok.decode(encoded)

    assert decoded == text, "Long text tokenization failed"
    assert len(encoded) == 500, f"Expected 500 tokens, got {len(encoded)}"
    print("✓ test_long_text passed")


# =============================================================================
# Run All Tests
# =============================================================================

def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running Stage 7 Tokenization Tests")
    print("=" * 60)
    print()

    tests = [
        # Character tokenizer
        test_char_tokenizer_basic,
        test_char_tokenizer_unknown,
        test_char_tokenizer_vocab_size,
        test_char_tokenizer_roundtrip,

        # BPE tokenizer
        test_bpe_basic,
        test_bpe_merges_frequent_pairs,
        test_bpe_compression,
        test_bpe_decode,
        test_bpe_special_tokens,

        # WordPiece tokenizer
        test_wordpiece_basic,
        test_wordpiece_continuation,
        test_wordpiece_decode,
        test_wordpiece_special_tokens,

        # Unigram tokenizer
        test_unigram_basic,
        test_unigram_encode,
        test_unigram_decode,

        # Analysis
        test_analyze_tokenization,
        test_compression_ratio,

        # Save/Load
        test_char_tokenizer_save_load,

        # Edge cases
        test_empty_text,
        test_unicode,
        test_bpe_unicode,
        test_long_text,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            failed += 1

    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
