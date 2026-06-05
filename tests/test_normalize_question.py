import pytest
from app.services.search import normalize_question


# ?? Basic normalisation ???????????????????????????????????????????????????????


def test_lowercases_input():
    orginal = "What Is Grace?"
    result = normalize_question(orginal)
    print("Original:", orginal)
    print("Normalized:", result)
    assert result == normalize_question("what is grace?")


def test_removes_stopwords():
    # "what", "is", and "the" are stop words
    orginal = "What is the meaning of grace?"
    result = normalize_question(orginal)
    assert "what" not in result.split()
    assert "is" not in result.split()
    assert "the" not in result.split()
    print("Original:", orginal)
    print("Normalized:", result)


def test_removes_punctuation():
    original = "Does God forgive sins?"
    result = normalize_question(original)
    print("Original:", original)
    print("Normalized:", result)
    assert "?" not in result
    assert "." not in result


def test_lemmatises_words():
    # "running" should be lemmatised to "run"
    original = "running toward salvation"
    result = normalize_question(original)
    print("Original:", original)
    print("Normalized:", result)
    assert "run" in result.split()
    assert "running" not in result.split()


def test_lemmatises_plural():
    # "sins" should be lemmatised to "sin"
    original = "Are sins forgiven?"
    result = normalize_question(original)
    print("Original:", original)
    print("Normalized:", result)
    assert "sin" in result.split()
    assert "sins" not in result.split()


# ?? Edge cases ????????????????????????????????????????????????????????????????


def test_empty_string_returns_empty():
    result = normalize_question("")
    print("Original: (empty string)")
    print("Normalized:", repr(result))
    assert result == ""


def test_only_stopwords_and_punctuation():
    # A sentence made entirely of stop words should return an empty string
    original = "is it a the"
    result = normalize_question(original)
    print("Original:", original)
    print("Normalized:", repr(result))
    assert result == ""


def test_single_content_word():
    result = normalize_question("grace")
    print("Original: grace")
    print("Normalized:", result)
    assert result == "grace"


# ?? Theological examples ??????????????????????????????????????????????????????


def test_typical_bible_question():
    original = "What does the Bible say about salvation?"
    result = normalize_question(original)
    print("Original:", original)
    print("Normalized:", result)
    tokens = result.split()
    assert "bible" in tokens
    assert "salvation" in tokens
    # stop words stripped
    assert "what" not in tokens
    assert "the" not in tokens
    assert "about" not in tokens


def test_equivalent_questions_produce_same_tokens():
    original1 = "What is justification by faith?"
    original2 = "What does justification by faith mean?"
    q1 = normalize_question(original1)
    q2 = normalize_question(original2)
    print("Original 1:", original1)
    print("Normalized 1:", q1)
    print("Original 2:", original2)
    print("Normalized 2:", q2)
    tokens1 = set(q1.split())
    tokens2 = set(q2.split())
    # Both should contain "justif" root and "faith"
    assert "justification" in tokens1 or any(t.startswith("justif") for t in tokens1)
    assert "faith" in tokens1
    assert "faith" in tokens2
