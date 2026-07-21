import unicodedata

import textnorm


def test_normalize_composes_nfc():
    decomposed = "a\u0301"

    normalized = textnorm.normalize(decomposed)

    assert normalized == "á"
    assert unicodedata.is_normalized("NFC", normalized)


def test_normalize_collapses_whitespace_and_lowercases():
    assert textnorm.normalize("  ĐIỆN\n  LƯỚI\t ") == "điện lưới"


def test_normalize_preserves_vietnamese_accents():
    assert textnorm.normalize("ỨNG CỨU THÔNG TIN") == "ứng cứu thông tin"


def test_normalize_is_idempotent():
    normalized = textnorm.normalize("  Điện   lưới  ")

    assert textnorm.normalize(normalized) == normalized
