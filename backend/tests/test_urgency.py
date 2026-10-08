import pytest

from app.urgency import assess


@pytest.mark.parametrize("text,expected", [
    ("Someone stole my identity and opened a card", "high"),
    ("A loan was opened in my name and I never knew", "high"),
    ("Money was taken from my account without my permission", "high"),
    ("They are threatening to foreclose on my house", "high"),
    ("I am locked out of my account", "high"),
    ("I was charged twice, please dispute", "medium"),
    ("A collector keeps calling me", "medium"),
    ("How do I update my address?", "low"),
    ("Please send me my statement", "low"),
])
def test_urgency_levels(text, expected):
    level, _ = assess(text)
    assert level == expected


def test_reasons_are_returned():
    level, reasons = assess("I was charged twice")
    assert level == "medium" and reasons


def test_high_beats_medium():
    level, _ = assess("collection agency called and my identity was stolen")
    assert level == "high"


@pytest.mark.xfail(reason="Known limitation: keyword rules ignore negation", strict=False)
def test_negation_is_not_understood():
    level, _ = assess("I am not claiming fraud, I just have a question")
    assert level == "low"
