from app.textclean import clean_text


def test_masks_urls_and_amounts_are_removed():
    out = clean_text("I XXXX owe {$250.00} on XX/XX/XXXX! See http://a.com NOW")
    assert out == "i owe on see now"


def test_lowercase_and_spaces_collapsed():
    assert clean_text("  HELLO   World  ") == "hello world"


def test_keeps_apostrophes_and_numbers():
    assert clean_text("I can't pay 30 days") == "i can't pay 30 days"


def test_empty_and_non_string_input():
    assert clean_text("") == ""
    assert clean_text(None) == "none"        # str(None); callers never pass None, this just must not crash
