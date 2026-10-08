from app import messages


def test_every_status_has_a_message_entry():
    statuses = {v for k, v in vars(messages).items() if k.startswith("STATUS_") and isinstance(v, str)
                and k != "STATUS_MESSAGES"}
    assert statuses == set(messages.STATUS_MESSAGES)


def test_refused_statuses_explain_why():
    for s in messages.REFUSED_STATUSES:
        assert messages.STATUS_MESSAGES[s]


def test_prompt_placeholders():
    out = messages.PROMPT.format(context="C", category="X", ticket="T")
    assert "INSUFFICIENT_CONTEXT" in out and "<ticket>" in out
