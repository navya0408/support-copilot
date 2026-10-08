"""Rule-based urgency (high / medium / low).

This is deliberately simple and transparent: the dataset has no urgency label, so we
do not pretend to have a learned model. Every decision comes with the matched reasons.
"""
import re

HIGH = {
    "possible fraud or identity theft": r"identity theft|identity (was|has been)|stole\b|stolen|fraud|scam|hacked|unauthori[sz]ed|used my (name|identity|ssn|social)|in my name|without my (permission|knowledge|consent)",
    "foreclosure, eviction or repossession": r"foreclos|evict|repossess|garnish",
    "legal action": r"lawsuit|\bsued\b|\bcourt\b|summons",
    "locked out or no access to money": r"locked out|can'?t access|cannot access|frozen account|account (is )?frozen",
    "vulnerable customer": r"veteran|servicemember|active duty|military|deployed|elderly|senior citizen|disabled",
}
MEDIUM = {
    "dispute or incorrect information": r"dispute|incorrect|inaccurate|\bwrong\b|not mine",
    "duplicate or unexpected charge": r"charged twice|double charge|unexpected charge|overcharg",
    "collections activity": r"collection|collector|past due",
    "fees": r"late fee|overdraft|penalty",
    "repeated contact without resolution": r"no response|never (received|responded|replied)|still waiting|ignored",
    "harassment": r"harass|threaten",
}


def assess(text: str):
    t = text.lower()
    high = [reason for reason, pat in HIGH.items() if re.search(pat, t)]
    if high:
        return "high", high
    medium = [reason for reason, pat in MEDIUM.items() if re.search(pat, t)]
    if medium:
        return "medium", medium
    return "low", []
