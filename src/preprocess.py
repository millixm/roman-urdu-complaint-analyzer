"""Preprocessing for Roman Urdu / code-mixed complaints.

Steps: lowercase -> strip URLs/digits noise -> normalize spelling variants
-> tokenize -> remove light stopwords (negations are kept on purpose).
"""
import re

# Roman Urdu has no standard spelling, so the same word appears in many forms.
# Map common variants to one canonical form.
NORMALIZE = {
    # negation
    "ni": "nahi", "nai": "nahi", "nhi": "nahi", "nahin": "nahi", "nahee": "nahi", "nh": "nahi",
    # hai / hain
    "h": "hai", "hy": "hai", "hae": "hai", "hn": "hain", "hein": "hain",
    # aur
    "or": "aur", "aor": "aur", "awr": "aur",
    # mein / me
    "me": "mein", "main": "mein", "mai": "mein", "mn": "mein",
    # ho raha / rahi
    "horaha": "ho raha", "hora": "ho raha", "horha": "ho raha", "hrha": "ho raha",
    "horahi": "ho rahi", "hori": "ho rahi", "horhi": "ho rahi",
    "hota": "hota", "hoti": "hoti",
    # pe / par
    "py": "pe", "pr": "pe", "par": "pe",
    # please / urgent / sir
    "plz": "please", "pls": "please", "plss": "please", "plzz": "please",
    "urgnt": "urgent", "urjent": "urgent", "zaruri": "zaroori", "zarori": "zaroori",
    "sr": "sir", "mam": "madam", "maam": "madam", "mum": "madam",
    # time words
    "kl": "kal", "aj": "aaj", "abi": "abhi", "abhi": "abhi", "jldi": "jaldi", "jaldi": "jaldi",
    "tmrw": "kal", "tomorrow": "kal", "today": "aaj", "tonight": "aaj",
    # common verbs / words
    "kr": "kar", "krna": "karna", "krdo": "kar do", "kardo": "kar do", "krdein": "kar dein",
    "kia": "kya", "kiya": "kya", "kese": "kaise", "kesay": "kaise", "kaisay": "kaise",
    "shukria": "shukriya", "shukriyah": "shukriya", "thanks": "shukriya", "thankyou": "shukriya",
    "bhot": "bohat", "bohot": "bohat", "boht": "bohat", "bohut": "bohat",
    "acha": "acha", "achha": "acha", "accha": "acha", "acchi": "acha", "achi": "acha",
    "wifi": "wifi", "wi-fi": "wifi", "internet": "wifi",
    "assignmnt": "assignment", "asignment": "assignment", "assingment": "assignment",
    "regestration": "registration", "registeration": "registration", "reg": "registration",
    "attendence": "attendance", "attendnce": "attendance",
    "challen": "challan", "chalan": "challan", "voucher": "challan",
    "exm": "exam", "paper": "exam", "papers": "exam", "imtihan": "exam",
    "feee": "fee", "fees": "fee", "dues": "fee",
}

# Light stopword list. Negation words (nahi, na, mat, bina) are intentionally NOT removed.
STOPWORDS = {
    "sir", "madam", "please", "hello", "salam", "assalam", "alaikum", "o", "dear", "respected",
    "ka", "ki", "ke", "ko", "se", "ne", "ek", "yeh", "ye", "woh", "wo", "is", "us",
    "the", "a", "an", "to", "of", "and", "i", "my", "me", "on", "in", "at", "for", "it",
    "mera", "meri", "mere", "main", "hum", "ap", "aap", "tha", "thi", "thay", "the",
}

_TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?|\d+")


def normalize_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^a-z0-9'\s]", " ", text)  # drop punctuation/emoji
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)  # sooooon -> soon
    tokens = text.split()
    out = []
    for tok in tokens:
        out.extend(NORMALIZE.get(tok, tok).split())
    return " ".join(out)


def tokenize(text: str):
    return _TOKEN_RE.findall(normalize_text(text))


def remove_stopwords(tokens):
    return [t for t in tokens if t not in STOPWORDS]


def preprocess(text: str) -> str:
    """Full pipeline: normalize -> tokenize -> stopword removal -> joined string."""
    return " ".join(remove_stopwords(tokenize(text)))


if __name__ == "__main__":
    demo = "Sir LMS pe assignment submit ni horahi aur deadline kal hai!!!"
    print("raw        :", demo)
    print("normalized :", normalize_text(demo))
    print("tokens     :", tokenize(demo))
    print("final      :", preprocess(demo))
