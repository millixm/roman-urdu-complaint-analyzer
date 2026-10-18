"""Priority scoring.

priority_score = category_weight + sentiment_weight + urgency_score

* category_weight  : how critical the complaint area usually is
* sentiment_weight : negative complaints are more pressing than queries
* urgency_score    : urgency words/phrases found in the text (time pressure,
                     deadlines, explicit "urgent"), halved for non-negative text

The score is mapped to High / Medium / Low with two thresholds.
"""
from .preprocess import normalize_text

CATEGORY_WEIGHT = {
    "Examination": 1.5,
    "Fee": 1.0,
    "Registration": 1.0,
    "Attendance": 1.0,
    "LMS": 0.5,
    "Technical": 0.5,
    "General Query": 0.0,
}

SENTIMENT_WEIGHT = {"Negative": 2.0, "Neutral": 0.0, "Positive": -1.0}

# token -> weight (text is normalized first, so spelling variants are already merged)
URGENT_TOKENS = {
    "urgent": 3.0, "deadline": 3.0, "asap": 3.0, "emergency": 3.0, "immediately": 3.0,
    "fori": 3.0, "zaroori": 2.0, "jaldi": 2.0, "abhi": 2.0, "aaj": 2.0, "kal": 2.0,
    "raat": 1.0, "subah": 0.5,
}
URGENT_PHRASES = {
    "last date": 3.0, "last day": 3.0, "ab tak": 0.0, "abhi tak": -2.0,  # "abhi tak" = "still not" -> not time pressure
    "ke bawajood": 0.0, "kal tak": 1.0,
}

HIGH_THRESHOLD = 4.5
MEDIUM_THRESHOLD = 2.0
URGENCY_CAP = 5.0


def urgency_score(text: str) -> tuple[float, list[str]]:
    norm = " " + normalize_text(text) + " "
    score, hits = 0.0, []
    # phrases first; blank them out so their tokens aren't double counted
    for phrase, w in URGENT_PHRASES.items():
        if f" {phrase} " in norm:
            score += w
            if w > 0:
                hits.append(phrase)
            norm = norm.replace(f" {phrase} ", " ")
    for tok in norm.split():
        if tok in URGENT_TOKENS:
            score += URGENT_TOKENS[tok]
            hits.append(tok)
    return min(max(score, 0.0), URGENCY_CAP), hits


def priority_score(text: str, category: str, sentiment: str) -> dict:
    urg, hits = urgency_score(text)
    if sentiment != "Negative":
        urg *= 0.5
    total = CATEGORY_WEIGHT.get(category, 0.0) + SENTIMENT_WEIGHT.get(sentiment, 0.0) + urg
    if total >= HIGH_THRESHOLD:
        label = "High"
    elif total >= MEDIUM_THRESHOLD:
        label = "Medium"
    else:
        label = "Low"
    return {"label": label, "score": round(total, 2), "urgency_hits": hits}


if __name__ == "__main__":
    t = "sir LMS pe assignment submit ni horahi aur deadline kal hai"
    print(priority_score(t, "LMS", "Negative"))
