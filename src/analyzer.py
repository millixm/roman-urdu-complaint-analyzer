"""Single entry point used by the UI: text -> category, sentiment, priority."""
import joblib
import pandas as pd
from pathlib import Path

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .preprocess import preprocess, normalize_text
from .priority import priority_score

ROOT = Path(__file__).resolve().parent.parent
_models = {}


def _build_pipeline():
    return Pipeline([
        ("feat", FeatureUnion([
            ("w", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)),
            ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2)),
        ])),
        ("clf", LogisticRegression(max_iter=2000, C=5.0, random_state=42)),
    ])


def _train_fresh():
    """Fallback: train on this machine if the saved models can't be loaded
    (e.g. different scikit-learn version). Takes a few seconds."""
    df = pd.read_csv(ROOT / "data" / "train.csv")
    clean = df["text"].map(preprocess)
    models = {}
    for task in ("category", "sentiment"):
        models[task] = _build_pipeline().fit(clean, df[task])
        (ROOT / "models").mkdir(exist_ok=True)
        joblib.dump(models[task], ROOT / "models" / f"{task}_model.joblib")
    return models


def _load():
    if not _models:
        try:
            _models["category"] = joblib.load(ROOT / "models" / "category_model.joblib")
            _models["sentiment"] = joblib.load(ROOT / "models" / "sentiment_model.joblib")
        except Exception:
            _models.update(_train_fresh())
    return _models


def analyze(text: str) -> dict:
    m = _load()
    clean = preprocess(text)
    out = {"text": text, "normalized": normalize_text(text), "clean": clean}
    for task in ("category", "sentiment"):
        clf = m[task]
        proba = clf.predict_proba([clean])[0]
        classes = list(clf.classes_)
        top = int(proba.argmax())
        out[task] = classes[top]
        out[f"{task}_confidence"] = float(proba[top])
        out[f"{task}_probs"] = dict(zip(classes, map(float, proba)))
    out["priority"] = priority_score(text, out["category"], out["sentiment"])
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(analyze("sir LMS pe assignment submit ni horahi aur deadline kal hai"), indent=2))
