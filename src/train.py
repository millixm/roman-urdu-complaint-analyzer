"""Train + evaluate category and sentiment classifiers, and evaluate priority.

Run from project root:  python -m src.train
Outputs: models/*.joblib, reports/results.json, reports/*.png
"""
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from .preprocess import preprocess
from .embeddings import AvgWord2Vec

try:
    import gensim  # noqa: F401
    HAS_GENSIM = True
except ImportError:
    HAS_GENSIM = False
from .priority import priority_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
SEED = 42

train = pd.read_csv(ROOT / "data" / "train.csv")
test = pd.read_csv(ROOT / "data" / "test_natural.csv")
for df in (train, test):
    df["clean"] = df["text"].map(preprocess)
    df["raw"] = df["text"].str.lower()


def word_tfidf():
    return TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)


def char_tfidf():
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2)


def combo_tfidf():
    return FeatureUnion([("w", word_tfidf()), ("c", char_tfidf())])


FEATURES = {
    "TF-IDF word": word_tfidf,
    "TF-IDF char": char_tfidf,
    "TF-IDF word+char": combo_tfidf,
    "Word2Vec (avg)": lambda: AvgWord2Vec(),
}
MODELS = {
    "LogReg": lambda: LogisticRegression(max_iter=2000, C=5.0, random_state=SEED),
    "LinearSVM": lambda: LinearSVC(C=1.0, random_state=SEED),
    "NaiveBayes": lambda: MultinomialNB(alpha=0.3),
}


def make_pipeline(feat, model):
    return Pipeline([("feat", FEATURES[feat]()), ("clf", MODELS[model]())])


def evaluate(task, text_col="clean"):
    rows = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for feat in FEATURES:
        for model in MODELS:
            if feat == "Word2Vec (avg)" and (model == "NaiveBayes" or not HAS_GENSIM):
                continue  # NB needs non-negative features; Word2Vec needs gensim
            pipe = make_pipeline(feat, model)
            cv_pred = cross_val_predict(pipe, train[text_col], train[task], cv=skf)
            pipe.fit(train[text_col], train[task])
            te_pred = pipe.predict(test[text_col])
            rows.append({
                "task": task, "features": feat, "model": model,
                "cv_acc": accuracy_score(train[task], cv_pred),
                "cv_f1": f1_score(train[task], cv_pred, average="macro"),
                "test_acc": accuracy_score(test[task], te_pred),
                "test_f1": f1_score(test[task], te_pred, average="macro"),
            })
    return pd.DataFrame(rows)


def main():
    results = {}
    all_rows = []
    for task in ["category", "sentiment"]:
        df = evaluate(task)
        all_rows.append(df)
        print(f"\n=== {task.upper()} ===")
        print(df.drop(columns="task").round(3).to_string(index=False))
    res = pd.concat(all_rows)
    res.to_csv(ROOT / "reports" / "model_comparison.csv", index=False)

    # ---- preprocessing ablation (final config, raw text vs normalized text) ----
    abl = []
    for task in ["category", "sentiment"]:
        for name, col in [("raw lowercase text", "raw"), ("with our preprocessing", "clean")]:
            pipe = make_pipeline("TF-IDF word", "LogReg")
            pipe.fit(train[col], train[task])
            p = pipe.predict(test[col])
            abl.append({"task": task, "input": name,
                        "test_acc": accuracy_score(test[task], p),
                        "test_f1": f1_score(test[task], p, average="macro")})
    abl = pd.DataFrame(abl)
    abl.to_csv(ROOT / "reports" / "preprocessing_ablation.csv", index=False)
    print("\n=== PREPROCESSING ABLATION (TF-IDF word + LogReg) ===")
    print(abl.round(3).to_string(index=False))

    # ---- final models (chosen a priori: word+char TF-IDF + LogReg, robust to spelling noise) ----
    final = {}
    for task in ["category", "sentiment"]:
        pipe = make_pipeline("TF-IDF word+char", "LogReg")
        pipe.fit(train["clean"], train[task])
        final[task] = pipe
        joblib.dump(pipe, ROOT / "models" / f"{task}_model.joblib")
        pred = pipe.predict(test["clean"])
        results[task] = {
            "test_acc": accuracy_score(test[task], pred),
            "test_f1": f1_score(test[task], pred, average="macro"),
            "report": classification_report(test[task], pred, output_dict=True, zero_division=0),
        }
        labels = sorted(train[task].unique())
        cm = confusion_matrix(test[task], pred, labels=labels)
        plt.figure(figsize=(6.2, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, cbar=False)
        plt.xlabel("Predicted"); plt.ylabel("Actual")
        plt.title(f"{task.capitalize()} - confusion matrix (natural test set)")
        plt.xticks(rotation=30, ha="right"); plt.tight_layout()
        plt.savefig(ROOT / "reports" / f"cm_{task}.png", dpi=160); plt.close()
        print(f"\n[final {task}] test acc={results[task]['test_acc']:.3f}  macro-F1={results[task]['test_f1']:.3f}")
        print(classification_report(test[task], pred, zero_division=0))

    # ---- priority ----
    pred_cat = final["category"].predict(test["clean"])
    pred_sent = final["sentiment"].predict(test["clean"])
    prio_gold = [priority_score(t, c, s)["label"] for t, c, s in zip(test.text, test.category, test.sentiment)]
    prio_pred = [priority_score(t, c, s)["label"] for t, c, s in zip(test.text, pred_cat, pred_sent)]
    prio_ml = make_pipeline("TF-IDF word+char", "LogReg").fit(train["clean"], train["priority"]).predict(test["clean"])
    prio = {
        "rule_based_with_gold_cat_sent": accuracy_score(test.priority, prio_gold),
        "rule_based_end_to_end": accuracy_score(test.priority, prio_pred),
        "ml_tfidf_logreg": accuracy_score(test.priority, prio_ml),
    }
    prio_f1 = {
        "rule_based_with_gold_cat_sent": f1_score(test.priority, prio_gold, average="macro"),
        "rule_based_end_to_end": f1_score(test.priority, prio_pred, average="macro"),
        "ml_tfidf_logreg": f1_score(test.priority, prio_ml, average="macro"),
    }
    results["priority"] = {"acc": prio, "macro_f1": prio_f1}
    print("\n=== PRIORITY (natural test) ===")
    for k in prio:
        print(f"{k:32s} acc={prio[k]:.3f} macro-F1={prio_f1[k]:.3f}")
    labs = ["High", "Medium", "Low"]
    cm = confusion_matrix(test.priority, prio_pred, labels=labs)
    plt.figure(figsize=(5, 4.2))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges", xticklabels=labs, yticklabels=labs, cbar=False)
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Priority (rule-based, end-to-end)")
    plt.tight_layout(); plt.savefig(ROOT / "reports" / "cm_priority.png", dpi=160); plt.close()

    # ---- full-pipeline error list ----
    err = test.copy()
    err["pred_category"], err["pred_sentiment"], err["pred_priority"] = pred_cat, pred_sent, prio_pred
    bad = err[(err.category != err.pred_category) | (err.sentiment != err.pred_sentiment) | (err.priority != err.pred_priority)]
    bad[["text", "category", "pred_category", "sentiment", "pred_sentiment", "priority", "pred_priority"]].to_csv(
        ROOT / "reports" / "error_analysis.csv", index=False)
    all_three = ((err.category == err.pred_category) & (err.sentiment == err.pred_sentiment) & (err.priority == err.pred_priority)).mean()
    results["all_three_correct"] = float(all_three)
    print(f"\nAll three outputs correct on same complaint: {all_three:.3f}")

    # ---- comparison chart (category, natural test macro-F1) ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, task in zip(axes, ["category", "sentiment"]):
        d = res[res.task == task].copy()
        d["label"] = d["features"] + "\n" + d["model"]
        d = d.sort_values("test_f1")
        ax.barh(d["label"], d["test_f1"], color="#2f6db5")
        ax.set_xlim(0, 1); ax.set_title(f"{task.capitalize()} - macro-F1 (natural test)")
        for y, v in enumerate(d["test_f1"]):
            ax.text(v + 0.01, y, f"{v:.2f}", va="center", fontsize=7)
        ax.tick_params(axis="y", labelsize=6.5)
    plt.tight_layout(); plt.savefig(ROOT / "reports" / "model_comparison.png", dpi=160); plt.close()

    (ROOT / "reports" / "results.json").write_text(json.dumps(results, indent=2, default=float))
    print("\nSaved models + reports.")


if __name__ == "__main__":
    main()
