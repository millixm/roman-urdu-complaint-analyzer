"""Transformer experiment: fine-tune a multilingual pre-trained model on the
complaint dataset and compare with the TF-IDF baselines.

NOT yet executed in the progress-report stage (needs internet access to the
Hugging Face Hub, ideally a GPU such as Google Colab T4).

Usage (Colab or local):
    pip install transformers datasets accelerate scikit-learn pandas torch
    python transformer_experiment.py --task category
    python transformer_experiment.py --task sentiment

Model options (change MODEL_NAME):
    xlm-roberta-base                  - multilingual, handles noisy/code-mixed text well
    bert-base-multilingual-cased      - lighter multilingual baseline
"""
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer)

MODEL_NAME = "xlm-roberta-base"
ROOT = Path(__file__).resolve().parent


class DS(torch.utils.data.Dataset):
    def __init__(self, enc, labels):
        self.enc, self.labels = enc, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
        item["labels"] = torch.tensor(self.labels[i])
        return item


def metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds),
            "macro_f1": f1_score(labels, preds, average="macro")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["category", "sentiment"], default="category")
    ap.add_argument("--epochs", type=int, default=5)
    args = ap.parse_args()

    train = pd.read_csv(ROOT / "data" / "train.csv")
    test = pd.read_csv(ROOT / "data" / "test_natural.csv")
    labels = sorted(train[args.task].unique())
    l2i = {l: i for i, l in enumerate(labels)}

    tr, val = train_test_split(train, test_size=0.1, random_state=42, stratify=train[args.task])
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    enc = lambda df: tok(df["text"].tolist(), truncation=True, padding=True, max_length=64)
    ds_tr = DS(enc(tr), tr[args.task].map(l2i).tolist())
    ds_val = DS(enc(val), val[args.task].map(l2i).tolist())
    ds_te = DS(enc(test), test[args.task].map(l2i).tolist())

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(labels))
    targs = TrainingArguments(
        output_dir=str(ROOT / "models" / f"hf_{args.task}"),
        num_train_epochs=args.epochs, per_device_train_batch_size=16,
        per_device_eval_batch_size=32, learning_rate=3e-5, weight_decay=0.01,
        eval_strategy="epoch", save_strategy="no", logging_steps=20, seed=42,
        report_to="none",
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds_tr,
                      eval_dataset=ds_val, compute_metrics=metrics)
    trainer.train()
    res = trainer.evaluate(ds_te)
    print(f"\n[{MODEL_NAME}] {args.task} on natural test set:")
    print(f"  accuracy = {res['eval_accuracy']:.3f}   macro-F1 = {res['eval_macro_f1']:.3f}")


if __name__ == "__main__":
    main()
