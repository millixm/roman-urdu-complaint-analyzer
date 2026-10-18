# Roman Urdu Student Complaint Analyzer
CSSE 606 - Topic of Current Interest III

Takes a complaint written in Roman Urdu / English / mixed and predicts
**category**, **sentiment** and **priority**.

## Project layout
```
app.py                      Streamlit UI (single complaint + CSV batch mode)
data/build_dataset.py       builds data/train.csv and data/test_natural.csv
src/preprocess.py           Roman Urdu normalization, tokenization, stopwords
src/embeddings.py           averaged Word2Vec transformer
src/train.py                trains/compares models, writes models/ and reports/
src/priority.py             rule-based priority scoring
src/analyzer.py             text -> category, sentiment, priority
transformer_experiment.py   XLM-RoBERTa fine-tuning (run on Colab/GPU)
reports/                    metrics, confusion matrices, error analysis
```

## Run
```
pip install -r requirements.txt
python data/build_dataset.py     # (re)create dataset
python -m src.train              # train + evaluate, saves models/
streamlit run app.py             # launch UI
```

## Notes
* `train.csv` is template-assisted (issue phrases x openers x urgency phrases x
  spelling noise). Cross-validation on it is near 100% and should NOT be read as
  real-world accuracy. `test_natural.csv` (70 hand-written complaints with
  different wording) is the primary evaluation set.
* Replace/extend `train.csv` with real collected complaints, re-run `src.train`.
