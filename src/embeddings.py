"""Averaged Word2Vec features as an sklearn transformer.

Word2Vec is trained inside fit() on the training texts only, so there is no
leakage during cross-validation.
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class AvgWord2Vec(BaseEstimator, TransformerMixin):
    def __init__(self, vector_size=64, window=4, epochs=60, seed=42):
        self.vector_size = vector_size
        self.window = window
        self.epochs = epochs
        self.seed = seed

    def fit(self, X, y=None):
        from gensim.models import Word2Vec  # imported lazily: gensim is optional
        sents = [str(t).split() for t in X]
        self.w2v_ = Word2Vec(
            sentences=sents, vector_size=self.vector_size, window=self.window,
            min_count=1, sg=1, epochs=self.epochs, seed=self.seed, workers=1,
        )
        return self

    def transform(self, X):
        out = np.zeros((len(X), self.vector_size), dtype=np.float32)
        for i, t in enumerate(X):
            vecs = [self.w2v_.wv[w] for w in str(t).split() if w in self.w2v_.wv]
            if vecs:
                out[i] = np.mean(vecs, axis=0)
        return out
