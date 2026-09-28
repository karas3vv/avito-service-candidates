"""Sparse lexical indices. CSC postings avoid scanning the corpus per query."""
import re
from functools import lru_cache

import joblib
import numpy as np
import snowballstemmer
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

STEMMER = snowballstemmer.stemmer('russian')


def clean(value):
    return re.sub(r'\s+', ' ', str(value or '').lower()).strip()


@lru_cache(maxsize=300_000)
def stem(word):
    return STEMMER.stemWord(word)


def tokenize(text):
    return [stem(w) for w in re.findall(r'[а-яёa-z0-9]+', text.lower().replace('ё', 'е'))
            if len(w) > 1]


def top(values, n):
    n = min(n, len(values))
    return np.argpartition(values, -n)[-n:] if n else np.empty(0, dtype=np.int32)


class TextIndex:
    def __init__(self, cache, name, documents, kind='tfidf', **kwargs):
        base = cache / name
        if base.with_suffix('.joblib').exists():
            self.vectorizer, self.idf = joblib.load(base.with_suffix('.joblib'))
            self.matrix = sparse.load_npz(base.with_suffix('.npz'))
            self.cover = (sparse.load_npz(cache / (name + '_cover.npz'))
                          if kind == 'bm25' else None)
        else:
            print('Building index:', name, flush=True)
            if kind == 'bm25':
                self.vectorizer = CountVectorizer(tokenizer=tokenize, token_pattern=None,
                                                  dtype=np.float32, **kwargs)
                x = self.vectorizer.fit_transform(documents).tocsr()
                lengths = np.asarray(x.sum(axis=1)).ravel()
                df = np.bincount(x.indices, minlength=x.shape[1])
                self.idf = np.log1p((x.shape[0] - df + .5) / (df + .5)).astype(np.float32)
                cover = x.copy()
                cover.data = self.idf[cover.indices]
                self.cover = cover.tocsc()
                del cover
                # BM25 with k1=1.2, b=0.65. IDF is positive (Robertson variant).
                norm = 1.2 * (.35 + .65 * lengths / max(lengths.mean(), 1))
                x.data = (x.data * 2.2 / (x.data + np.repeat(norm, np.diff(x.indptr)))
                          * self.idf[x.indices])
                self.matrix = x.tocsc()
                sparse.save_npz(cache / (name + '_cover.npz'), self.cover)
            else:
                self.vectorizer = TfidfVectorizer(dtype=np.float32, sublinear_tf=True, **kwargs)
                self.matrix = self.vectorizer.fit_transform(documents).tocsc()
                self.idf = None
                self.cover = None
            sparse.save_npz(base.with_suffix('.npz'), self.matrix)
            joblib.dump((self.vectorizer, self.idf), base.with_suffix('.joblib'))
        self.kind = kind

    def prepare(self, queries):
        self.queries = self.vectorizer.transform(queries).tocsr()

    def score(self, qi):
        q = self.queries[qi]
        score = (q @ self.matrix.T).toarray().ravel()
        if self.kind == 'bm25':
            denominator = max(float((q @ self.idf).item()), 1e-6)
            coverage = (q @ self.cover.T).toarray().ravel() / denominator
            return score / denominator, coverage
        return score
