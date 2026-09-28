"""Local candidate retrieval for the Avito services task.

Run: python src/solution.py
Place train.parquet, benchmark_queries.parquet and benchmark_items.parquet
in the project data/ directory. The script uses local scikit-learn models and
validates the CSV before writing answer.csv in the project root.
"""

import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ROOT = PROJECT_ROOT / "data"
OUT = PROJECT_ROOT / "answer.csv"


def clean(s):
    return re.sub(r"\s+", " ", str(s or "").lower()).strip()


def top_indices(values, n, allowed=None):
    if allowed is not None:
        if len(allowed) == 0:
            return np.empty(0, np.int32)
        v = values[allowed]
        k = min(n, len(v))
        return allowed[np.argpartition(v, -k)[-k:]]
    k = min(n, len(values))
    return np.argpartition(values, -k)[-k:]


print("Reading data", flush=True)
items = pd.read_parquet(ROOT / "benchmark_items.parquet").fillna("")
queries = pd.read_parquet(ROOT / "benchmark_queries.parquet").fillna("")
train = pd.read_parquet(ROOT / "train.parquet", columns=[
    "search_query", "search_location_id", "search_infm_params_text",
    "item_id", "item_location_id", "item_microcat_id",
])
item_ids = items.item_id.astype(str).to_numpy()
item_index = {item_id: i for i, item_id in enumerate(item_ids)}
# Hold out clicks for a local sanity check. Sampling distinct phrases reduces
# validation dominated by a few very common queries. These clicks are removed
# from every click-derived prior below.
eligible = train[train.item_id.isin(item_index)].copy()
eligible["query_count"] = eligible.search_query.map(train.search_query.value_counts())
rare = eligible[eligible.query_count <= 5].drop_duplicates("search_query")
common = eligible[eligible.query_count > 5].drop_duplicates("search_query")
val = pd.concat([rare.sample(min(250, len(rare)), random_state=43),
                 common.sample(min(150, len(common)), random_state=44)])
train = train.drop(index=val.index)
val_queries = val[["search_query", "search_location_id", "search_infm_params_text"]].copy()
val_queries["query_id"] = [f"valid_{i:010d}" for i in range(len(val))]
val_queries["search_is_delivery_search"] = 0
val_queries["search_category"] = 114
all_queries = pd.concat([queries, val_queries[queries.columns]], ignore_index=True)
locs = items.item_location_id.to_numpy()
micros = items.item_microcat_id.to_numpy()
reviews = pd.to_numeric(items.item_rating_reviews_count, errors="coerce").fillna(0).to_numpy(np.float32)

# Index locations once; exact locality is the strongest geographic signal.
loc_index = {int(loc): np.flatnonzero(locs == loc).astype(np.int32)
             for loc in np.unique(locs)}

# Click counts provide a prior for the requested service and possible adjacent
# locations. They never target benchmark query IDs.
query_micro = defaultdict(Counter)
loc_click = defaultdict(Counter)
for row in train.itertuples(index=False):
    qtext = clean(row.search_query)
    query_micro[qtext][int(row.item_microcat_id)] += 1
    loc_click[int(row.search_location_id)][int(row.item_location_id)] += 1
query_item_click = defaultdict(Counter)
for qtext, iid in zip(train.search_query, train.item_id):
    if iid in item_index:
        query_item_click[clean(qtext)][iid] += 1

# Similar query spellings let the microcategory prior generalize to unseen
# benchmark phrases, including minor morphology and typing differences.
unique_train_queries = list(query_micro)
qvec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2,
                       max_features=180_000, dtype=np.float32, sublinear_tf=True)
qmat = qvec.fit_transform(unique_train_queries)

print("Fitting text indices", flush=True)
title = items.item_title_raw.map(clean)
params = items.item_infm_params_text.map(clean)
desc = items.item_description_raw.map(clean).str.slice(0, 900)
title_params = (title + " " + params).tolist()
word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=220_000,
                       sublinear_tf=True, dtype=np.float32)
word_mat = word.fit_transform(title_params)
char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                       max_features=350_000, sublinear_tf=True, dtype=np.float32)
char_mat = char.fit_transform(title.tolist())
desc_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=230_000,
                           sublinear_tf=True, dtype=np.float32)
desc_mat = desc_vec.fit_transform(desc.tolist())
param_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100_000,
                            sublinear_tf=True, dtype=np.float32)
param_mat = param_vec.fit_transform(params.tolist())
print("Matrices", word_mat.shape, word_mat.nnz, char_mat.nnz,
      desc_mat.nnz, flush=True)

qtexts = all_queries.search_query.map(clean).tolist()
qfilters = all_queries.search_infm_params_text.map(clean).tolist()
word_q = word.transform(qtexts)
char_q = char.transform(qtexts)
desc_q = desc_vec.transform(qtexts)
param_q = param_vec.transform(qfilters)
near_q = qvec.transform(qtexts)

# Query-specific microcategory probability: exact click distribution when
# available, smoothed with nearest historical queries.
micro_priors = []
for qi, qtext in enumerate(qtexts):
    prior = Counter()
    if qtext in query_micro:
        for m, n in query_micro[qtext].items():
            prior[m] += 2.0 * n / max(3, sum(query_micro[qtext].values()))
    sim = (qmat @ near_q[qi].T).tocoo()
    if sim.nnz:
        order = np.argsort(sim.data)[-20:]
        for j in order:
            score = float(sim.data[j])
            if score < 0.36:
                continue
            cc = query_micro[unique_train_queries[int(sim.row[j])]]
            scale = score ** 4 / max(10, sum(cc.values()))
            for m, n in cc.items():
                prior[m] += scale * n
    micro_priors.append(prior)

print("Scoring benchmark queries", flush=True)
predictions = []
for qi, row in enumerate(all_queries.itertuples(index=False)):
    # Four complementary lexical representations cover concise titles,
    # morphology and descriptions. Sparse multiplication avoids an all-pairs
    # query/item matrix in memory.
    wt = (word_mat @ word_q[qi].T).toarray().ravel()
    ct = (char_mat @ char_q[qi].T).toarray().ravel()
    dt = (desc_mat @ desc_q[qi].T).toarray().ravel()
    pt = (param_mat @ param_q[qi].T).toarray().ravel()
    lexical = 2.0 * wt + 1.5 * ct + 0.8 * dt + 0.45 * pt

    local = int(row.search_location_id)
    geo = np.zeros(len(items), np.float32)
    geo[locs == local] = 1.0
    lc = loc_click.get(local, {})
    if lc:
        max_count = max(lc.values())
        for loc, count in lc.most_common(25):
            if loc in loc_index:
                geo[loc_index[loc]] = np.maximum(
                    geo[loc_index[loc]], 0.75 * count / max_count)

    prior = micro_priors[qi]
    micro_score = np.fromiter((prior.get(int(m), 0.0) for m in micros),
                              dtype=np.float32, count=len(items))
    if micro_score.max() > 0:
        micro_score /= micro_score.max()
    quality = 0.035 * np.log1p(reviews) / np.log(500)
    score = lexical + 0.38 * geo + 0.27 * micro_score + quality
    qclicks = query_item_click.get(qtexts[qi], {})
    for iid, n in qclicks.items():
        score[item_index[iid]] += 0.2 * min(1, np.log1p(n) / 4)

    # Guarantee geographic coverage even when global lexical matches are
    # concentrated in large cities. Reserve a few slots for nonlocal items.
    local_ids = loc_index.get(local)
    if local_ids is not None and len(local_ids) >= 50:
        a = top_indices(score, 43, local_ids)
        b = top_indices(score, 60)
        ordered = sorted(set(a) | set(b), key=lambda i: score[i], reverse=True)
        chosen = ordered[:50]
    else:
        chosen = top_indices(score, 50)
        chosen = sorted(chosen, key=lambda i: score[i], reverse=True)
    predictions.append([item_ids[i] for i in chosen])
    if (qi + 1) % 250 == 0:
        print("Scored", qi + 1, "of", len(all_queries), flush=True)

val_predictions = predictions[len(queries):]
recall = np.mean([str(item_id) in set(pred)
                  for item_id, pred in zip(val.item_id, val_predictions)])
print("Held-out single-click Recall@50", round(float(recall), 4),
      "n=", len(val), flush=True)
predictions = predictions[:len(queries)]

answer = pd.DataFrame({"query_id": queries.query_id.astype(str),
                       "answer": [" ".join(p) for p in predictions]})
assert len(answer) == len(queries) == answer.query_id.nunique()
assert answer.query_id.str.len().eq(16).all()
valid = set(item_ids)
for s in answer.answer:
    ids = s.split()
    assert len(ids) <= 50 and len(ids) == len(set(ids))
    assert all(i in valid for i in ids)
OUT.parent.mkdir(exist_ok=True)
answer.to_csv(OUT, index=False)
print("Saved", OUT, flush=True)
