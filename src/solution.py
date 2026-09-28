"""BM25/TF-IDF candidate union and supervised selection of 50 items.

Run with --stage build, train, or all. Index and feature caches are local;
the entire pipeline needs no API or downloaded pretrained model.
"""
import argparse
import gc
import json
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from retrieval import TextIndex, clean, top, tokenize


SEARCH = ['search_query', 'search_location_id', 'search_infm_params_text',
          'search_is_delivery_search', 'search_category']


def make_split(train, item_ids):
    """Keep the original 400-query test; tune and training texts are disjoint.

    All known positive items for each sampled query context become labels.
    Remove every occurrence of each labelled (query text, item) pair from
    click-derived features, including occurrences in other query contexts.
    """
    eligible = train[train.item_id.isin(set(item_ids))].copy()
    eligible['query_count'] = eligible.search_query.map(train.search_query.value_counts())
    rare = eligible[eligible.query_count <= 5].drop_duplicates('search_query')
    common = eligible[eligible.query_count > 5].drop_duplicates('search_query')
    test = pd.concat([rare.sample(250, random_state=43), common.sample(150, random_state=44)])
    used = set(test.search_query)
    tune = pd.concat([rare[~rare.search_query.isin(used)].sample(250, random_state=143),
                      common[~common.search_query.isin(used)].sample(150, random_state=144)])
    used.update(tune.search_query)
    fit = pd.concat([rare[~rare.search_query.isin(used)].sample(1500, random_state=243),
                     common[~common.search_query.isin(used)].sample(900, random_state=244)])
    groups = eligible.groupby(SEARCH, sort=False, dropna=False).item_id.agg(lambda s: sorted(set(s)))
    selected = []
    remove = set()
    for split, part in [('fit', fit), ('tune', tune), ('test', test)]:
        for _, row in part.iterrows():
            labels = groups.loc[tuple(row[SEARCH])]
            d = {key: row[key] for key in SEARCH}
            d.update(split=split, labels=labels, reference_item=row.item_id,
                     query_count=int(row.query_count), query_id=f'{split}_{len(selected)}')
            selected.append(d)
            remove.update((clean(row.search_query), iid) for iid in labels)
    keep = [(clean(q), iid) not in remove for q, iid in zip(train.search_query, train.item_id)]
    return pd.DataFrame(selected), train.loc[keep].copy()


def build(args):
    cache = args.cache_dir
    cache.mkdir(parents=True, exist_ok=True)
    items = pd.read_parquet(args.data_dir / 'benchmark_items.parquet').fillna('')
    queries = pd.read_parquet(args.data_dir / 'benchmark_queries.parquet').fillna('')
    train = pd.read_parquet(args.data_dir / 'train.parquet', columns=SEARCH + [
        'item_id', 'item_location_id', 'item_microcat_id']).fillna('')
    ids = items.item_id.to_numpy()
    idmap = {iid: i for i, iid in enumerate(ids)}
    sampled, history = make_split(train, ids)
    queries['split'] = 'benchmark'
    queries['labels'] = [[] for _ in range(len(queries))]
    queries['reference_item'] = ''
    queries['query_count'] = 0
    queries = pd.concat([sampled, queries], ignore_index=True)
    print('Query split:', queries.split.value_counts().to_dict(), 'history:', len(history), flush=True)
    del train

    # Click statistics exclude all selected label interactions.
    qm, lc, qc = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    popularity = Counter(history.item_id)
    for r in history.itertuples(index=False):
        q = clean(r.search_query)
        qm[q][int(r.item_microcat_id)] += 1
        lc[int(r.search_location_id)][int(r.item_location_id)] += 1
        if r.item_id in idmap:
            qc[q][r.item_id] += 1
    del history
    qnames = list(qm)
    qv = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), min_df=2,
                         max_features=180_000, dtype=np.float32, sublinear_tf=True)
    qindex = qv.fit_transform(qnames).tocsc()
    qtexts = queries.search_query.map(clean).tolist()
    qfilters = queries.search_infm_params_text.map(clean).tolist()
    qvectors = qv.transform(qtexts)

    title = items.item_title_raw.map(clean).tolist()
    params = items.item_infm_params_text.map(clean).tolist()
    descriptions = items.item_description_raw.map(clean)
    indices = [
        TextIndex(cache, 'word', [t + ' ' + p for t, p in zip(title, params)],
                  ngram_range=(1, 2), min_df=2, max_features=220_000),
        TextIndex(cache, 'char', title, analyzer='char_wb', ngram_range=(3, 5),
                  min_df=2, max_features=350_000),
        TextIndex(cache, 'desc', descriptions.str.slice(0, 900).tolist(),
                  ngram_range=(1, 2), min_df=3, max_features=230_000),
        TextIndex(cache, 'params', params, ngram_range=(1, 2), min_df=2, max_features=100_000),
        TextIndex(cache, 'bm_title', title, kind='bm25', min_df=1, max_features=200_000),
        TextIndex(cache, 'bm_desc', descriptions.str.slice(0, 3000).tolist(), kind='bm25',
                  min_df=2, max_features=250_000),
        TextIndex(cache, 'bm_params', params, kind='bm25', min_df=2, max_features=150_000),
    ]
    for j, index in enumerate(indices):
        index.prepare(qfilters if j == 3 else qtexts)
    gc.collect()
    locs = items.item_location_id.to_numpy()
    micros = items.item_microcat_id.to_numpy()
    unique_m, micro_inverse = np.unique(micros, return_inverse=True)
    loc_indices = {int(k): np.asarray(v) for k, v in items.groupby('item_location_id').indices.items()}
    numeric = lambda name: pd.to_numeric(items[name], errors='coerce').fillna(0).to_numpy(np.float32)
    reviews, ratings, prices = (numeric(k) for k in ['item_rating_reviews_count', 'item_rating', 'item_price'])
    lat, lon = numeric('item_latitude'), numeric('item_longitude')
    centers = {l: (np.median(lat[ix]), np.median(lon[ix])) for l, ix in loc_indices.items()}
    pops = np.array([np.log1p(popularity.get(iid, 0)) for iid in ids], np.float32)
    phone = numeric('item_is_phone_hidden')
    msg = numeric('item_is_message_forbidden')
    sizes = np.array([len(t) for t in title], np.float32)
    n = len(items)
    features, labels, candidates, offsets = [], [], [], [0]
    feature_names = [
        'word', 'char', 'description', 'filter', 'bm_title', 'bm_description', 'bm_params',
        'coverage_title', 'coverage_description', 'coverage_params', 'geo', 'same_location',
        'location_probability', 'micro', 'click', 'baseline', 'bm_hybrid', 'log_reviews',
        'rating', 'log_price', 'item_popularity', 'log_distance', 'has_center', 'phone_hidden',
        'message_forbidden', 'title_length', 'query_length', 'history_count', 'exact_title_phrase',
        'baseline_relative', 'bm_relative', 'word_relative', 'char_relative', 'bm_title_relative',
        'micro_confidence', 'log_location_inventory']
    for qi, row in enumerate(queries.itertuples(index=False)):
        w, c, d, p = [ind.score(qi) for ind in indices[:4]]
        (bt, covt), (bd, covd), (bp, covp) = [ind.score(qi) for ind in indices[4:]]
        prior = Counter()
        qtext = qtexts[qi]
        if qtext in qm:
            total = max(3, sum(qm[qtext].values()))
            for m, count in qm[qtext].items():
                prior[m] += 2.0 * count / total
        sim = (qvectors[qi] @ qindex.T).tocoo()
        for j in np.argsort(sim.data)[-20:]:
            s = float(sim.data[j])
            if s < .36:
                continue
            counts = qm[qnames[int(sim.col[j])]]
            scale = s ** 4 / max(10, sum(counts.values()))
            for m, count in counts.items():
                prior[m] += scale * count
        mp = np.array([prior.get(int(m), 0) for m in unique_m], np.float32)
        micro_confidence = float(mp.max())
        mp /= max(micro_confidence, 1e-6)
        micro = mp[micro_inverse]
        geo, locprob = np.zeros(n, np.float32), np.zeros(n, np.float32)
        local = int(row.search_location_id)
        same = (locs == local).astype(np.float32)
        geo[:] = same
        counts = lc.get(local, Counter())
        if counts:
            maximum, total = max(counts.values()), sum(counts.values())
            for loc, count in counts.most_common(25):
                if loc in loc_indices:
                    ix = loc_indices[loc]
                    geo[ix] = np.maximum(geo[ix], .75 * count / maximum)
                    locprob[ix] = count / total
        click = np.zeros(n, np.float32)
        for iid, count in qc.get(qtext, {}).items():
            click[idmap[iid]] = .2 * min(1, np.log1p(count) / 4)
        lexical = 2*w + 1.5*c + .8*d + .45*p
        baseline = lexical + .38*geo + .27*micro + .035*np.log1p(reviews)/np.log(500) + click
        bm = 1.3*bt + .8*bd + .3*bp
        bh = bm + .8*geo + .4*micro
        # Union retains baseline top-50 and retrieves candidates from distinct
        # lexical views. The supervised model decides which 50 to retain.
        sets = [top(baseline, 250), top(lexical, 100), top(bh, 300), top(bm, 150),
                top(c + .25*geo, 100)]
        local_ix = loc_indices.get(local, np.empty(0, np.int32))
        if len(local_ix):
            sets.append(local_ix[top(bm[local_ix], 150)])
        chosen = np.unique(np.concatenate(sets)).astype(np.int32)
        k = len(chosen)
        if local in centers:
            clat, clon = centers[local]
            distance = np.log1p(111*np.hypot(lat[chosen]-clat,
                               (lon[chosen]-clon)*np.cos(np.deg2rad(clat))))
            has_center = 1.
        else:
            distance, has_center = np.full(k, 9.), 0.
        arrays = [w,c,d,p,bt,bd,bp,covt,covd,covp,geo,same,locprob,micro,click,baseline,bh]
        f = [x[chosen] for x in arrays] + [
            np.log1p(reviews[chosen]), ratings[chosen], np.log1p(np.maximum(prices[chosen],0)),
            pops[chosen], distance, np.full(k,has_center), phone[chosen],msg[chosen],sizes[chosen],
            np.full(k,len(tokenize(qtext))),np.full(k,np.log1p(sum(qm.get(qtext,{}).values()))),
            np.array([qtext in title[i] for i in chosen],np.float32),
            baseline[chosen]/max(float(baseline.max()),1e-6),
            bh[chosen]/max(float(bh.max()),1e-6),w[chosen]/max(float(w.max()),1e-6),
            c[chosen]/max(float(c.max()),1e-6),bt[chosen]/max(float(bt.max()),1e-6),
            np.full(k,micro_confidence),np.full(k,np.log1p(len(local_ix)))]
        features.append(np.nan_to_num(np.stack(f,axis=1).astype(np.float32)))
        relevant = set(row.labels)
        labels.append(np.array([ids[i] in relevant for i in chosen],np.int8))
        candidates.append(chosen)
        offsets.append(offsets[-1]+k)
        if (qi+1)%200 == 0:
            print('Features',qi+1,'/',len(queries),'candidates',offsets[-1],flush=True)
    # Free the indices before concatenating the feature blocks.
    del indices, items, descriptions, qindex, qvectors
    gc.collect()
    np.save(cache/'features.npy',np.concatenate(features))
    np.save(cache/'labels.npy',np.concatenate(labels))
    np.save(cache/'candidates.npy',np.concatenate(candidates))
    np.save(cache/'offsets.npy',np.array(offsets))
    joblib.dump({'queries':queries,'ids':ids,'features':feature_names},cache/'metadata.joblib')
    print('Feature cache complete',flush=True)


def train_model(args):
    import lightgbm as lgb
    cache, out = args.cache_dir, args.output_dir
    out.mkdir(parents=True,exist_ok=True)
    meta=joblib.load(cache/'metadata.joblib')
    x=np.load(cache/'features.npy',mmap_mode='r')
    y=np.load(cache/'labels.npy')
    candidates=np.load(cache/'candidates.npy')
    offsets=np.load(cache/'offsets.npy')
    queries,ids,names=meta['queries'],meta['ids'],meta['features']
    split=queries.split.to_numpy()
    sizes=np.diff(offsets)
    def rows(which):
        q=np.flatnonzero(split==which)
        return np.concatenate([np.arange(offsets[i],offsets[i+1]) for i in q]),q
    fit,fitq=rows('fit'); tune,tuneq=rows('tune'); test,testq=rows('test')
    # Raw phrases can differ only by case/spacing. Remove those last overlaps
    # as well before fitting; validation text must not occur in model training.
    held_texts=set(queries.loc[np.isin(split,['tune','test']),'search_query'].map(clean))
    fitq=np.array([i for i in fitq if clean(queries.iloc[i].search_query) not in held_texts])
    fit=np.concatenate([np.arange(offsets[i],offsets[i+1]) for i in fitq])
    def metric(scores,qix):
        recalls=[]; refs=[]; oracle=[]
        for i in qix:
            a,b=offsets[i:i+2]
            ix=top(scores[a:b],50)+a
            truth=set(queries.iloc[i].labels)
            pred=set(ids[candidates[ix]])
            recalls.append(len(pred&truth)/len(truth))
            refs.append(queries.iloc[i].reference_item in pred)
            oracle.append(float(y[a:b].sum())/len(truth))
        return {'recall':float(np.mean(recalls)), 'single_click':float(np.mean(refs)),
                'pool_recall':float(np.mean(oracle)), 'queries':len(qix)}
    baseline=np.asarray(x[:,names.index('baseline')])
    bm=np.asarray(x[:,names.index('bm_hybrid')])
    report={'baseline_tune':metric(baseline,tuneq),'baseline_test':metric(baseline,testq),
            'bm25_tune':metric(bm,tuneq),'bm25_test':metric(bm,testq)}
    print(json.dumps(report,indent=2),flush=True)
    # Every labelled query/item pair is removed from the history. Consequently
    # its direct click count is zero by construction. Do not train on that
    # artefact, or on per-item popularity that favors previously seen items.
    excluded={'click','item_popularity','baseline','baseline_relative'}
    used=[i for i,name in enumerate(names) if name not in excluded]
    model_names=[names[i] for i in used]
    model_x=np.asarray(x[:,used])
    model=lgb.LGBMRanker(objective='lambdarank',n_estimators=300,learning_rate=.04,
                         num_leaves=23,max_depth=-1,min_child_samples=80,
                         reg_lambda=10,subsample=1.,colsample_bytree=.9,
                         random_state=2026,n_jobs=4,deterministic=True,force_col_wise=True,
                         lambdarank_truncation_level=60,verbosity=-1)
    model.fit(model_x[fit],y[fit],group=sizes[fitq],feature_name=model_names,
              eval_set=[(model_x[tune],y[tune])],eval_group=[sizes[tuneq]],eval_at=[50],
              callbacks=[lgb.log_evaluation(50)])
    # Select the iteration using only the development queries; inspect test once.
    options=[]
    for trees in [50,100,150,200,250,300]:
        scores=np.zeros(len(y),np.float32)
        scores[tune]=model.predict(model_x[tune],num_iteration=trees)
        m=metric(scores,tuneq)
        options.append((m['recall'],trees))
        print('Tune trees',trees,m,flush=True)
    _,best_trees=max(options)
    scores=model.predict(model_x,num_iteration=best_trees).astype(np.float32)
    report['selected_trees']=best_trees
    report['model_tune']=metric(scores,tuneq)
    report['model_test']=metric(scores,testq)
    trained_items=set(iid for ix in fitq for iid in queries.iloc[ix].labels)
    coldq=np.array([i for i in testq if not (set(queries.iloc[i].labels)&trained_items)])
    report['cold_item_test']=metric(scores,coldq)
    report['fit_queries']=len(fitq)
    report['normalized_train_validation_overlap']=0
    report['features']=dict(sorted(zip(model_names,model.feature_importances_.tolist()),key=lambda x:-x[1]))
    report['tuning']=[{'trees':t,'recall':r} for r,t in options]
    print(json.dumps(report,indent=2),flush=True)
    predictions=[]
    for qi in np.flatnonzero(split=='benchmark'):
        a,b=offsets[qi:qi+2]
        selected=top(scores[a:b],50)+a
        selected=selected[np.argsort(-scores[selected],kind='stable')]
        predictions.append(' '.join(ids[candidates[selected]]))
    answer=pd.DataFrame({'query_id':queries.loc[split=='benchmark','query_id'].to_numpy(),
                         'answer':predictions})
    assert answer.query_id.is_unique and len(answer)==2452
    for p in predictions:
        v=p.split(); assert len(v)==len(set(v))==50
    answer.to_csv(out/'answer.csv',index=False)
    model.booster_.save_model(str(out/'model.txt'),num_iteration=best_trees)
    (out/'metrics.json').write_text(json.dumps(report,indent=2))
    np.save(cache/'model_scores.npy',scores)


def main():
    root=Path(__file__).resolve().parent.parent
    p=argparse.ArgumentParser()
    p.add_argument('--stage',choices=['build','train','all'],default='all')
    p.add_argument('--data-dir',type=Path,default=root/'data')
    p.add_argument('--cache-dir',type=Path,default=root/'.cache_v2')
    p.add_argument('--output-dir',type=Path,default=root)
    args=p.parse_args()
    if args.stage in ['build','all']: build(args)
    if args.stage in ['train','all']: train_model(args)


if __name__=='__main__':
    main()
