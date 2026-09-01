import argparse
import os
import numpy as np
import pandas as pd
from collections import defaultdict
DATA = os.environ.get('ECUP_DATA', 'data')
POS_THR = 8 / 9 - 1e-09
NEG_THR = 1 / 9 + 1e-09
MAX_CLUSTER = 8
MAX_NEW_POS = 1500000
MAX_NEW_NEG = 1500000
ml = pd.read_parquet(f'{DATA}/matches_llm.parquet')
hv = pd.read_parquet(f'{DATA}/artifacts/hardval_v1.parquet')
hv_ids = set(pd.unique(np.concatenate([hv.id1.values, hv.id2.values])))
import hashlib
import re
_ws = re.compile('\\s+')

def _fp(name, attrs, cat):
    s_ = f'{str(name).lower().strip()}|{str(attrs).lower().strip()}|{cat}'
    return hashlib.md5(_ws.sub(' ', s_).encode()).hexdigest()
hv_items_df = pd.read_parquet(f'{DATA}/artifacts/hardval_v1_items.parquet')
hv_fps = {_fp(n, a, c) for (n, a, c) in zip(hv_items_df['name'], hv_items_df['attributes'], hv_items_df['category'])}
del hv_items_df
import pyarrow.parquet as pq
ap = argparse.ArgumentParser()
ap.add_argument('--content_alias', action='store_true')
args = ap.parse_args()
banned_alias = set()
pf_ = pq.ParquetFile(f'{DATA}/items.parquet')
scanned = 0
if args.content_alias:
    for batch in pf_.iter_batches(batch_size=131072, columns=['id', 'name', 'attributes', 'category']):
        b = batch.to_pandas()
        fps = [_fp(n, a, c) for (n, a, c) in zip(b['name'], b['attributes'], b['category'])]
        m_ = [i for (i, f) in enumerate(fps) if f in hv_fps]
        if m_:
            banned_alias.update(b['id'].values[m_])
    scanned += len(b)
hv_items = hv_ids | banned_alias
print(f'пул {len(ml):,}; hardval по ID {len(hv_ids):,}; ДОБАВЛЕНО content-alias товаров {len(banned_alias - hv_ids):,}; всего под запретом {len(hv_items):,}', flush=True)
pos = ml[ml.target >= POS_THR]
neg = ml[ml.target <= NEG_THR]
ids = pd.unique(np.concatenate([pos.id1.values, pos.id2.values]))
idx = {v: i for (i, v) in enumerate(ids)}
par = np.arange(len(ids))

def find(x):
    r = x
    while par[r] != r:
        r = par[r]
    while par[x] != r:
        (par[x], x) = (r, par[x])
    return r
for (a, b) in zip(pos.id1.map(idx).values, pos.id2.map(idx).values):
    (ra, rb) = (find(a), find(b))
    if ra != rb:
        par[rb] = ra
comp = np.array([find(i) for i in range(len(ids))])
members = defaultdict(list)
for (i, cid) in enumerate(comp):
    members[cid].append(ids[i])
known = set()
for (a, b) in zip(pos.id1.values, pos.id2.values):
    known.add((a, b) if a < b else (b, a))
rng = np.random.default_rng(0)
new_pos = []
kept_clusters = {}
for (cid, mem) in members.items():
    if not 2 <= len(mem) <= MAX_CLUSTER:
        continue
    if any((m in hv_items for m in mem)):
        continue
    kept_clusters[cid] = mem
    for i in range(len(mem)):
        for j in range(i + 1, len(mem)):
            (a, b) = (mem[i], mem[j]) if mem[i] < mem[j] else (mem[j], mem[i])
            if (a, b) not in known:
                new_pos.append((a, b, 1.0))
print(f'кластеров 2..{MAX_CLUSTER} без hardval: {len(kept_clusters):,}; новых позитивов {len(new_pos):,}', flush=True)
if len(new_pos) > MAX_NEW_POS:
    sel = rng.choice(len(new_pos), MAX_NEW_POS, replace=False)
    new_pos = [new_pos[i] for i in sel]
comp_of = {}
for (cid, mem) in kept_clusters.items():
    for m in mem:
        comp_of[m] = cid
known_neg = set()
for (a, b) in zip(neg.id1.values, neg.id2.values):
    known_neg.add((a, b) if a < b else (b, a))
new_neg = []
for (a, b) in zip(neg.id1.values, neg.id2.values):
    (ca, cb) = (comp_of.get(a), comp_of.get(b))
    if ca is None and cb is None:
        continue
    if ca is not None and cb is not None and (ca == cb):
        continue
    ma = kept_clusters.get(ca, [a]) if ca is not None else [a]
    mb = kept_clusters.get(cb, [b]) if cb is not None else [b]
    for x in ma:
        for yy in mb:
            if x in hv_items or yy in hv_items:
                continue
            p = (x, yy) if x < yy else (yy, x)
            if p not in known_neg and p not in known:
                new_neg.append((p[0], p[1], 0.0))
    if len(new_neg) > MAX_NEW_NEG * 2:
        break
print(f'новых негативов (до сэмплинга): {len(new_neg):,}', flush=True)
if len(new_neg) > MAX_NEW_NEG:
    sel = rng.choice(len(new_neg), MAX_NEW_NEG, replace=False)
    new_neg = [new_neg[i] for i in sel]
out = pd.DataFrame(new_pos + new_neg, columns=['id1', 'id2', 'target'])
out = out.drop_duplicates(subset=['id1', 'id2']).reset_index(drop=True)
out.to_parquet(f'{DATA}/artifacts/transitive_pairs.parquet', index=False)
print(f'ИТОГО транзитивных пар: {len(out):,} (поз {int(out.target.sum()):,} / нег {int((out.target == 0).sum()):,})')
print(f'сохранено: analysis/artifacts/transitive_pairs.parquet')
