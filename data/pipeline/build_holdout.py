import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
DATA = os.environ.get('ECUP_DATA', 'data')
OUT = f'{DATA}/artifacts'
N_TARGET = 150000
ml = pd.read_parquet(f'{DATA}/matches_llm.parquet')
rng = np.random.default_rng(42)
pre_idx = set(rng.choice(len(ml), size=2500000, replace=False))
mask_free = ~ml.index.isin(pre_idx)
free = ml[mask_free]
t = free['target'].values
conf = (t <= 1 / 9 + 1e-09) | (t >= 7 / 9 - 1e-09)
free = free[conf]
free['label'] = (free['target'] >= 7 / 9 - 1e-09).astype(int)
sample = free.sample(min(N_TARGET * 3, len(free)), random_state=7)
need = np.unique(np.concatenate([sample.id1.values, sample.id2.values]))
pf = pq.ParquetFile(f'{DATA}/items.parquet')
chunks = []
for batch in pf.iter_batches(batch_size=131072, columns=['id', 'name', 'attributes', 'category']):
    b = batch.to_pandas()
    m = np.isin(b['id'].values, need)
    if m.any():
        chunks.append(b.loc[m])
items = pd.concat(chunks, ignore_index=True)
have = set(items.id)
sample = sample[sample.id1.isin(have) & sample.id2.isin(have)].reset_index(drop=True)
cat_of = items.set_index('id')['category']
sample['category'] = cat_of.loc[sample.id1].values
r = np.random.default_rng(7)
keep = np.ones(len(sample), bool)
for c in sample['category'].unique():
    m = (sample['category'] == c).values
    pos_idx = np.where(m & (sample['label'] == 1))[0]
    neg = int((m & (sample['label'] == 0)).sum())
    want = int(0.1 / 0.9 * neg)
    if want < len(pos_idx):
        keep[r.choice(pos_idx, len(pos_idx) - want, replace=False)] = False
hv = sample[keep].reset_index(drop=True)
if len(hv) > N_TARGET:
    hv = hv.groupby('category', group_keys=False).apply(lambda g: g.sample(int(len(g) * N_TARGET / len(hv)), random_state=7)).reset_index(drop=True)
hv[['id1', 'id2', 'label', 'category']].to_parquet(f'{OUT}/hardval_v1.parquet', index=False)
need2 = np.unique(np.concatenate([hv.id1.values, hv.id2.values]))
items[items.id.isin(need2)].to_parquet(f'{OUT}/hardval_v1_items.parquet', index=False)
print(f'saved: {len(hv):,} pairs')
