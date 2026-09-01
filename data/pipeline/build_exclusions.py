import hashlib
import os
import re
import numpy as np
import pandas as pd
DATA = os.environ.get('ECUP_DATA', 'data')
_ws = re.compile(r'\s+')

def fingerprint(name, attrs, category):
    s = f'{str(name).lower().strip()}|{str(attrs).lower().strip()}|{category}'
    return hashlib.md5(_ws.sub(' ', s).encode()).hexdigest()

def main():
    pairs = pd.read_parquet(f'{DATA}/artifacts/hardval_v1.parquet')
    items = pd.read_parquet(f'{DATA}/artifacts/hardval_v1_items.parquet')
    ids = set(np.unique(np.concatenate([pairs.id1.values, pairs.id2.values])).tolist())
    items = items[items['id'].isin(ids)].reset_index(drop=True)
    fps = [fingerprint(n, a, c) for n, a, c in zip(items['name'], items['attributes'], items['category'])]
    out = pd.DataFrame({'id': items['id'].values, 'fp': fps})
    out.to_parquet(f'{DATA}/artifacts/hardval_exclude.parquet', index=False)
    print(f'запретный список: {len(out):,} товаров, уникальных отпечатков {out.fp.nunique():,}')

if __name__ == '__main__':
    main()
