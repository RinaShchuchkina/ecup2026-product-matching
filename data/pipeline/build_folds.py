import hashlib
import os
import re
import numpy as np
import pandas as pd
DATA = os.environ.get('ECUP_DATA', 'data')
N_FOLDS = 5
SEED = 0
_ws = re.compile(r'\s+')

def fingerprint(name, attrs, category):
    s = f'{str(name).lower().strip()}|{str(attrs).lower().strip()}|{category}'
    return hashlib.md5(_ws.sub(' ', s).encode()).hexdigest()

def main():
    items = pd.read_parquet(f'{DATA}/items_human.parquet', columns=['id', 'name', 'attributes', 'category'])
    matches = pd.read_parquet(f'{DATA}/matches.parquet')
    n = len(items)
    parent = np.arange(n)

    def find(x):
        r = x
        while parent[r] != r:
            r = parent[r]
        while parent[x] != r:
            parent[x], x = r, parent[x]
        return r

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    seen = {}
    aliases = 0
    for i, (name, attrs, category) in enumerate(zip(items['name'], items['attributes'], items['category'])):
        h = fingerprint(name, attrs, category)
        j = seen.get(h)
        if j is None:
            seen[h] = i
        else:
            union(j, i)
            aliases += 1
    pos = pd.Series(np.arange(n), index=items['id'].values)
    a = pos[matches.id1.values].values
    b = pos[matches.id2.values].values
    for x, y in zip(a, b):
        union(x, y)
    roots = np.array([find(i) for i in range(n)])
    comp = roots[a]
    out = matches.copy()
    out['category'] = items['category'].values[a]
    out['comp'] = comp
    uniq = np.unique(comp)
    fold_of = pd.Series(np.random.default_rng(SEED).integers(0, N_FOLDS, size=len(uniq)), index=uniq)
    out['fold'] = fold_of[comp].values
    out.to_parquet(f'{DATA}/artifacts/folds_v1.parquet', index=False)
    sizes = out.fold.value_counts().sort_index()
    print(f'алиасов по содержимому: {aliases:,}; компонент: {len(uniq):,}')
    print('пар по фолдам: ' + ', '.join(f'{k}={v:,}' for k, v in sizes.items()))

if __name__ == '__main__':
    main()
