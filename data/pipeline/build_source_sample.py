import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
DATA = os.environ.get('ECUP_DATA', 'data')
SEED = 42
SIZE = 2500000

def main():
    ml = pd.read_parquet(f'{DATA}/matches_llm.parquet')
    take = np.sort(np.random.default_rng(SEED).choice(len(ml), size=SIZE, replace=False))
    pairs = ml.iloc[take].reset_index(drop=True)
    pairs.to_parquet(f'{DATA}/llm_pairs_2p5m.parquet', index=False)
    need = np.unique(np.concatenate([pairs.id1.values, pairs.id2.values]))
    chunks = []
    pf = pq.ParquetFile(f'{DATA}/items.parquet')
    for batch in pf.iter_batches(batch_size=131072, columns=['id', 'name', 'attributes', 'category']):
        b = batch.to_pandas()
        m = np.isin(b['id'].values, need)
        if m.any():
            chunks.append(b.loc[m])
    items = pd.concat(chunks, ignore_index=True)
    items.to_parquet(f'{DATA}/items_llm_2p5m.parquet', index=False)
    print(f'пар {len(pairs):,}; карточек {len(items):,}')

if __name__ == '__main__':
    main()
