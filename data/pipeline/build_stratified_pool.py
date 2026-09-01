import argparse
import json
import re
import time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
TOKEN = re.compile('[a-zа-яё0-9]+')
J_EDGES = [-0.01, 0.2, 0.5, 1.01]
J_NAMES = ['far', 'mid', 'near']
BANDS = ['pos', 'mid', 'neg']
CELL_PRIOR = {('pos', 'far'): 3.0, ('pos', 'mid'): 2.0, ('pos', 'near'): 1.0, ('mid', 'far'): 0.5, ('mid', 'mid'): 0.7, ('mid', 'near'): 0.8, ('neg', 'far'): 0.5, ('neg', 'mid'): 1.2, ('neg', 'near'): 2.5}

def name_jaccard(pairs_id1, pairs_id2, items_path, verbose=True):
    need = np.unique(np.concatenate([pairs_id1, pairs_id2]))
    buf = [None] * len(need)
    t0 = time.time()
    for batch in pq.ParquetFile(items_path).iter_batches(batch_size=262144, columns=['id', 'name']):
        d = batch.to_pydict()
        ids = np.asarray(d['id'], dtype=np.int64)
        p = np.clip(np.searchsorted(need, ids), 0, len(need) - 1)
        hit = need[p] == ids
        for k in np.where(hit)[0]:
            ts = set(TOKEN.findall(str(d['name'][k]).lower()))
            buf[p[k]] = np.fromiter((hash(t) & 2147483647 for t in ts), dtype=np.int32, count=len(ts))
    lens = np.array([0 if b is None else len(b) for b in buf], dtype=np.int32)
    off = np.zeros(len(need) + 1, dtype=np.int64)
    off[1:] = np.cumsum(lens)
    flat = np.concatenate([b for b in buf if b is not None and len(b)])
    del buf
    if verbose:
        print(f'  name index built in {time.time() - t0:.0f}s', flush=True)
    p1 = np.searchsorted(need, pairs_id1)
    p2 = np.searchsorted(need, pairs_id2)
    out = np.zeros(len(pairs_id1), dtype=np.float32)
    for k in range(len(out)):
        a = flat[off[p1[k]]:off[p1[k] + 1]]
        b = flat[off[p2[k]]:off[p2[k] + 1]]
        if len(a) == 0 or len(b) == 0:
            out[k] = -1.0
            continue
        inter = len(np.intersect1d(a, b, assume_unique=True))
        out[k] = inter / (len(a) + len(b) - inter)
    return out

def item_category(ids, items_path):
    need = np.unique(ids)
    code = np.full(len(need), -1, dtype=np.int16)
    (names, lut) = ([], {})
    for batch in pq.ParquetFile(items_path).iter_batches(batch_size=262144, columns=['id', 'category']):
        d = batch.to_pydict()
        iid = np.asarray(d['id'], dtype=np.int64)
        p = np.clip(np.searchsorted(need, iid), 0, len(need) - 1)
        hit = need[p] == iid
        for k in np.where(hit)[0]:
            c = d['category'][k]
            if c not in lut:
                lut[c] = len(names)
                names.append(c)
            code[p[k]] = lut[c]
    return (pd.Series(code, index=need), names)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pairs', default='matches_llm.parquet')
    ap.add_argument('--items', default='items.parquet')
    ap.add_argument('--size', type=int, default=2000000)
    ap.add_argument('--pos-share', type=float, default=0.3, help='target share of teacher-positive rows per category')
    ap.add_argument('--mid-share', type=float, default=0.12, help='target share of ambiguous 3/9..7/9 rows per category')
    ap.add_argument('--exclude-pairs', default='', help='parquet id1,id2 to drop (hardval)')
    ap.add_argument('--exclude-items', default='', help='parquet id[,fp] whose items must not appear')
    ap.add_argument('--out', required=True)
    ap.add_argument('--stats-out', default='')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    pairs = pd.read_parquet(args.pairs)
    t9 = np.rint(pairs.target.values * 9).astype(np.int8)
    i1 = pairs.id1.values.astype(np.int64)
    i2 = pairs.id2.values.astype(np.int64)
    print(f'pool {len(pairs):,}', flush=True)
    keep = np.ones(len(pairs), dtype=bool)
    if args.exclude_pairs:
        ex = pd.read_parquet(args.exclude_pairs)
        bad = set(map(tuple, np.sort(ex[['id1', 'id2']].values, axis=1).tolist()))
        srt = np.sort(np.stack([i1, i2], 1), axis=1)
        keep &= ~np.fromiter(((a, b) in bad for (a, b) in srt.tolist()), dtype=bool, count=len(srt))
        print(f'  after exclude-pairs: {keep.sum():,}', flush=True)
    if args.exclude_items:
        ex = pd.read_parquet(args.exclude_items)
        bad_id = set(ex['id'].values.tolist())
        keep &= ~(pd.Series(i1).isin(bad_id) | pd.Series(i2).isin(bad_id)).values
        print(f'  after exclude-items: {keep.sum():,}', flush=True)
    idx = np.where(keep)[0]
    (i1, i2, t9) = (i1[idx], i2[idx], t9[idx])
    tgt = pairs.target.values[idx].astype(np.float32)
    del pairs
    print('computing name jaccard ...', flush=True)
    jac = name_jaccard(i1, i2, args.items)
    (cat_of, cat_names) = item_category(np.concatenate([i1, i2]), args.items)
    cat = cat_of.reindex(i1).values
    band = np.where(t9 >= 8, 0, np.where(t9 <= 2, 2, 1))
    jb = np.clip(np.digitize(jac, J_EDGES[1:-1]), 0, 2)
    jb[jac < 0] = 1
    df = pd.DataFrame({'i': np.arange(len(i1)), 'cat': cat, 'band': band, 'jb': jb, 't9': t9})
    n_cat = df.cat.nunique()
    per_cat = args.size // n_cat
    share = {0: args.pos_share, 1: args.mid_share, 2: 1.0 - args.pos_share - args.mid_share}
    (chosen, stats) = ([], [])
    for (c, gc) in df.groupby('cat'):
        for (b, gb) in gc.groupby('band'):
            budget = int(per_cat * share[b])
            prior = np.array([CELL_PRIOR[BANDS[b], J_NAMES[j]] for j in range(3)])
            avail = np.array([int((gb.jb == j).sum()) for j in range(3)])
            w = prior * (avail > 0)
            if w.sum() == 0:
                continue
            quota = np.floor(budget * w / w.sum()).astype(int)
            for _ in range(3):
                short = int((quota - np.minimum(quota, avail)).sum())
                quota = np.minimum(quota, avail)
                if short <= 0:
                    break
                room = avail - quota
                if room.sum() == 0:
                    break
                quota = quota + np.minimum(room, np.floor(short * room / room.sum()).astype(int))
            for j in range(3):
                gj = gb[gb.jb == j]
                if len(gj) == 0 or quota[j] == 0:
                    continue
                take = gj.i.values
                if quota[j] < len(take):
                    take = take[rng.choice(len(take), quota[j], replace=False)]
                chosen.append(take)
                stats.append({'category': cat_names[c], 'band': BANDS[b], 'jband': J_NAMES[j], 'available': int(len(gj)), 'selected': int(len(take))})
    sel = np.concatenate(chosen)
    rng.shuffle(sel)
    conf = np.abs(2 * tgt[sel] - 1)
    p = np.clip(tgt[sel], 0.001, 1 - 0.001)
    var = p * (1 - p) / 9.0
    w = 1.0 / (var + 1.0 / 36.0)
    w = (w / np.median(w)).astype(np.float32)
    w = np.clip(w, 0.4, 2.0)
    out = pd.DataFrame({'id1': i1[sel], 'id2': i2[sel], 'target': tgt[sel], 'sample_weight': w, 'name_jaccard': jac[sel], 'teacher_votes': t9[sel]})
    out.to_parquet(args.out, index=False)
    st = pd.DataFrame(stats)
    print(f'\nselected {len(out):,} rows; positive share {(out.teacher_votes >= 8).mean():.3f}; mean |2p-1| {conf.mean():.3f}', flush=True)
    print(st.groupby(['band', 'jband'])[['available', 'selected']].sum().to_string())
    print('\nper-category positives selected:')
    print(st[st.band == 'pos'].groupby('category')['selected'].sum().sort_values().to_string())
    if args.stats_out:
        with open(args.stats_out, 'w') as f:
            json.dump({'rows': int(len(out)), 'cells': stats, 'config': vars(args)}, f, ensure_ascii=False, indent=1)
if __name__ == '__main__':
    main()
