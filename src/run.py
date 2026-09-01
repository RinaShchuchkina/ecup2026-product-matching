import argparse
import json
from serialize import make_text
import os
import time
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'true')
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
CE = [('model2', 1.0, 384)]
CONFIG = os.environ.get('SOLUTION_CONFIG', os.path.join(os.path.dirname(HERE), 'configs', 'solution_1.json'))
WEIGHTS_ROOT = os.environ.get('WEIGHTS_ROOT', os.path.join(os.path.dirname(HERE), 'weights'))
WG = 0.0
MAXLEN = 384


def rank01(x):
    order = np.argsort(x, kind='stable')
    r = np.empty(len(x), dtype=np.float64)
    r[order] = np.arange(1, len(x) + 1)
    return r / len(x)

def score_ce(model_dir, t1, t2, batch, device, dtype, t0, tta, maxlen=MAXLEN, deadline=None):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, dtype=dtype)
    model.to(device).eval()
    order = np.argsort([len(a) + len(b) for (a, b) in zip(t1, t2)])
    out = np.zeros(len(order))
    passes = [(t1, t2)] + ([(t2, t1)] if tta else [])
    done = 0
    dur = 0.0
    with torch.inference_mode():
        for (pa, pb) in passes:
            if done > 0 and deadline is not None and (time.time() - t0 + dur * 1.05 > deadline):
                print(f'[{time.time() - t0:.0f}s] tta guard: swap pass needs {dur:.0f}s, skipping', flush=True)
                break
            _p = time.time()
            for s in range(0, len(order), batch):
                idx = order[s:s + batch]
                enc = tok([pa[i] for i in idx], [pb[i] for i in idx], padding=True, truncation=True, max_length=maxlen, return_tensors='pt')
                enc = {k: v.to(device, non_blocking=True) for (k, v) in enc.items()}
                logits = model(**enc).logits.squeeze(-1).float()
                out[idx] += torch.sigmoid(logits).cpu().numpy()
            dur = time.time() - _p
            done += 1
    out /= max(done, 1)
    del model
    if device == 'cuda':
        torch.cuda.empty_cache()
    print(f'[{time.time() - t0:.0f}s] {os.path.basename(model_dir)} done (tta={tta})', flush=True)
    return out

def main():
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument('--items_path', '--items-path', dest='items_path', required=True)
    ap.add_argument('--matches_path', '--matches-path', dest='matches_path', required=True)
    ap.add_argument('--output_path', '--output-path', dest='output_path', required=True)
    ap.add_argument('--config', default=CONFIG)
    args = ap.parse_args()
    ce = list(CE)
    cap = 500
    maxlen = MAXLEN
    use_tta = True
    if args.config:
        cfg = json.load(open(args.config))
        ce = [(cfg['checkpoint'], 1.0, int(cfg.get('batch', 384)))]
        cap = int(cfg.get('max_attr_chars', 500))
        maxlen = int(cfg.get('max_length', MAXLEN))
        use_tta = bool(cfg.get('swap_tta', True))
    items = pd.read_parquet(args.items_path)
    matches = pd.read_parquet(args.matches_path)
    n_pairs = len(matches)
    print(f'[{time.time() - t0:.0f}s] {len(items):,} items, {n_pairs:,} pairs', flush=True)
    need = set(matches['id1'].tolist()) | set(matches['id2'].tolist())
    items = items[items['id'].isin(need)].drop_duplicates(subset=['id']).reset_index(drop=True)
    print(f'[{time.time() - t0:.0f}s] карточек после фильтра {len(items):,}', flush=True)
    gb = None
    import torch
    if torch.cuda.is_available():
        (device, dtype) = ('cuda', torch.bfloat16)
    else:
        (device, dtype) = ('cpu', torch.float32)
    print(f'[{time.time() - t0:.0f}s] нужных карточек {len(items):,}', flush=True)
    text_of = {}
    for (pid, n, c, a) in zip(items['id'], items['name'], items['category'], items['attributes']):
        text_of[pid] = make_text(n, c, a, cap)
    t1 = matches['id1'].map(text_of).fillna('').values
    t2 = matches['id2'].map(text_of).fillna('').values
    limit = 360.0 if n_pairs < 180000 else 780.0
    budget = 0.45 * limit
    pred = WG * rank01(gb) if gb is not None else np.zeros(n_pairs, dtype=float)
    used_w = WG if gb is not None else 0.0
    scored = 0
    for (i, (sub, w, batch)) in enumerate(ce):
        model_dir = sub if os.path.isdir(sub) else os.path.join(WEIGHTS_ROOT, sub)
        if not os.path.isdir(model_dir):
            model_dir = os.path.join(HERE, sub)
        if not os.path.isdir(model_dir):
            raise SystemExit(f'checkpoint not found: {sub}')
        if not os.path.isdir(model_dir) or w <= 0:
            continue
        elapsed = time.time() - t0
        if scored > 0 and elapsed > budget:
            print(f'[{elapsed:.0f}s] budget guard: skip {sub}', flush=True)
            break
        tta = use_tta
        _tm = time.time()
        try:
            s = score_ce(model_dir, t1, t2, batch, device, dtype, t0, tta, maxlen, 0.92 * limit)
            print(f'[{time.time() - t0:.0f}s] {sub} занял {time.time() - _tm:.0f}s', flush=True)
        except Exception as exc:
            print(f'[{time.time() - t0:.0f}s] {sub} failed: {exc}', flush=True)
            continue
        pred = pred + w * rank01(s)
        used_w += w
        scored += 1
    if scored == 0 and used_w <= 0:
        print('fallback: lexical', flush=True)
        lex = np.array([len(set(a.split()) & set(b.split())) / max(len(set(a.split()) | set(b.split())), 1) for (a, b) in zip(t1, t2)])
        (pred, used_w) = (rank01(lex), 1.0)
    if used_w <= 0:
        used_w = 1.0
    pred = np.nan_to_num(pred / used_w, nan=0.5, posinf=1.0, neginf=0.0)
    out = pd.DataFrame({'id1': matches['id1'], 'id2': matches['id2'], 'predict': pred})
    out.to_csv(args.output_path, index=False)
    print(f'[{time.time() - t0:.0f}s] saved {args.output_path} ({len(out):,} rows)', flush=True)
if __name__ == '__main__':
    main()
