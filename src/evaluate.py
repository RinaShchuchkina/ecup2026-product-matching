from serialize import make_text
import argparse
import json
import os
import time
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, logging as hf_logging
DATA = os.environ.get('ECUP_DATA', 'data')
OUT = os.environ.get('ECUP_OUT', 'runs')
ap = argparse.ArgumentParser()
ap.add_argument('--models', nargs='+', required=True)
ap.add_argument('--pairs', default='valid_lb_calibrated.parquet')
ap.add_argument('--items', default='valid_items.parquet')
ap.add_argument('--maxlen', type=int, default=384)
ap.add_argument('--batch', type=int, default=128)
ap.add_argument('--tag', default='lbval')
ap.add_argument('--minpos', type=int, default=5)
ap.add_argument('--cap', type=int, default=500)
ap.add_argument('--tta', action='store_true')
ap.add_argument('--halves', action='store_true')
args = ap.parse_args()
hf_logging.set_verbosity_error()
pairs = pd.read_parquet(f'{DATA}/{args.pairs}')
items = pd.read_parquet(f'{DATA}/{args.items}').set_index('id')
lab = next((c for c in ('y', 'label', 'target') if c in pairs.columns))
cat = next((c for c in ('cat', 'category') if c in pairs.columns))
need = pd.unique(pd.concat([pairs.id1, pairs.id2]))
text_of = {p: make_text(r['name'], r['category'], r['attributes'], args.cap) for (p, r) in items.loc[need].iterrows()}
t1 = pairs.id1.map(text_of).fillna('').values
t2 = pairs.id2.map(text_of).fillna('').values
order = np.argsort([len(a) + len(b) for (a, b) in zip(t1, t2)])
y = pairs[lab].values
c = pairs[cat].values
ucats = np.unique(c)
print(f'пар {len(pairs):,}, карточек {len(text_of):,}, категорий {len(ucats)}', flush=True)

def macro(sc):
    v = [average_precision_score(y[c == cc], sc[c == cc]) for cc in ucats if y[c == cc].sum() >= args.minpos and y[c == cc].sum() < (c == cc).sum()]
    return float(np.mean(v))
for (col, fact) in (('v16a', 0.4357), ('v16b', 0.437), ('v16c', 0.4389), ('gb', 0.2924)):
    if col in pairs.columns:
        print(f'  контроль {col}: {macro(pairs[col].values):.4f} (факт ЛБ {fact:.4f})', flush=True)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
out_path = f'{OUT}/lbval_scores_{args.tag}.parquet'
for mdir in args.models:
    name = os.path.basename(mdir.rstrip('/'))
    tok = AutoTokenizer.from_pretrained(mdir)
    model = AutoModelForSequenceClassification.from_pretrained(mdir).to(dev).eval()
    if dev == 'cuda':
        model = model.half()
    t0 = time.time()

    acc = np.zeros(len(order))

    passes = [(t1, t2)] + ([(t2, t1)] if args.tta else [])

    for (pa, pb) in passes:

        sc = np.zeros(len(order))
        with torch.inference_mode():
            for s in range(0, len(order), args.batch):
                idx = order[s:s + args.batch]
                enc = tok([pa[i] for i in idx], [pb[i] for i in idx], padding=True, truncation=True, max_length=args.maxlen, return_tensors='pt')
                enc = {k: v.to(dev) for (k, v) in enc.items()}
                sc[idx] = torch.sigmoid(model(**enc).logits.squeeze(-1).float()).cpu().numpy()
    

        acc += sc

    sc = acc / len(passes)

    pairs[name] = sc
    print(f'РЕЗУЛЬТАТ {name}: macro AP = {macro(sc):.4f}  ({len(order) / (time.time() - t0):.0f} п/с)', flush=True)
    pairs.to_parquet(out_path, index=False)
    del model
    torch.cuda.empty_cache()
print(f'сохранено {out_path}', flush=True)
