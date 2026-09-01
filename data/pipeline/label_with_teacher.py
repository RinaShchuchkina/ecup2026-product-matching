import argparse
import json
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from transformers import AutoModelForCausalLM, AutoTokenizer, logging as hf_logging
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from teacher_prompts import build_messages
DATA = os.environ.get('ECUP_DATA', 'data')
OUT = os.environ.get('ECUP_OUT', 'runs')
MAXLEN_BY_VARIANT = {'short': 1536, 'medium': 2432, 'fewshot': 2944}
ap = argparse.ArgumentParser()
ap.add_argument('--model', default='Qwen/Qwen3-30B-A3B')
ap.add_argument('--pairs', default='mine_candidates.parquet')
ap.add_argument('--items', default='mine_candidates_items.parquet')
ap.add_argument('--limit', type=int, default=0)
ap.add_argument('--batch', type=int, default=8)
ap.add_argument('--maxlen', type=int, default=0)
ap.add_argument('--variant', default='medium', choices=['short', 'medium', 'fewshot'])
ap.add_argument('--symmetric', action='store_true')
ap.add_argument('--tag', default='q30b')
ap.add_argument('--eval_against', default='')
args = ap.parse_args()
hf_logging.set_verbosity_error()
maxlen = args.maxlen or MAXLEN_BY_VARIANT[args.variant]
pairs = pd.read_parquet(f'{DATA}/{args.pairs}')
if args.limit:
    pairs = pairs.sample(min(args.limit, len(pairs)), random_state=0).reset_index(drop=True)
items_frame = pd.read_parquet(f'{DATA}/{args.items}')
if items_frame['id'].duplicated().any():
    raise RuntimeError('item table contains duplicate IDs')
items = items_frame.set_index('id')
if pairs[['id1', 'id2']].isna().any().any() or (pairs['id1'] == pairs['id2']).any():
    raise RuntimeError('pair table contains null IDs or self-pairs')
lo = np.minimum(pairs['id1'].to_numpy(dtype=np.int64), pairs['id2'].to_numpy(dtype=np.int64))
hi = np.maximum(pairs['id1'].to_numpy(dtype=np.int64), pairs['id2'].to_numpy(dtype=np.int64))
if pd.MultiIndex.from_arrays([lo, hi]).duplicated().any():
    raise RuntimeError('pair table contains duplicate unordered pairs')
cat1 = pairs['id1'].map(items['category'])
cat2 = pairs['id2'].map(items['category'])
if cat1.isna().any() or cat2.isna().any() or (not np.array_equal(cat1.astype(str).to_numpy(), cat2.astype(str).to_numpy())):
    raise RuntimeError('teacher diagnostic contains missing/cross-category items')
if 'category' not in pairs or not np.array_equal(pairs['category'].astype(str).to_numpy(), cat1.astype(str).to_numpy()):
    raise RuntimeError('declared pair category disagrees with item cards')
print(f'пар {len(pairs):,}; карточек {len(items):,}; variant={args.variant} maxlen={maxlen} sym={args.symmetric}', flush=True)
tok = AutoTokenizer.from_pretrained(args.model)
tok.padding_side = 'left'
tok.truncation_side = 'left'
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
yes_ids = tok.encode('да', add_special_tokens=False)
no_ids = tok.encode('нет', add_special_tokens=False)
if len(yes_ids) != 1 or len(no_ids) != 1 or yes_ids == no_ids:
    raise RuntimeError(f'exact lowercase verbalizers must each be one distinct token: да={yes_ids}, нет={no_ids}')
print(f'yes_ids={yes_ids} no_ids={no_ids}', flush=True)
dt = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dt, device_map='cuda').eval()
print(f'модель загружена, dtype={dt}', flush=True)

def render(a_row, b_row, cat):
    msgs = build_messages(a_row, b_row, cat, args.variant)
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError as exc:
        raise RuntimeError('tokenizer cannot prove enable_thinking=False') from exc
if len(pairs):
    probe = render(items.loc[pairs.iloc[0].id1], items.loc[pairs.iloc[0].id2], pairs.iloc[0].category)
    probe_ids = tok.encode(probe, add_special_tokens=False)
    for (word, token_id) in (('да', yes_ids[0]), ('нет', no_ids[0])):
        if tok.encode(probe + word, add_special_tokens=False) != probe_ids + [token_id]:
            raise RuntimeError(f'{word!r} is not a one-token suffix at the chat boundary')

def score_direction(rows_a, rows_b, cats, label):
    texts = [render(items.loc[a], items.loc[b], c) for (a, b, c) in zip(rows_a, rows_b, cats)]
    lens = np.array([len(tok(t)['input_ids']) for t in texts])
    over = int((lens > maxlen).sum())
    print(f'[{label}] длина p50={np.percentile(lens, 50):.0f} p99={np.percentile(lens, 99):.0f} max={lens.max()} превышают maxlen: {over} ({over / len(lens) * 100:.2f}%)', flush=True)
    if over:
        raise RuntimeError(f'[{label}] {over} prompts exceed the audited maxlen={maxlen}; refusing tokenizer truncation')
    order = np.argsort(lens)
    out = np.zeros(len(texts))
    t0 = time.time()
    with torch.inference_mode():
        for s in range(0, len(order), args.batch):
            idx = order[s:s + args.batch]
            enc = tok([texts[i] for i in idx], return_tensors='pt', padding=True, truncation=False).to('cuda')
            logits = model(**enc).logits[:, -1, :].float()
            lp = torch.log_softmax(logits, dim=-1)
            y = torch.logsumexp(lp[:, yes_ids], dim=-1)
            n = torch.logsumexp(lp[:, no_ids], dim=-1)
            out[idx] = (y - n).cpu().numpy()
            if s // args.batch % 40 == 0:
                el = max(time.time() - t0, 1e-09)
                rate = max(s, 1) / el
                print(f'  [{label}] {s}/{len(order)} {rate:.1f} pairs/s eta {(len(order) - s) / rate / 60:.0f} мин', flush=True)
    return out
t_start = time.time()
sc_ab = score_direction(pairs.id1.values, pairs.id2.values, pairs['category'].values, 'A→B')
if args.symmetric:
    sc_ba = score_direction(pairs.id2.values, pairs.id1.values, pairs['category'].values, 'B→A')
    scores = (sc_ab + sc_ba) / 2.0
    pairs['teacher_ab'] = sc_ab
    pairs['teacher_ba'] = sc_ba
else:
    scores = sc_ab
pairs['teacher_score'] = scores
pairs['teacher_prob'] = 1.0 / (1.0 + np.exp(-scores))
os.makedirs(OUT, exist_ok=True)
out_path = f"{OUT}/labels_{args.tag}_{args.variant}{('_sym' if args.symmetric else '')}.parquet"
pairs.to_parquet(out_path, index=False)
print(f'сохранено {out_path} за {(time.time() - t_start) / 60:.1f} мин', flush=True)
if args.eval_against:
    ref = pd.read_parquet(f'{DATA}/{args.eval_against}')
    col = 'target' if 'target' in ref.columns else 'label'
    mg = pairs.merge(ref[['id1', 'id2', col]], on=['id1', 'id2'])
    if len(mg) > 100:
        (y, c) = (mg[col].values, mg['category'].values)
        per = {}
        for cc in np.unique(c):
            m = c == cc
            if 0 < y[m].sum() < m.sum():
                per[cc] = average_precision_score(y[m], mg['teacher_score'].values[m])
        acc = ((mg['teacher_score'] > 0).astype(int) == y).mean()
        print(f'РЕЗУЛЬТАТ variant={args.variant}: macro AP={np.mean(list(per.values())):.4f} accuracy={acc:.4f} на {len(mg):,} парах', flush=True)
        for cc in ('Обувь', 'Одежда', 'Ювелирные изделия', 'Галантерея и аксессуары'):
            if cc in per:
                print(f'    {cc}: {per[cc]:.4f}', flush=True)
