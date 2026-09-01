import argparse
import hashlib
import json
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, logging as hf_logging
import consensus_contract
from serialize import make_text
DATA = os.environ.get('ECUP_DATA', 'data')
OUT = os.environ.get('ECUP_OUT', 'runs')

class SoftPairDS(Dataset):

    def __init__(self, t1, t2, y, w, cat=None, swap_p=0.5):
        (self.t1, self.t2, self.y, self.w, self.swap_p) = (t1, t2, y, w, swap_p)
        self.cat = cat if cat is not None else np.zeros(len(y), dtype=np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        if np.random.random() < self.swap_p:
            return (self.t2[i], self.t1[i], self.y[i], self.w[i], self.cat[i])
        return (self.t1[i], self.t2[i], self.y[i], self.w[i], self.cat[i])

class CatGroupedSampler(torch.utils.data.Sampler):

    def __init__(self, cat, batch_size, seed=0):
        (self.cat, self.bs, self.seed, self.epoch) = (np.asarray(cat), batch_size, seed, 0)
        self.groups = {}
        for c in np.unique(self.cat):
            self.groups[int(c)] = np.flatnonzero(self.cat == c)

    def set_epoch(self, e):
        self.epoch = e

    def __len__(self):
        return sum(((len(v) + self.bs - 1) // self.bs for v in self.groups.values()))

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        batches = []
        for idx in self.groups.values():
            idx = idx.copy()
            rng.shuffle(idx)
            for s in range(0, len(idx), self.bs):
                batches.append([int(i) for i in idx[s:s + self.bs]])
        rng.shuffle(batches)
        return iter(batches)

def ab_consistency_loss(model, enc, tok, frac=0.25):
    return None

def pairwise_rank_loss(logits, y, cat_ids, thr=0.5):
    total = logits.new_zeros(())
    n = 0
    for c in torch.unique(cat_ids):
        m = cat_ids == c
        (s, yy) = (logits[m], y[m])
        (pos, neg) = (yy > thr, yy < thr)
        if int(pos.sum()) == 0 or int(neg.sum()) == 0:
            continue
        d = s[pos].unsqueeze(1) - s[neg].unsqueeze(0)
        total = total + nn.functional.softplus(-d).mean()
        n += 1
    return total / n if n else total

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='ai-forever/ruBert-base')
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--batch', type=int, default=48)
    ap.add_argument('--maxlen', type=int, default=256)
    ap.add_argument('--lr', type=float, default=3e-05)
    ap.add_argument('--head_lr', type=float, default=0.001)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--min_conf', type=float, default=0.0, help='drop pairs with |2p-1|<min_conf')
    ap.add_argument('--base_weight', type=float, default=1.0, help='global weight for the old LLM-labelled base pool')
    ap.add_argument('--tag', default='rubase_pre')
    ap.add_argument('--eval_every', type=int, default=4000, help='steps between human-val probes')
    ap.add_argument('--workers', type=int, default=0)
    ap.add_argument('--accum', type=int, default=1)
    ap.add_argument('--grad_ckpt', action='store_true')
    ap.add_argument('--rank_lambda', type=float, default=1.0)
    ap.add_argument('--force_group', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--ab_lambda', type=float, default=0.0)
    ap.add_argument('--trust_remote_code', action='store_true')
    ap.add_argument('--llm_pairs_file', default='llm_pairs_2p5m.parquet')
    ap.add_argument('--llm_items_file', default='items_llm_2p5m.parquet')
    ap.add_argument('--exclude_pairs', default='', help='parquet with id1,id2 to drop from pretraining (e.g. hardval)')
    ap.add_argument('--exclude_items', default='', help='parquet with id,fp: drop pairs touching these items by id OR content fingerprint (entity-disjoint purge)')
    ap.add_argument('--extra_pairs', default='', help='parquet id1,id2,target: transitive/pseudo pairs')
    ap.add_argument('--extra_weight', type=float, default=1.0)
    ap.add_argument('--extra_weight_col', default='sample_weight', help='optional per-row curriculum weight in extra parquet; empty disables')
    ap.add_argument('--require_extra_audit', action='store_true', help='require a verified multi-teacher consensus audit sidecar')
    ap.add_argument('--verified_stream_fractions', type=float, nargs=3, metavar=('BASE', 'CONSENSUS', 'HUMAN'), default=None, help='required with --require_extra_audit: post-purge total loss-mass fractions for base, consensus and human streams')
    ap.add_argument('--allow_legacy_extra', action='store_true', help='explicit opt-in for old non-consensus pseudo-pair files')
    ap.add_argument('--extra_limit', type=int, default=0, help='deterministic cap for extra pairs (0 keeps all)')
    ap.add_argument('--extra_pos_frac', type=float, default=-1.0, help='positive share when sampling capped extra pairs; -1 keeps natural share')
    ap.add_argument('--mix_human', type=int, default=0, help='add N human train pairs (hard labels) into the pool')
    ap.add_argument('--human_weight', type=float, default=2.0)
    ap.add_argument('--shard', default='', help="'i/K': deterministic 1/K slice of the pair pool (for 3h-chained jobs)")
    args = ap.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if not np.isfinite(args.base_weight) or args.base_weight < 0:
        raise ValueError('--base_weight must be non-negative')
    if not np.isfinite(args.extra_weight) or args.extra_weight <= 0:
        raise ValueError('--extra_weight must be finite and strictly positive')
    if args.mix_human < 0:
        raise ValueError('--mix_human must be non-negative')
    if not np.isfinite(args.human_weight) or args.human_weight <= 0:
        raise ValueError('--human_weight must be finite and strictly positive')
    if args.require_extra_audit and (not args.extra_pairs):
        raise ValueError('--require_extra_audit needs --extra_pairs')
    if args.require_extra_audit and args.allow_legacy_extra:
        raise ValueError('verified consensus and --allow_legacy_extra are mutually exclusive')
    if args.extra_pairs and (not (args.require_extra_audit or args.allow_legacy_extra)):
        raise ValueError('extra pairs require either --require_extra_audit or explicit --allow_legacy_extra')
    verified_stream_fractions = None
    if args.verified_stream_fractions is not None:
        verified_stream_fractions = dict(zip(consensus_contract.STREAM_NAMES, args.verified_stream_fractions))
        values = np.asarray(list(verified_stream_fractions.values()), dtype=np.float64)
        if not np.isfinite(values).all() or (values <= 0).any() or (not np.isclose(values.sum(), 1.0, rtol=0, atol=1e-09)):
            raise ValueError('--verified_stream_fractions must contain three positive values summing to 1')
    if args.require_extra_audit and verified_stream_fractions is None:
        raise ValueError('verified consensus requires explicit --verified_stream_fractions')
    if not args.require_extra_audit and verified_stream_fractions is not None:
        raise ValueError('--verified_stream_fractions is only valid with --require_extra_audit')
    if args.require_extra_audit:
        if args.extra_weight_col != 'sample_weight':
            raise ValueError('verified consensus requires --extra_weight_col sample_weight')
        if args.extra_limit != 0 or args.extra_pos_frac != -1.0:
            raise ValueError('verified consensus cannot be capped or label-resampled downstream')
        if not args.exclude_pairs or not args.exclude_items:
            raise ValueError('verified consensus requires pair and entity/fingerprint exclusions')
        if args.base_weight <= 0:
            raise ValueError('verified consensus requires a non-zero base stream')
        if args.mix_human <= 0:
            raise ValueError('verified consensus requires a non-zero human stream')

    def dpath(name):
        if os.path.isabs(name):
            return name
        p = f'{DATA}/{name}'
        return p if os.path.exists(p) else f'{OUT}/{name}'
    verified_consensus = None
    if args.extra_pairs and args.require_extra_audit:
        extra_path = dpath(args.extra_pairs)
        audit_path = extra_path + '.audit.json'
        verified_consensus = consensus_contract.verify_consensus_artifact(extra_path, audit_path, exclude_pair_paths=[dpath(args.exclude_pairs)], exclude_item_paths=[dpath(args.exclude_items)], producer_script_path=os.path.join(os.path.dirname(__file__), 'teacher_consensus.py'))
        print(f'accepted fail-closed multi-teacher consensus contract: {audit_path}', flush=True)
    hf_logging.set_verbosity_error()
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    use_bf16 = device == 'cuda' and torch.cuda.get_device_capability()[0] >= 8
    use_fp16 = device == 'cuda' and (not use_bf16)
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16) if device == 'cuda' else None
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    if use_bf16 and args.accum > 1:
        args.batch *= args.accum
        args.accum = 1
        print(f'bf16 GPU: folded accumulation into batch={args.batch}', flush=True)
    print(f'amp: bf16={use_bf16} fp16={use_fp16}', flush=True)
    print('device:', device, '| model:', args.model, flush=True)
    pairs = pd.read_parquet(dpath(args.llm_pairs_file))
    litems = pd.read_parquet(dpath(args.llm_items_file))
    excluded_pair_keys = set()
    (ex_ids, ex_fps) = (set(), set())
    if args.exclude_pairs:
        if verified_consensus is not None:
            excluded_pair_keys = set(verified_consensus.exclusions.pair_keys)
        else:
            ex = pd.read_parquet(dpath(args.exclude_pairs))
            excluded_pair_keys = set(zip(ex.id1.values, ex.id2.values))
        m = np.array([(a, b) in excluded_pair_keys or (b, a) in excluded_pair_keys for (a, b) in zip(pairs.id1.values, pairs.id2.values)])
        print(f'excluding {m.sum():,} pairs listed in {args.exclude_pairs}', flush=True)
        pairs = pairs[~m].reset_index(drop=True)
    if args.shard:
        (i, k) = map(int, args.shard.split('/'))
        h = pd.util.hash_array(pairs.id1.values * 31 + pairs.id2.values) % k
        pairs = pairs[h == i].reset_index(drop=True)
        print(f'shard {i}/{k}: {len(pairs):,} pairs', flush=True)
    print(f'llm pairs: {len(pairs):,}; items: {len(litems):,}', flush=True)
    t0 = time.time()
    text_of = {}
    for (pid, n, a, c) in zip(litems['id'], litems['name'], litems['attributes'], litems['category']):
        text_of[pid] = make_text(n, c, a)
    print(f'texts built in {time.time() - t0:.0f}s', flush=True)
    del litems
    ok = pairs.id1.isin(text_of) & pairs.id2.isin(text_of)
    pairs = pairs[ok].reset_index(drop=True)
    import re
    ws = re.compile('\\s+')

    def fp(name, attrs, cat):
        s = f'{str(name).lower().strip()}|{str(attrs).lower().strip()}|{cat}'
        return hashlib.md5(ws.sub(' ', s).encode()).hexdigest()
    fpath = f'{DATA}/artifacts/folds_v1.parquet'
    if not os.path.exists(fpath):
        fpath = f'{DATA}/artifacts/folds_v1.parquet'
    _folds = pd.read_parquet(fpath)
    _val_ids = set(pd.unique(pd.concat([_folds[_folds.fold == 0].id1, _folds[_folds.fold == 0].id2])))
    _ih = pd.read_parquet(f'{DATA}/items_human.parquet')
    _ih = _ih[_ih['id'].isin(_val_ids)]
    val_fps = {fp(n, a, c) for (n, a, c) in zip(_ih['name'], _ih['attributes'], _ih['category'])}
    del _ih
    litems2 = pd.read_parquet(dpath(args.llm_items_file))
    llm_fp = {pid: fp(n, a, c) for (pid, n, a, c) in zip(litems2['id'], litems2['name'], litems2['attributes'], litems2['category'])}
    del litems2
    bad = pairs.id1.map(llm_fp).isin(val_fps) | pairs.id2.map(llm_fp).isin(val_fps)
    print(f'dropping {bad.sum():,} llm pairs with content match to human-val items', flush=True)
    pairs = pairs[~bad].reset_index(drop=True)
    if args.exclude_items:
        if verified_consensus is not None:
            ex_ids = set(verified_consensus.exclusions.item_ids)
            ex_fps = set(verified_consensus.exclusions.item_fingerprints)
        else:
            ex = pd.read_parquet(dpath(args.exclude_items))
            ex_ids = set(ex['id'].values)
            ex_fps = set(ex['fp'].values)
        bad2 = pairs.id1.isin(ex_ids) | pairs.id2.isin(ex_ids) | pairs.id1.map(llm_fp).isin(ex_fps) | pairs.id2.map(llm_fp).isin(ex_fps)
        print(f'entity-disjoint purge: dropping {bad2.sum():,} pairs touching {len(ex_ids):,} excluded items', flush=True)
        pairs = pairs[~bad2].reset_index(drop=True)
    conf = np.abs(2 * pairs.target.values - 1)
    keep = conf >= args.min_conf
    pairs = pairs[keep].reset_index(drop=True)
    conf = conf[keep]
    if args.limit:
        take = np.sort(np.random.default_rng(0).choice(len(pairs), size=min(args.limit, len(pairs)), replace=False))
        pairs = pairs.iloc[take].reset_index(drop=True)
        conf = conf[take]
    print(f'training on {len(pairs):,} llm pairs (soft targets, conf weight)', flush=True)
    t1 = pairs.id1.map(text_of).values
    t2 = pairs.id2.map(text_of).values
    y = pairs.target.values.astype(np.float32)
    w = (args.base_weight * (0.3 + 0.7 * conf)).astype(np.float32)
    base_count = len(y)
    consensus_count = 0
    human_count = 0
    if args.extra_pairs:
        extra_path = dpath(args.extra_pairs)
        if verified_consensus is not None:
            ex = verified_consensus.frame.copy()
        else:
            ex = pd.read_parquet(extra_path)
            consensus_markers = {'consensus_probability', 'consensus_quality', 'confident_teacher_count', 'teacher_probability_spread'}
            if os.path.isfile(extra_path + '.audit.json') or consensus_markers & set(ex.columns):
                raise RuntimeError('consensus-shaped extra pairs cannot bypass --require_extra_audit')
        required = {'id1', 'id2', 'target'}
        missing = required.difference(ex.columns)
        if missing:
            raise ValueError(f'{args.extra_pairs} is missing columns: {sorted(missing)}')
        ok_e = ex.id1.isin(text_of) & ex.id2.isin(text_of) & (ex.id1 != ex.id2)
        if verified_consensus is not None and (not bool(ok_e.all())):
            raise RuntimeError(f'verified consensus lost {int((~ok_e).sum()):,} rows against its bound item catalog')
        ex = ex[ok_e].reset_index(drop=True)
        bad_e = ex.id1.map(llm_fp).isin(val_fps) | ex.id2.map(llm_fp).isin(val_fps) | ex.id1.isin(ex_ids) | ex.id2.isin(ex_ids) | ex.id1.map(llm_fp).isin(ex_fps) | ex.id2.map(llm_fp).isin(ex_fps)
        if excluded_pair_keys:
            bad_pair_e = np.array([(a, b) in excluded_pair_keys or (b, a) in excluded_pair_keys for (a, b) in zip(ex.id1.values, ex.id2.values)])
            bad_e = bad_e | bad_pair_e
        print(f'extra entity/pair purge: dropping {int(np.asarray(bad_e).sum()):,} / {len(ex):,}', flush=True)
        if verified_consensus is not None and bool(np.asarray(bad_e).any()):
            raise RuntimeError('verified consensus contains pair/entity/fingerprint leakage')
        ex = ex[~np.asarray(bad_e)].copy()
        ex['_pair_lo'] = np.minimum(ex.id1.to_numpy(dtype=np.int64), ex.id2.to_numpy(dtype=np.int64))
        ex['_pair_hi'] = np.maximum(ex.id1.to_numpy(dtype=np.int64), ex.id2.to_numpy(dtype=np.int64))
        duplicated_extra = ex.duplicated(['_pair_lo', '_pair_hi'], keep=False)
        if verified_consensus is not None and bool(duplicated_extra.any()):
            raise RuntimeError('verified consensus contains duplicate unordered pairs')
        ex = ex.drop_duplicates(['_pair_lo', '_pair_hi']).reset_index(drop=True)
        if not -1.0 <= args.extra_pos_frac <= 1.0:
            raise ValueError('--extra_pos_frac must be -1 or a value in [0, 1]')
        if args.extra_limit and len(ex) > args.extra_limit:
            rng = np.random.default_rng(0)
            if args.extra_pos_frac < 0:
                take = rng.choice(len(ex), args.extra_limit, replace=False)
            else:
                is_pos = ex.target.to_numpy() >= 0.5
                (pos_idx, neg_idx) = (np.flatnonzero(is_pos), np.flatnonzero(~is_pos))
                want_pos = min(len(pos_idx), int(round(args.extra_limit * args.extra_pos_frac)))
                want_neg = min(len(neg_idx), args.extra_limit - want_pos)
                take = np.concatenate([rng.choice(pos_idx, want_pos, replace=False), rng.choice(neg_idx, want_neg, replace=False)])
                spare = args.extra_limit - len(take)
                if spare:
                    remaining = np.setdiff1d(np.arange(len(ex)), take, assume_unique=True)
                    take = np.concatenate([take, rng.choice(remaining, spare, replace=False)])
                rng.shuffle(take)
            ex = ex.iloc[take].reset_index(drop=True)
        extra_keys = pd.MultiIndex.from_frame(ex[['_pair_lo', '_pair_hi']])
        base_lo = np.minimum(pairs.id1.to_numpy(dtype=np.int64), pairs.id2.to_numpy(dtype=np.int64))
        base_hi = np.maximum(pairs.id1.to_numpy(dtype=np.int64), pairs.id2.to_numpy(dtype=np.int64))
        overlap = pd.MultiIndex.from_arrays([base_lo, base_hi]).isin(extra_keys)
        if overlap.any():
            print(f'teacher override: removing {int(overlap.sum()):,} old-label base pairs', flush=True)
            pairs = pairs.loc[~overlap].reset_index(drop=True)
            (t1, t2, y, w) = (t1[~overlap], t2[~overlap], y[~overlap], w[~overlap])
        ex = ex.drop(columns=['_pair_lo', '_pair_hi'])
        if 'source_target' in ex:
            print('dropping diagnostic-only extra source_target before training', flush=True)
            ex = ex.drop(columns=['source_target'])
        extra_targets = ex.target.to_numpy(dtype=np.float32)
        if not np.isfinite(extra_targets).all() or ((extra_targets < 0) | (extra_targets > 1)).any():
            raise ValueError('extra target must be finite in [0,1]')
        ex1 = ex.id1.map(text_of).values
        ex2 = ex.id2.map(text_of).values
        base_count = len(y)
        consensus_count = len(extra_targets)
        t1 = np.concatenate([t1, ex1])
        t2 = np.concatenate([t2, ex2])
        y = np.concatenate([y, extra_targets])
        if args.extra_weight_col and args.extra_weight_col in ex:
            extra_w = ex[args.extra_weight_col].to_numpy(dtype=np.float32) * args.extra_weight
            if not np.isfinite(extra_w).all() or (extra_w <= 0).any():
                raise ValueError(f'invalid weights in extra column {args.extra_weight_col!r}')
            weight_desc = f'{args.extra_weight} * {args.extra_weight_col}'
        else:
            if verified_consensus is not None:
                raise RuntimeError('verified consensus is missing audited sample_weight')
            extra_w = np.full(len(ex), args.extra_weight, dtype=np.float32)
            weight_desc = str(args.extra_weight)
        w = np.concatenate([w, extra_w])
        pos_share = float((ex.target.values >= 0.5).mean()) if len(ex) else float('nan')
        print(f'added {len(ex):,} extra pairs (positive share {pos_share:.3f}, weight {weight_desc}); pool now {len(y):,}', flush=True)
    if args.mix_human:
        hf = pd.read_parquet(fpath)
        human_pool = hf[hf.fold != 0].copy()
        ih = pd.read_parquet(f'{DATA}/items_human.parquet').set_index('id')
        hneed = pd.unique(pd.concat([human_pool.id1, human_pool.id2]))
        missing_human_items = set((int(value) for value in hneed)) - set(ih.index.astype(int))
        if missing_human_items:
            raise RuntimeError(f'human item catalog misses {len(missing_human_items):,} training entities')
        human_cards = ih.loc[hneed]
        human_fp = {int(pid): fp(name, attributes, category) for (pid, name, attributes, category) in zip(human_cards.index, human_cards['name'], human_cards['attributes'], human_cards['category'])}
        (human_bad, human_purge_audit) = consensus_contract.pair_entity_fingerprint_exclusion_mask(human_pool, pair_keys=excluded_pair_keys, item_ids=set(ex_ids) | set(_val_ids), item_fingerprint_by_id=human_fp, forbidden_fingerprints=set(ex_fps) | set(val_fps), source='human training stream')
        print('human pair/entity/fingerprint purge: ' + json.dumps(human_purge_audit, sort_keys=True), flush=True)
        human_pool = human_pool.loc[~human_bad].reset_index(drop=True)
        htr = human_pool.sample(min(args.mix_human, len(human_pool)), random_state=1 + args.seed)
        hneed = pd.unique(pd.concat([htr.id1, htr.id2]))
        htxt = {pid: make_text(r['name'], r['category'], r['attributes']) for (pid, r) in ih.loc[hneed].iterrows()}
        t1 = np.concatenate([t1, htr.id1.map(htxt).values])
        t2 = np.concatenate([t2, htr.id2.map(htxt).values])
        y = np.concatenate([y, htr.target.values.astype(np.float32)])
        w = np.concatenate([w, np.full(len(htr), args.human_weight, dtype=np.float32)])
        human_count = len(htr)
        print(f'mixed in {len(htr):,} human pairs (weight {args.human_weight}); pool now {len(y):,}', flush=True)
    if verified_consensus is not None:
        stream_counts = {'base': base_count, 'consensus': consensus_count, 'human': human_count}
        (w, stream_mass_audit) = consensus_contract.normalize_stream_weights(w, stream_counts, verified_stream_fractions)
        print('verified stream mass audit: ' + json.dumps(stream_mass_audit, sort_keys=True, ensure_ascii=True), flush=True)
    fp = f'{DATA}/artifacts/folds_v1.parquet'
    if not os.path.exists(fp):
        fp = f'{DATA}/artifacts/folds_v1.parquet'
    folds = pd.read_parquet(fp)
    hval = folds[folds.fold == 0].sample(20000, random_state=0)
    items_h = pd.read_parquet(f'{DATA}/items_human.parquet').set_index('id')
    need = pd.unique(pd.concat([hval.id1, hval.id2]))
    htext = {pid: make_text(r['name'], r['category'], r['attributes']) for (pid, r) in items_h.loc[need].iterrows()}
    hv1 = hval.id1.map(htext).values
    hv2 = hval.id2.map(htext).values
    hy = hval.target.values
    hcat = hval.category.values
    del items_h
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=1, trust_remote_code=args.trust_remote_code).to(device)
    if getattr(model.config, 'pad_token_id', None) is None:
        model.config.pad_token_id = tok.pad_token_id
    if args.grad_ckpt:
        model.gradient_checkpointing_enable()

    def probe():
        model.eval()
        if hv2 is None:
            order = np.argsort([len(a) for a in hv1])
        else:
            order = np.argsort([len(a) + len(b) for (a, b) in zip(hv1, hv2)])
        sc = np.zeros(len(order))
        with torch.inference_mode():
            for s in range(0, len(order), 256):
                oi = order[s:s + 256]
                if hv2 is None:
                    enc = tok(list(hv1[oi]), padding=True, truncation=True, max_length=args.maxlen, return_tensors='pt')
                else:
                    enc = tok(list(hv1[oi]), list(hv2[oi]), padding=True, truncation=True, max_length=args.maxlen, return_tensors='pt')
                enc = {k: v.to(device) for (k, v) in enc.items()}
                logits = model(**enc).logits.squeeze(-1)
                sc[oi] = torch.sigmoid(logits.float()).cpu().numpy()
        model.train()
        return np.mean([average_precision_score(hy[hcat == c], sc[hcat == c]) for c in np.unique(hcat) if hy[hcat == c].sum() > 0])

    def collate(batch):
        (a, b, yy, ww, cc) = zip(*batch)
        enc = tok(list(a), list(b), padding=True, truncation=True, max_length=args.maxlen, return_tensors='pt')
        return (enc, torch.tensor(yy), torch.tensor(ww), torch.tensor(cc))
    head = [p for (n_, p) in model.named_parameters() if 'classifier' in n_ or 'score' in n_]
    body = [p for (n_, p) in model.named_parameters() if not ('classifier' in n_ or 'score' in n_)]
    opt = torch.optim.AdamW([{'params': body, 'lr': args.lr}, {'params': head, 'lr': args.head_lr}], weight_decay=0.01)

    def _cat_of(s):
        s = str(s)
        if s.startswith('[CAT] '):
            return s[6:].split('\n', 1)[0].strip()
        return s.split(' | ', 1)[0].strip()
    _cnames = np.array([_cat_of(s) for s in t1])
    _cuniq = {c: i for (i, c) in enumerate(sorted(set(_cnames.tolist())))}
    cat_ids = np.array([_cuniq[c] for c in _cnames], dtype=np.int64)
    print(f'категорий в обучении: {len(_cuniq)}; rank_lambda={args.rank_lambda}', flush=True)
    _ds = SoftPairDS(t1, t2, y, w, cat=cat_ids)
    if args.rank_lambda < 1.0 or args.force_group:
        _sampler = CatGroupedSampler(cat_ids, args.batch, seed=args.seed)
        dl = DataLoader(_ds, batch_sampler=_sampler, num_workers=args.workers, collate_fn=collate)
    else:
        _sampler = None
        dl = DataLoader(_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, collate_fn=collate)
    total = len(dl) * args.epochs
    sched_total = args.epochs * (len(dl) // args.accum) + 2
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[args.lr, args.head_lr], total_steps=sched_total, pct_start=0.03)
    lossf = nn.BCEWithLogitsLoss(reduction='none')
    print(f'steps total: {total}; probe every {args.eval_every}', flush=True)
    initial_ap = probe()
    print(f'probe before training: human macro AP = {initial_ap:.4f}', flush=True)
    best = initial_ap
    model.save_pretrained(f'{OUT}/ce_{args.tag}_best')
    tok.save_pretrained(f'{OUT}/ce_{args.tag}_best')
    (step, t0) = (0, time.time())
    for ep in range(args.epochs):
        if _sampler is not None:
            _sampler.set_epoch(ep)
        for (enc, yb, wb, cb) in dl:
            enc = {k: v.to(device) for (k, v) in enc.items()}
            (yb, wb, cb) = (yb.to(device), wb.to(device), cb.to(device))
            if use_bf16 or use_fp16:
                with torch.autocast('cuda', dtype=amp_dtype):
                    logits = model(**enc).logits.squeeze(-1)
                    _bce = (lossf(logits.float(), yb) * wb).mean()
                    if args.rank_lambda < 1.0:
                        _rk = pairwise_rank_loss(logits.float(), yb, cb)
                        loss = (args.rank_lambda * _bce + (1 - args.rank_lambda) * _rk) / args.accum
                    else:
                        loss = _bce / args.accum
            else:
                logits = model(**enc).logits.squeeze(-1)
                _bce = (lossf(logits, yb) * wb).mean()
                if args.rank_lambda < 1.0:
                    _rk = pairwise_rank_loss(logits, yb, cb)
                    loss = (args.rank_lambda * _bce + (1 - args.rank_lambda) * _rk) / args.accum
                else:
                    loss = _bce / args.accum
            if use_fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            step += 1
            if step % args.accum == 0:
                if use_fp16:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                if sched.last_epoch < sched_total - 1:
                    sched.step()
                opt.zero_grad(set_to_none=True)
            if step % 500 == 0:
                print(f'  step {step}/{total} loss {loss.item():.4f} {step * args.batch / (time.time() - t0):.0f} pairs/s', flush=True)
            if step % args.eval_every == 0:
                ap_ = probe()
                print(f'== step {step}: human-val macro AP = {ap_:.4f} ==', flush=True)
                if ap_ > best:
                    best = ap_
                    model.save_pretrained(f'{OUT}/ce_{args.tag}_best')
                    tok.save_pretrained(f'{OUT}/ce_{args.tag}_best')
    ap_ = probe()
    print(f'final probe: {ap_:.4f}; best {max(best, ap_):.4f}', flush=True)
    if ap_ > best:
        best = ap_
        model.save_pretrained(f'{OUT}/ce_{args.tag}_best')
        tok.save_pretrained(f'{OUT}/ce_{args.tag}_best')
    model.save_pretrained(f'{OUT}/ce_{args.tag}')
    tok.save_pretrained(f'{OUT}/ce_{args.tag}')
    print(f'saved ce_{args.tag}')
if __name__ == '__main__':
    main()
