from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import consensus_contract as training_contract
PAIR_COLUMNS = ('id1', 'id2')
SCORE_COLUMNS = {'teacher_logit', 'teacher_sequence_logprob_diff', 'teacher_probability', 'teacher_probability_raw', 'teacher_logit_ab', 'teacher_logit_ba', 'teacher_order_gap', 'teacher_calibration_slope', 'teacher_calibration_intercept', 'teacher_yes_token', 'teacher_no_token', 'teacher_scoring', 'input_tokens', 'input_tokens_ba', 'was_truncated', 'prompt_compaction', 'prompt_compaction_ba'}
DEFAULT_RULE_COLUMNS = ('identifier_conflict', 'quantity_conflict', 'model_conflict')
PRODUCTION_MIN_GATE_AP = 0.58
PRODUCTION_MIN_NEGATIVE_PRECISION = 0.85
PRODUCTION_MIN_POSITIVE_PRECISION = 0.7
PRODUCTION_MIN_CLASS_ROWS = 20
PRODUCTION_MIN_CALIBRATION_ROWS = 1000
PRODUCTION_MIN_HARD_NEGATIVE_ROWS = 200
PRODUCTION_MAX_HARD_NEGATIVE_FALSE_ACCEPT_RATE = 0.1
PRODUCTION_EXPECTED_HARD_NEGATIVE_CATEGORIES = 20
PRODUCTION_MIN_HARD_NEGATIVE_ROWS_PER_CATEGORY = 10
PRODUCTION_MAX_HARD_NEGATIVE_FALSE_ACCEPT_RATE_PER_CATEGORY = 0.25
NAME_RE = re.compile('^[A-Za-z][A-Za-z0-9_]*$')
WS_RE = re.compile('\\s+')

@dataclass(frozen=True)
class TeacherSpec:
    name: str
    score_path: Path
    gate_path: Path
    multiplier: float

@dataclass(frozen=True)
class GateInfo:
    path: Path
    model: str
    ap: float
    constant_ap: float
    slope: float
    intercept: float
    reference_positive_prior: float
    calibration_rows: int
    prompt_contract_sha256: str
    report: dict

def stable_sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    out = np.empty_like(values)
    positive = values >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    out[~positive] = exp_values / (1.0 + exp_values)
    return out

def sha256_file(path: Path, block_size: int=1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while True:
            block = stream.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()

def content_fingerprint(name: object, attributes: object, category: object) -> str:
    raw = f'{str(name).lower().strip()}|{str(attributes).lower().strip()}|{category}'
    return hashlib.md5(WS_RE.sub(' ', raw).encode('utf-8')).hexdigest()

def integral_ids(series: pd.Series, column: str) -> np.ndarray:
    if series.isna().any():
        raise ValueError(f'{column} contains null IDs')
    try:
        values = series.to_numpy(dtype=np.int64, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{column} must contain int64 IDs') from exc
    numeric = pd.to_numeric(series, errors='coerce').to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all() or not np.array_equal(numeric, values.astype(np.float64)):
        raise ValueError(f'{column} contains non-integral IDs')
    return values

def canonicalize(frame: pd.DataFrame, source: str, require_unique: bool=True) -> pd.DataFrame:
    if not set(PAIR_COLUMNS).issubset(frame.columns):
        raise ValueError(f'{source} needs id1,id2; columns={list(frame.columns)}')
    out = frame.copy()
    left = integral_ids(out['id1'], f'{source}:id1')
    right = integral_ids(out['id2'], f'{source}:id2')
    if np.any(left == right):
        raise ValueError(f'{source} contains {int(np.sum(left == right)):,} self-pairs')
    out['_lo'] = np.minimum(left, right)
    out['_hi'] = np.maximum(left, right)
    duplicate = out.duplicated(['_lo', '_hi'], keep=False)
    if require_unique and duplicate.any():
        raise ValueError(f'{source} contains {int(duplicate.sum()):,} rows in duplicate unordered pairs')
    return out

def pair_key_set(frame: pd.DataFrame) -> set[tuple[int, int]]:
    canonical = canonicalize(frame, 'pair exclusion', require_unique=False)
    return set(zip(canonical['_lo'].astype(int), canonical['_hi'].astype(int)))

def parse_named_paths(values: Sequence[str], flag: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if '=' not in value:
            raise ValueError(f'{flag} expects NAME=PATH, got {value!r}')
        (name, raw_path) = (part.strip() for part in value.split('=', 1))
        if not NAME_RE.fullmatch(name):
            raise ValueError(f'invalid teacher name {name!r}')
        if name in result:
            raise ValueError(f'duplicate {flag} name {name!r}')
        path = Path(raw_path).expanduser()
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f'{flag} file is absent or empty: {path}')
        result[name] = path.resolve()
    return result

def parse_multipliers(values: Sequence[str]) -> dict[str, float]:
    result: dict[str, float] = {}
    for value in values:
        if '=' not in value:
            raise ValueError(f'--teacher-multiplier expects NAME=FLOAT, got {value!r}')
        (name, raw_weight) = (part.strip() for part in value.split('=', 1))
        weight = float(raw_weight)
        if not NAME_RE.fullmatch(name) or not math.isfinite(weight) or weight <= 0:
            raise ValueError(f'invalid teacher multiplier {value!r}')
        if name in result:
            raise ValueError(f'duplicate multiplier for {name!r}')
        result[name] = weight
    return result

def build_specs(args: argparse.Namespace) -> list[TeacherSpec]:
    score_paths = parse_named_paths(args.teacher, '--teacher')
    gate_paths = parse_named_paths(args.gate, '--gate')
    multipliers = parse_multipliers(args.teacher_multiplier)
    if set(score_paths) != set(gate_paths):
        raise ValueError(f'teacher/gate names differ: scores-only={sorted(set(score_paths) - set(gate_paths))}, gates-only={sorted(set(gate_paths) - set(score_paths))}')
    unknown_weights = set(multipliers) - set(score_paths)
    if unknown_weights:
        raise ValueError(f'multipliers reference unknown teachers: {sorted(unknown_weights)}')
    if len(score_paths) < 2 and (not args.allow_single_teacher):
        raise ValueError('at least two independently gated teachers are required')
    if len(set(gate_paths.values())) != len(gate_paths):
        raise ValueError('each teacher must have its own gate JSON; duplicate path detected')
    return [TeacherSpec(name, score_paths[name], gate_paths[name], multipliers.get(name, 1.0)) for name in score_paths]

def load_gate(spec: TeacherSpec, args: argparse.Namespace) -> GateInfo:
    with spec.gate_path.open(encoding='utf-8') as stream:
        report = json.load(stream)
    invariants = {'passed': True, 'exact_logit_difference': True, 'exact_full_sequence_logprob': True, 'no_tokenizer_truncation': True, 'bidirectional': True, 'calibration_evaluation_disjoint': True}
    bad = {key: report.get(key) for (key, expected) in invariants.items() if report.get(key) != expected}
    if bad:
        raise RuntimeError(f'{spec.name}: gate failed exact-scoring invariants: {bad}')
    model = str(report.get('model', '')).strip()
    if not model:
        raise RuntimeError(f'{spec.name}: gate has no model identity')
    ap_value = float(report.get('teacher_macro_ap', float('nan')))
    if not math.isfinite(ap_value) or ap_value < args.min_gate_ap or ap_value > 1.0:
        raise RuntimeError(f'{spec.name}: representative gate AP {ap_value!r} is below --min-gate-ap={args.min_gate_ap} or invalid')
    constant_ap = float(report.get('teacher_macro_ap_constant', float('nan')))
    if not math.isfinite(constant_ap) or not 0.0 < constant_ap < ap_value:
        raise RuntimeError(f'{spec.name}: gate lacks a valid category-constant AP baseline')
    prompt_hash = str(report.get('prompt_contract_sha256', ''))
    if not re.fullmatch('[0-9a-f]{64}', prompt_hash):
        raise RuntimeError(f'{spec.name}: gate lacks a valid prompt contract hash')
    calibration = report.get('calibration', {})
    if calibration.get('kind') != 'global_platt_category_class_balanced_prior_explicit_v2':
        raise RuntimeError(f'{spec.name}: gate lacks v2 explicit-prior Platt calibration')
    if calibration.get('fit_role') != 'calibration':
        raise RuntimeError(f'{spec.name}: calibration was not fit on the calibration role')
    if calibration.get('fit_stratum') != 'representative':
        raise RuntimeError(f"{spec.name}: calibration must use representative human rows, got {calibration.get('fit_stratum')!r}")
    rows = int(calibration.get('rows', 0))
    if rows < args.min_gate_calibration_rows:
        raise RuntimeError(f'{spec.name}: calibration has only {rows:,} representative rows; need {args.min_gate_calibration_rows:,}')
    slope = float(calibration.get('slope', float('nan')))
    intercept = float(calibration.get('intercept', float('nan')))
    reference_prior = float(calibration.get('reference_positive_prior', float('nan')))
    if not math.isfinite(slope) or not math.isfinite(intercept) or slope <= 0:
        raise RuntimeError(f'{spec.name}: invalid/reversed calibration (slope={slope!r}, intercept={intercept!r})')
    if not 0.0 < reference_prior < 1.0:
        raise RuntimeError(f'{spec.name}: invalid calibration reference prior')
    decision = report.get('heldout_decision_audit', {})
    for (key, expected) in (('negative_threshold', args.negative_threshold), ('positive_threshold', args.positive_threshold)):
        actual = float(decision.get(key, float('nan')))
        if not math.isclose(actual, expected, rel_tol=0, abs_tol=args.calibration_tolerance):
            raise RuntimeError(f'{spec.name}: consensus {key}={expected} was not validated by gate ({actual})')
    for key in ('negative_precision', 'positive_precision'):
        if not math.isfinite(float(decision.get(key, float('nan')))):
            raise RuntimeError(f'{spec.name}: gate lacks held-out {key}')
    decision_floors = (('negative_precision', PRODUCTION_MIN_NEGATIVE_PRECISION), ('positive_precision', PRODUCTION_MIN_POSITIVE_PRECISION), ('negative_rows', PRODUCTION_MIN_CLASS_ROWS), ('positive_rows', PRODUCTION_MIN_CLASS_ROWS))
    for (key, floor) in decision_floors:
        actual = float(decision.get(key, float('nan')))
        if not math.isfinite(actual) or actual < floor:
            raise RuntimeError(f'{spec.name}: held-out {key}={actual!r} is below production floor {floor}')
    hard = report.get('hard_negative_audit', {})
    for (key, expected) in (('negative_threshold', args.negative_threshold), ('positive_threshold', args.positive_threshold)):
        actual = float(hard.get(key, float('nan')))
        if not math.isclose(actual, expected, rel_tol=0, abs_tol=args.calibration_tolerance):
            raise RuntimeError(f'{spec.name}: hard-negative audit {key}={actual!r} was not validated at consensus threshold {expected}')
    hard_rows = int(hard.get('rows', 0))
    hard_negative_rows = int(hard.get('negative_rows', 0))
    false_accept_rate = float(hard.get('confident_positive_false_accept_rate', float('nan')))
    if hard_rows < PRODUCTION_MIN_HARD_NEGATIVE_ROWS or hard_negative_rows < PRODUCTION_MIN_HARD_NEGATIVE_ROWS or (not math.isfinite(false_accept_rate)) or (false_accept_rate > PRODUCTION_MAX_HARD_NEGATIVE_FALSE_ACCEPT_RATE):
        raise RuntimeError(f'{spec.name}: hard-negative false-accept audit failed (rows={hard_rows}, negative_rows={hard_negative_rows}, rate={false_accept_rate!r})')
    by_category = hard.get('by_category', {})
    representative_categories = set(report.get('heldout_category_audit', {}))
    if not isinstance(by_category, dict) or int(hard.get('expected_categories', 0)) != PRODUCTION_EXPECTED_HARD_NEGATIVE_CATEGORIES or int(hard.get('observed_categories', 0)) != PRODUCTION_EXPECTED_HARD_NEGATIVE_CATEGORIES or (hard.get('categories_complete') is not True) or (len(by_category) != PRODUCTION_EXPECTED_HARD_NEGATIVE_CATEGORIES) or (set(by_category) != representative_categories):
        raise RuntimeError(f'{spec.name}: hard-negative category coverage is incomplete')
    for (category, category_audit) in by_category.items():
        rows = int(category_audit.get('rows', 0))
        negatives = int(category_audit.get('negative_rows', 0))
        rate = float(category_audit.get('confident_positive_false_accept_rate', float('nan')))
        if category_audit.get('passed') is not True or rows < PRODUCTION_MIN_HARD_NEGATIVE_ROWS_PER_CATEGORY or negatives < PRODUCTION_MIN_HARD_NEGATIVE_ROWS_PER_CATEGORY or (not math.isfinite(rate)) or (rate > PRODUCTION_MAX_HARD_NEGATIVE_FALSE_ACCEPT_RATE_PER_CATEGORY):
            raise RuntimeError(f'{spec.name}: hard-negative category audit failed for {category!r} (rows={rows}, negative_rows={negatives}, rate={rate!r})')
    if hard.get('passed') is not True:
        raise RuntimeError(f'{spec.name}: hard-negative audit did not pass')
    if report.get('baseline_column') != 'gbdt':
        raise RuntimeError(f'{spec.name}: gate lacks the fixed GBDT blend audit')
    if not math.isclose(float(report.get('fixed_blend_weight', float('nan'))), 0.1, rel_tol=0, abs_tol=args.calibration_tolerance):
        raise RuntimeError(f'{spec.name}: gate used a different fixed blend weight')
    blend_delta = float(report.get('fixed_blend_delta', float('nan')))
    halves = report.get('component_halves', {})
    if not math.isfinite(blend_delta) or blend_delta < 0 or set(halves) != {'0', '1'}:
        raise RuntimeError(f'{spec.name}: gate lacks a non-regressing fixed blend audit')
    for half in ('0', '1'):
        delta = float(halves[half].get('delta', float('nan')))
        if not math.isfinite(delta) or delta < 0:
            raise RuntimeError(f'{spec.name}: fixed blend regresses on component half {half}')
    return GateInfo(spec.gate_path, model, ap_value, constant_ap, slope, intercept, reference_prior, rows, prompt_hash, report)

def load_score_metadata(spec: TeacherSpec, gate: GateInfo, args: argparse.Namespace) -> dict:
    path = Path(str(spec.score_path) + '.meta.json')
    if not path.is_file():
        if args.allow_missing_score_meta:
            return {'missing': True}
        raise RuntimeError(f'{spec.name}: score metadata is required: {path}')
    with path.open(encoding='utf-8') as stream:
        meta = json.load(stream)
    if meta.get('model') != gate.model:
        raise RuntimeError(f"{spec.name}: mass score model {meta.get('model')!r} != gate model {gate.model!r}")
    if meta.get('tokenizer_truncation') is not False:
        raise RuntimeError(f'{spec.name}: score metadata does not prove truncation=False')
    if meta.get('scoring') != 'exact_full_sequence_logprob_yes_minus_no':
        raise RuntimeError(f'{spec.name}: score metadata describes a non-exact scorer')
    for (key, expected) in (('yes_word', 'да'), ('no_word', 'нет'), ('yes_token_id', gate.report.get('yes_token_id')), ('no_token_id', gate.report.get('no_token_id'))):
        if meta.get(key) != expected:
            raise RuntimeError(f'{spec.name}: mass verbalizer metadata {key}={meta.get(key)!r}, expected {expected!r} from gate')
    if meta.get('verbalizer_sequence_lengths') != {'да': 1, 'нет': 1}:
        raise RuntimeError(f'{spec.name}: verbalizers are not exact one-token sequences')
    if meta.get('bidirectional') is not True and (not args.allow_missing_order_gap):
        raise RuntimeError(f'{spec.name}: bidirectional scoring is required for order audit')
    if meta.get('prompt_contract_sha256') != gate.prompt_contract_sha256:
        raise RuntimeError(f'{spec.name}: mass scores used a different prompt than their gate')
    calibration = meta.get('probability_calibration', {})
    for (key, expected) in (('kind', 'global_platt_category_class_balanced_prior_explicit_v2'), ('fit_role', 'calibration'), ('fit_stratum', 'representative')):
        if calibration.get(key) != expected:
            raise RuntimeError(f'{spec.name}: score metadata calibration {key} is invalid')
    for (key, expected) in (('slope', gate.slope), ('intercept', gate.intercept)):
        actual = float(calibration.get(key, float('nan')))
        if not math.isclose(actual, expected, rel_tol=0, abs_tol=args.calibration_tolerance):
            raise RuntimeError(f'{spec.name}: score metadata uses another gate calibration: {key}={actual!r}, expected {expected!r}')
    return meta

def derive_order_probability_gap(frame: pd.DataFrame, gate: GateInfo, args: argparse.Namespace, name: str) -> np.ndarray:
    if {'teacher_logit_ab', 'teacher_logit_ba'}.issubset(frame.columns):
        ab = frame['teacher_logit_ab'].to_numpy(dtype=np.float64)
        ba = frame['teacher_logit_ba'].to_numpy(dtype=np.float64)
        if not np.isfinite(ab).all() or not np.isfinite(ba).all():
            raise RuntimeError(f'{name}: orientation margins contain NaN/inf')
        return np.abs(stable_sigmoid(gate.slope * ab + gate.intercept) - stable_sigmoid(gate.slope * ba + gate.intercept))
    if 'teacher_order_gap' in frame:
        margin = frame['teacher_logit'].to_numpy(dtype=np.float64)
        half_gap = 0.5 * frame['teacher_order_gap'].to_numpy(dtype=np.float64)
        if not np.isfinite(half_gap).all() or np.any(half_gap < 0):
            raise RuntimeError(f'{name}: teacher_order_gap is invalid')
        high = stable_sigmoid(gate.slope * (margin + half_gap) + gate.intercept)
        low = stable_sigmoid(gate.slope * (margin - half_gap) + gate.intercept)
        return np.abs(high - low)
    if not args.allow_missing_order_gap:
        raise RuntimeError(f'{name}: no bidirectional margins/order gap')
    return np.full(len(frame), np.nan, dtype=np.float64)

def validate_and_calibrate_teacher(spec: TeacherSpec, gate: GateInfo, args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    meta = load_score_metadata(spec, gate, args)
    frame = canonicalize(pd.read_parquet(spec.score_path), str(spec.score_path))
    required = {'teacher_logit', 'teacher_probability_raw', 'teacher_probability'}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f'{spec.name}: score parquet misses {sorted(missing)}')
    if 'was_truncated' in frame and frame['was_truncated'].astype(bool).any():
        raise RuntimeError(f'{spec.name}: truncated teacher prompts are forbidden')
    if 'teacher_scoring' in frame and set(frame['teacher_scoring'].astype(str)) != {'exact_full_sequence_logprob_yes_minus_no'}:
        raise RuntimeError(f'{spec.name}: non-exact teacher scoring rows found')
    for (column, expected) in (('teacher_yes_token', gate.report.get('yes_token_id')), ('teacher_no_token', gate.report.get('no_token_id'))):
        if column not in frame or frame[column].nunique(dropna=False) != 1:
            raise RuntimeError(f'{spec.name}: missing/non-constant {column}')
        if int(frame[column].iloc[0]) != int(expected):
            raise RuntimeError(f'{spec.name}: {column} differs from its gate')
    margin = frame['teacher_logit'].to_numpy(dtype=np.float64)
    if not np.isfinite(margin).all():
        raise RuntimeError(f'{spec.name}: raw margins contain NaN/inf')
    raw_expected = stable_sigmoid(margin)
    raw_stored = frame['teacher_probability_raw'].to_numpy(dtype=np.float64)
    calibrated = stable_sigmoid(gate.slope * margin + gate.intercept)
    stored = frame['teacher_probability'].to_numpy(dtype=np.float64)
    if not np.isfinite(raw_stored).all() or np.max(np.abs(raw_stored - raw_expected), initial=0) > args.probability_tolerance:
        raise RuntimeError(f'{spec.name}: raw probability does not match its raw margin')
    if not np.isfinite(stored).all() or np.max(np.abs(stored - calibrated), initial=0) > args.probability_tolerance:
        raise RuntimeError(f'{spec.name}: stored probability does not match its own representative gate')
    if 'teacher_calibration_slope' in frame:
        slopes = frame['teacher_calibration_slope'].to_numpy(dtype=np.float64)
        if np.max(np.abs(slopes - gate.slope), initial=0) > args.calibration_tolerance:
            raise RuntimeError(f'{spec.name}: per-row calibration slope is stale')
    if 'teacher_calibration_intercept' in frame:
        offsets = frame['teacher_calibration_intercept'].to_numpy(dtype=np.float64)
        if np.max(np.abs(offsets - gate.intercept), initial=0) > args.calibration_tolerance:
            raise RuntimeError(f'{spec.name}: per-row calibration intercept is stale')
    order_gap = derive_order_probability_gap(frame, gate, args, spec.name)
    out = frame.copy()
    out[f'{spec.name}_margin'] = margin.astype(np.float32)
    out[f'{spec.name}_raw_probability'] = raw_expected.astype(np.float32)
    out[f'{spec.name}_probability'] = calibrated.astype(np.float32)
    out[f'{spec.name}_order_probability_gap'] = order_gap.astype(np.float32)
    return (out, meta)

def source_target_of(frame: pd.DataFrame) -> pd.Series | None:
    for column in ('source_target', 'target', 'label'):
        if column in frame:
            values = pd.to_numeric(frame[column], errors='coerce')
            if values.notna().all() and values.between(0, 1).all():
                return values.astype(np.float32)
    return None

def align_teachers(specs: Sequence[TeacherSpec], gates: Mapping[str, GateInfo], args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, dict]]:
    loaded: list[tuple[TeacherSpec, pd.DataFrame, dict]] = []
    for spec in specs:
        (frame, meta) = validate_and_calibrate_teacher(spec, gates[spec.name], args)
        loaded.append((spec, frame, meta))
    (base_spec, base_frame, _) = loaded[0]
    base_index = pd.MultiIndex.from_frame(base_frame[['_lo', '_hi']])
    diagnostic_columns = [column for column in base_frame.columns if column not in SCORE_COLUMNS and (not column.endswith(('_margin', '_raw_probability', '_probability', '_order_probability_gap'))) and (column not in {'target', 'label', 'source_target'})]
    base = base_frame[diagnostic_columns].copy()
    source = source_target_of(base_frame)
    if source is not None:
        base['source_target'] = source.to_numpy(dtype=np.float32)
    meta_by_teacher: dict[str, dict] = {}
    reference_source = source.to_numpy(dtype=np.float64) if source is not None else None
    for (spec, frame, meta) in loaded:
        index = pd.MultiIndex.from_frame(frame[['_lo', '_hi']])
        missing = base_index.difference(index)
        extra = index.difference(base_index)
        if len(missing) or len(extra):
            raise RuntimeError(f'{spec.name}: candidate key set differs from {base_spec.name}: missing={len(missing):,}, extra={len(extra):,}')
        keyed = frame.set_index(['_lo', '_hi'], drop=False).loc[base_index]
        if spec.name != base_spec.name:
            for column in diagnostic_columns:
                if column not in keyed:
                    raise RuntimeError(f'{spec.name}: shared candidate diagnostic {column!r} is absent')
                left = base[column].reset_index(drop=True)
                right = keyed[column].reset_index(drop=True)
                if not left.equals(right):
                    raise RuntimeError(f'{spec.name}: shared candidate diagnostic {column!r} differs from {base_spec.name}')
        teacher_source = source_target_of(keyed)
        if reference_source is not None and teacher_source is not None:
            if not np.allclose(reference_source, teacher_source.to_numpy(dtype=np.float64), atol=1e-07):
                raise RuntimeError(f'{spec.name}: copied source targets disagree across score files')
        for suffix in ('margin', 'raw_probability', 'probability', 'order_probability_gap'):
            column = f'{spec.name}_{suffix}'
            base[column] = keyed[column].to_numpy()
        meta_by_teacher[spec.name] = meta
    return (base.reset_index(drop=True), meta_by_teacher)

def read_item_subset(path: Path, ids: Iterable[int]) -> pd.DataFrame:
    wanted = {int(value) for value in ids}
    parts: list[pd.DataFrame] = []
    parquet = pq.ParquetFile(path)
    columns = ['id', 'name', 'attributes', 'category']
    for batch in parquet.iter_batches(batch_size=262144, columns=columns):
        frame = batch.to_pandas()
        mask = frame['id'].isin(wanted)
        if mask.any():
            parts.append(frame.loc[mask, columns])
    if not parts:
        return pd.DataFrame(columns=columns)
    result = pd.concat(parts, ignore_index=True)
    duplicate = result[result['id'].duplicated(keep=False)]
    if len(duplicate):
        conflicts = duplicate.groupby('id', observed=True)[['name', 'attributes', 'category']].nunique(dropna=False).gt(1).any(axis=1)
        if conflicts.any():
            raise RuntimeError(f'item catalog has conflicting duplicate IDs: {int(conflicts.sum())}')
    return result.drop_duplicates('id', keep='first')

def attach_fingerprints(frame: pd.DataFrame, items_path: str) -> tuple[pd.DataFrame, bool]:
    has_upstream = {'item_fp1', 'item_fp2'}.issubset(frame.columns)
    if not items_path:
        if has_upstream:
            if frame[['item_fp1', 'item_fp2']].isna().any().any():
                raise RuntimeError('upstream item fingerprints contain nulls')
            return (frame, True)
        return (frame, False)
    path = Path(items_path)
    ids = pd.unique(pd.concat([frame['id1'], frame['id2']])).astype(np.int64)
    cards = read_item_subset(path, ids)
    fingerprints = {int(pid): content_fingerprint(name, attrs, category) for (pid, name, attrs, category) in zip(cards['id'], cards['name'], cards['attributes'], cards['category'])}
    out = frame.copy()
    computed1 = out['id1'].map(fingerprints)
    computed2 = out['id2'].map(fingerprints)
    missing = computed1.isna() | computed2.isna()
    if missing.any():
        raise RuntimeError(f'items parquet misses cards for {int(missing.sum()):,} candidate pairs')
    if has_upstream:
        mismatch = (out['item_fp1'].astype(str).to_numpy() != computed1.astype(str).to_numpy()) | (out['item_fp2'].astype(str).to_numpy() != computed2.astype(str).to_numpy())
        if mismatch.any():
            raise RuntimeError(f'upstream item fingerprints disagree with --items for {int(mismatch.sum()):,} rows')
    out['item_fp1'] = computed1
    out['item_fp2'] = computed2
    return (out, True)

def leakage_mask(frame: pd.DataFrame, args: argparse.Namespace) -> tuple[np.ndarray, list[list[str]], dict]:
    reasons: list[list[str]] = [[] for _ in range(len(frame))]
    mask = np.zeros(len(frame), dtype=bool)
    exact_pairs: set[tuple[int, int]] = set()
    for raw_path in args.exclude_pairs:
        exact_pairs.update(pair_key_set(pd.read_parquet(raw_path, columns=list(PAIR_COLUMNS))))
    if exact_pairs:
        hit = np.fromiter(((int(lo), int(hi)) in exact_pairs for (lo, hi) in zip(frame['_lo'], frame['_hi'])), dtype=bool, count=len(frame))
        for index in np.flatnonzero(hit):
            reasons[int(index)].append('excluded_exact_pair')
        mask |= hit
    forbidden_ids: set[int] = set()
    forbidden_fps: set[str] = set()
    for raw_path in args.exclude_items:
        excluded = pd.read_parquet(raw_path)
        if 'id' in excluded:
            forbidden_ids.update((int(value) for value in excluded['id'].dropna()))
        if 'fp' in excluded:
            forbidden_fps.update((str(value) for value in excluded['fp'].dropna()))
    if forbidden_ids:
        hit = frame['id1'].isin(forbidden_ids).to_numpy() | frame['id2'].isin(forbidden_ids).to_numpy()
        for index in np.flatnonzero(hit):
            reasons[int(index)].append('excluded_item_id')
        mask |= hit
    fingerprint_verified = {'item_fp1', 'item_fp2'}.issubset(frame.columns)
    if forbidden_fps:
        if not fingerprint_verified:
            raise RuntimeError('exclude-items contains content fingerprints, but candidate/item fingerprints cannot be verified; pass --items')
        hit = frame['item_fp1'].astype(str).isin(forbidden_fps).to_numpy() | frame['item_fp2'].astype(str).isin(forbidden_fps).to_numpy()
        for index in np.flatnonzero(hit):
            reasons[int(index)].append('excluded_item_fingerprint')
        mask |= hit
    content_duplicates = np.zeros(len(frame), dtype=bool)
    if args.deduplicate_content_pairs and fingerprint_verified:
        fp1 = frame['item_fp1'].astype(str).to_numpy()
        fp2 = frame['item_fp2'].astype(str).to_numpy()
        clo = np.minimum(fp1, fp2)
        chi = np.maximum(fp1, fp2)
        content_duplicates = pd.DataFrame({'lo': clo, 'hi': chi}).duplicated(['lo', 'hi'], keep='first').to_numpy()
        for index in np.flatnonzero(content_duplicates):
            reasons[int(index)].append('duplicate_content_pair')
        mask |= content_duplicates
    audit = {'excluded_pair_keys': len(exact_pairs), 'forbidden_item_ids': len(forbidden_ids), 'forbidden_item_fingerprints': len(forbidden_fps), 'fingerprints_verified': fingerprint_verified, 'excluded_rows': int(mask.sum()), 'duplicate_content_pair_rows': int(content_duplicates.sum())}
    return (mask, reasons, audit)

def rule_conflict(frame: pd.DataFrame, args: argparse.Namespace) -> tuple[np.ndarray, list[str]]:
    requested = [value.strip() for value in args.rule_conflict_cols.split(',') if value.strip()]
    available = [column for column in requested if column in frame.columns]
    if not available:
        return (np.zeros(len(frame), dtype=np.float64), [])
    matrix = frame[available].to_numpy(dtype=np.float64)
    if not np.isfinite(matrix).all() or np.any((matrix < 0) | (matrix > 1)):
        raise RuntimeError(f'rule-conflict columns must be finite in [0,1]: {available}')
    return (matrix.max(axis=1), available)

def cross_encoder_veto(frame: pd.DataFrame, positive: np.ndarray, negative: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, list[str]]:
    columns = [value.strip() for value in args.cross_encoder_score_cols.split(',') if value.strip()]
    if not columns:
        zeros = np.zeros(len(frame), dtype=bool)
        return (zeros, zeros.copy(), [])
    missing = [column for column in columns if column not in frame]
    if missing:
        raise RuntimeError(f'cross-encoder abstention columns are absent: {missing}')
    if 'category' not in frame or frame['category'].isna().any():
        raise RuntimeError('cross-encoder rank abstention needs complete categories')
    scores = frame[columns].to_numpy(dtype=np.float64)
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise RuntimeError('cross-encoder scores must be finite probabilities in [0,1]')
    stable_columns: list[np.ndarray] = []
    for column in columns:
        required = [f'{column}_ab', f'{column}_ba', f'{column}_order_gap']
        absent = [name for name in required if name not in frame]
        if absent:
            raise RuntimeError(f'{column}: bidirectional CE evidence is required; missing {absent}')
        ab = frame[required[0]].to_numpy(dtype=np.float64)
        ba = frame[required[1]].to_numpy(dtype=np.float64)
        gap = frame[required[2]].to_numpy(dtype=np.float64)
        if not (np.isfinite(ab).all() and np.isfinite(ba).all() and np.isfinite(gap).all()):
            raise RuntimeError(f'{column}: CE orientation audit contains NaN/inf')
        if np.max(np.abs(frame[column].to_numpy(dtype=float) - 0.5 * (ab + ba)), initial=0) > 2e-05:
            raise RuntimeError(f'{column}: mean score does not equal its A/B orientations')
        if np.max(np.abs(gap - np.abs(ab - ba)), initial=0) > 2e-05:
            raise RuntimeError(f'{column}: stored CE order gap is inconsistent')
        stable_columns.append(gap <= args.max_ce_order_gap)
    stable = np.column_stack(stable_columns)
    ranks = np.column_stack([frame.groupby('category', observed=True)[column].rank(pct=True).to_numpy(dtype=float) for column in columns])
    negative_veto = negative & np.all((ranks >= args.ce_negative_veto_quantile) & stable, axis=1)
    positive_veto = positive & np.all((ranks <= args.ce_positive_veto_quantile) & stable, axis=1)
    return (negative_veto, positive_veto, columns)

def compute_consensus(frame: pd.DataFrame, specs: Sequence[TeacherSpec], gates: Mapping[str, GateInfo], args: argparse.Namespace) -> tuple[pd.DataFrame, np.ndarray, list[list[str]], dict]:
    names = [spec.name for spec in specs]
    probabilities = np.column_stack([frame[f'{name}_probability'].to_numpy(dtype=np.float64) for name in names])
    order_gaps = np.column_stack([frame[f'{name}_order_probability_gap'].to_numpy(dtype=np.float64) for name in names])
    if np.any((probabilities < 0) | (probabilities > 1)) or not np.isfinite(probabilities).all():
        raise RuntimeError('calibrated teacher probabilities must be finite in [0,1]')
    if np.isnan(order_gaps).any():
        if not args.allow_missing_order_gap:
            raise RuntimeError('missing order gaps without --allow-missing-order-gap')
        order_gaps = np.nan_to_num(order_gaps, nan=args.max_order_probability_gap)
    evidence = np.asarray([max((gates[spec.name].ap - gates[spec.name].constant_ap) / max(1e-09, 1.0 - gates[spec.name].constant_ap), args.gate_weight_floor) ** args.gate_weight_power * spec.multiplier for spec in specs], dtype=np.float64)
    reliability = evidence / evidence.sum()
    confidence = np.abs(2.0 * probabilities - 1.0)
    order_factor = np.exp(-order_gaps / args.order_scale)
    effective = reliability[None, :] * (args.confidence_floor + confidence) * order_factor
    consensus = np.sum(effective * probabilities, axis=1) / np.sum(effective, axis=1)
    spread = probabilities.max(axis=1) - probabilities.min(axis=1)
    reliability_mean = probabilities @ reliability
    weighted_std = np.sqrt(np.sum(reliability[None, :] * (probabilities - reliability_mean[:, None]) ** 2, axis=1))
    max_order_gap = order_gaps.max(axis=1)
    confident_negative = probabilities <= args.vote_negative_threshold
    confident_positive = probabilities >= args.vote_positive_threshold
    confident_count = np.sum(confident_negative | confident_positive, axis=1)
    conflicting_votes = confident_negative.any(axis=1) & confident_positive.any(axis=1)
    positive = consensus >= args.positive_threshold
    negative = consensus <= args.negative_threshold
    certain = positive | negative
    high_disagreement = (spread > args.max_teacher_spread) | conflicting_votes
    unstable = max_order_gap > args.max_order_probability_gap
    insufficient = confident_count < args.min_confident_teachers
    (conflict_score, rule_columns) = rule_conflict(frame, args)
    rule_veto = positive & (conflict_score >= args.rule_veto_threshold)
    (ce_negative_veto, ce_positive_veto, ce_columns) = cross_encoder_veto(frame, positive, negative, args)
    accepted = certain & ~high_disagreement & ~unstable & ~insufficient & ~rule_veto & ~ce_negative_veto & ~ce_positive_veto
    reasons: list[list[str]] = [[] for _ in range(len(frame))]
    for index in range(len(frame)):
        if not certain[index]:
            reasons[index].append('consensus_uncertain')
        if high_disagreement[index]:
            reasons[index].append('teacher_disagreement')
        if unstable[index]:
            reasons[index].append('order_instability')
        if insufficient[index]:
            reasons[index].append('insufficient_confident_teachers')
        if rule_veto[index]:
            reasons[index].append('rule_positive_veto')
        if ce_negative_veto[index]:
            reasons[index].append('cross_encoder_negative_veto')
        if ce_positive_veto[index]:
            reasons[index].append('cross_encoder_positive_veto')
    certainty = np.abs(2.0 * consensus - 1.0)
    agreement = np.clip(1.0 - spread, 0.0, 1.0)
    order_stability = np.exp(-max_order_gap / args.order_scale)
    weighted_ap = float(sum((reliability[i] * gates[name].ap for (i, name) in enumerate(names))))
    gate_strength = np.clip((weighted_ap - args.min_gate_ap) / max(1e-09, 1.0 - args.min_gate_ap), 0.0, 1.0)
    quality = (0.25 + 0.75 * certainty) * (0.5 + 0.5 * agreement) * order_stability * (0.75 + 0.25 * gate_strength)
    sample_weight = np.clip(args.weight_scale * quality, args.min_sample_weight, args.max_sample_weight)
    out = frame.copy()
    for (i, name) in enumerate(names):
        out[f'{name}_confidence'] = confidence[:, i].astype(np.float32)
        out[f'{name}_gate_ap'] = np.float32(gates[name].ap)
        out[f'{name}_consensus_weight'] = np.float32(reliability[i])
    out['consensus_probability'] = consensus.astype(np.float32)
    out['consensus_label'] = positive.astype(np.int8)
    out['teacher_probability_spread'] = spread.astype(np.float32)
    out['teacher_probability_weighted_std'] = weighted_std.astype(np.float32)
    out['max_order_probability_gap'] = max_order_gap.astype(np.float32)
    out['confident_teacher_count'] = confident_count.astype(np.int16)
    out['rule_conflict_score'] = conflict_score.astype(np.float32)
    out['consensus_quality'] = quality.astype(np.float32)
    out['sample_weight'] = sample_weight.astype(np.float32)
    teacher_audit = {name: {'gate_ap': gates[name].ap, 'gate_constant_ap': gates[name].constant_ap, 'gate_normalized_ap_lift': float((gates[name].ap - gates[name].constant_ap) / max(1e-09, 1.0 - gates[name].constant_ap)), 'manual_multiplier': specs[i].multiplier, 'normalized_consensus_weight': float(reliability[i]), 'mean_probability': float(probabilities[:, i].mean()), 'mean_order_probability_gap': float(order_gaps[:, i].mean())} for (i, name) in enumerate(names)}
    audit = {'teachers': teacher_audit, 'weighted_gate_ap': weighted_ap, 'rule_columns_used': rule_columns, 'cross_encoder_veto_columns': ce_columns, 'cross_encoder_negative_veto_rows': int(ce_negative_veto.sum()), 'cross_encoder_positive_veto_rows': int(ce_positive_veto.sum()), 'raw_consensus_eligible_rows': int(accepted.sum())}
    return (out, accepted, reasons, audit)

def apportion(total: int, categories: Sequence[object], weights: np.ndarray) -> dict[object, int]:
    if total <= 0:
        return {category: 0 for category in categories}
    normalized = weights / weights.sum()
    raw = normalized * total
    floor = np.floor(raw).astype(int)
    remainder = total - int(floor.sum())
    order = sorted(range(len(categories)), key=lambda i: (-(raw[i] - floor[i]), str(categories[i])))
    for index in order[:remainder]:
        floor[index] += 1
    return {category: int(floor[i]) for (i, category) in enumerate(categories)}

def desired_cell_quotas(total: int, categories: Sequence[object], category_weights: np.ndarray, mode: str) -> dict[tuple[object, int], int]:
    category_quota = apportion(total, categories, category_weights)
    result: dict[tuple[object, int], int] = {}
    odd_cells_seen = 0
    for (position, category) in enumerate(categories):
        quota = category_quota[category]
        if mode == 'category-label':
            pos = quota // 2 + int(quota % 2 and odd_cells_seen % 2 == 0)
            if quota % 2:
                odd_cells_seen += 1
            result[category, 1] = pos
            result[category, 0] = quota - pos
        else:
            result[category, -1] = quota
    return result

def select_balanced(frame: pd.DataFrame, eligible: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, dict]:
    pool = frame.loc[eligible].copy()
    if pool.empty:
        raise RuntimeError('no pairs survive teacher consensus gates')
    requested = min(args.output_size or len(pool), len(pool))
    if args.balance_mode == 'none':
        chosen = pool.sort_values(['consensus_quality', '_lo', '_hi'], ascending=[False, True, True], kind='stable').head(requested).index.to_numpy()
        reserve = np.setdiff1d(pool.index.to_numpy(), chosen, assume_unique=True)
        return (chosen, reserve, {'mode': 'none', 'requested': requested, 'selected': len(chosen)})
    if 'category' not in pool or pool['category'].isna().any():
        raise RuntimeError('category/category-label balancing requires a complete category column')
    categories = sorted(pd.unique(frame['category']), key=str)
    pool_categories = set(pd.unique(pool['category']))
    missing_categories = [category for category in categories if category not in pool_categories]
    if missing_categories and (not args.allow_missing_categories):
        raise RuntimeError(f'no accepted teacher pairs for categories: {missing_categories}')
    categories = [category for category in categories if category in pool_categories]
    if args.category_quota == 'uniform':
        category_weights = np.ones(len(categories), dtype=np.float64)
    else:
        source_counts = frame['category'].value_counts()
        category_weights = np.asarray([source_counts[category] for category in categories], dtype=np.float64)
    if args.balance_mode == 'category-label':
        capacities = pool.groupby(['category', 'consensus_label'], observed=True).size().to_dict()
    else:
        capacities = {(category, -1): int((pool['category'] == category).sum()) for category in categories}

    def feasible(total: int) -> tuple[bool, dict[tuple[object, int], int]]:
        quota = desired_cell_quotas(total, categories, category_weights, args.balance_mode)
        return (all((capacities.get(cell, 0) >= want for (cell, want) in quota.items())), quota)
    chosen_total = requested
    (ok, quota) = feasible(chosen_total)
    while not ok and chosen_total > 0:
        ratios = [capacities.get(cell, 0) / want for (cell, want) in quota.items() if want > 0]
        ratio = min(ratios, default=0.0)
        next_total = min(chosen_total - 1, int(math.floor(chosen_total * ratio)))
        chosen_total = max(0, next_total)
        (ok, quota) = feasible(chosen_total)
    if chosen_total < args.min_output:
        limiting = sorted(((str(category), label, int(capacities.get((category, label), 0))) for category in categories for label in ((0, 1) if args.balance_mode == 'category-label' else (-1,))), key=lambda value: value[2])[:10]
        raise RuntimeError(f'balanced accepted output would have {chosen_total:,} rows, below --min-output={args.min_output:,}; limiting cells={limiting}')
    selected_parts: list[np.ndarray] = []
    quota_audit: dict[str, int] = {}
    for ((category, label), want) in quota.items():
        cell = pool[pool['category'] == category]
        if label >= 0:
            cell = cell[cell['consensus_label'] == label]
        cell = cell.sort_values(['consensus_quality', '_lo', '_hi'], ascending=[False, True, True], kind='stable')
        selected_parts.append(cell.head(want).index.to_numpy())
        quota_audit[f'{category}|{label}'] = want
    chosen = np.concatenate(selected_parts) if selected_parts else np.empty(0, dtype=np.int64)
    reserve = np.setdiff1d(pool.index.to_numpy(), chosen, assume_unique=True)
    return (chosen, reserve, {'mode': args.balance_mode, 'category_quota': args.category_quota, 'requested': requested, 'feasible_balanced_total': chosen_total, 'selected': len(chosen), 'cell_quotas': quota_audit})

def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)

def value_counts_json(frame: pd.DataFrame, columns: list[str]) -> dict[str, int]:
    if frame.empty:
        return {}
    counts = frame.groupby(columns, observed=True).size()
    if len(columns) == 1:
        return {str(key): int(value) for (key, value) in counts.items()}
    return {'|'.join(map(str, key)): int(value) for (key, value) in counts.items()}

def write_outputs(frame: pd.DataFrame, selected: np.ndarray, adjudication_reasons: list[list[str]], args: argparse.Namespace, audit: dict) -> dict:
    selected_set = set((int(value) for value in selected))
    accepted = frame.loc[list(selected)].copy()
    accepted['target'] = accepted['consensus_probability'].astype(np.float32)
    accepted['label'] = accepted['consensus_label'].astype(np.int8)
    required = ['id1', 'id2', 'target', 'sample_weight']
    remaining = [column for column in accepted.columns if column not in required + ['_lo', '_hi']]
    accepted = accepted[required + remaining]
    rejected_indices = [index for index in frame.index if int(index) not in selected_set]
    rejected = frame.loc[rejected_indices].copy()
    rejected['adjudication_reason'] = [';'.join(adjudication_reasons[index]) or 'balance_reserve' for index in rejected_indices]
    rejected = rejected.drop(columns=['target', 'label', 'sample_weight'], errors='ignore')
    rejected = rejected.drop(columns=['_lo', '_hi'], errors='ignore')
    output = Path(args.output)
    adjudication_output = Path(args.adjudication_output) if args.adjudication_output else output.with_name(output.stem + '.adjudication.parquet')
    audit_output = Path(args.audit_output) if args.audit_output else Path(str(output) + '.audit.json')
    atomic_parquet(accepted, output)
    atomic_parquet(rejected, adjudication_output)
    raw_reread = pd.read_parquet(output)
    reread = canonicalize(raw_reread, 'accepted read-back')
    if len(reread) != len(accepted):
        raise RuntimeError('accepted parquet read-back row count mismatch')
    targets = reread['target'].to_numpy(dtype=np.float64)
    weights = reread['sample_weight'].to_numpy(dtype=np.float64)
    if not np.isfinite(targets).all() or np.any((targets < 0) | (targets > 1)):
        raise RuntimeError('accepted read-back target invariant failed')
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise RuntimeError('accepted read-back sample_weight invariant failed')
    reason_counts: Counter[str] = Counter()
    for values in adjudication_reasons:
        reason_counts.update(values)
    reason_counts['balance_reserve'] += sum((not adjudication_reasons[index] for index in rejected_indices))
    downstream = audit.get('downstream_contract', {})
    prerequisites_satisfied = downstream.get('satisfied') is True
    downstream.update({'consumer': 'pretrain_ce.py --extra_pairs', 'required_columns': required, 'satisfied': bool(prerequisites_satisfied and set(required).issubset(raw_reread.columns)), 'soft_target': True, 'source_target_is_diagnostic_only': True, 'output_schema': training_contract.output_schema(raw_reread)})
    audit['downstream_contract'] = downstream
    audit.update({'output': str(output), 'adjudication_output': str(adjudication_output), 'accepted_rows': len(accepted), 'adjudication_rows': len(rejected), 'accepted_category_label_counts': value_counts_json(accepted, ['category', 'label']) if 'category' in accepted else {}, 'accepted_positive_fraction': float(accepted['label'].mean()) if len(accepted) else None, 'accepted_mean_target': float(accepted['target'].mean()) if len(accepted) else None, 'accepted_mean_sample_weight': float(accepted['sample_weight'].mean()) if len(accepted) else None, 'adjudication_reason_counts': dict(sorted(reason_counts.items())), 'accepted_unordered_duplicates': int(reread.duplicated(['_lo', '_hi']).sum()), 'source_target_used_for_truth': False, 'source_target_used_for_weight': False, 'rules_can_create_positive': False, 'output_sha256': sha256_file(output), 'adjudication_sha256': sha256_file(adjudication_output)})
    if 'source_target' in accepted and len(accepted):
        source_label = accepted['source_target'].to_numpy(dtype=float) >= 0.5
        audit['source_consensus_label_agreement_diagnostic_only'] = float(np.mean(source_label == accepted['label'].to_numpy(dtype=bool)))
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    temporary_audit = audit_output.with_name(audit_output.name + f'.tmp.{os.getpid()}')
    with temporary_audit.open('w', encoding='utf-8') as stream:
        json.dump(audit, stream, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(temporary_audit, audit_output)
    print(f'accepted {len(accepted):,} -> {output}; adjudication {len(rejected):,} -> {adjudication_output}; audit -> {audit_output}', flush=True)
    return audit

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=__doc__)
    parser.add_argument('--teacher', action='append', default=[], metavar='NAME=PARQUET')
    parser.add_argument('--gate', action='append', default=[], metavar='NAME=JSON')
    parser.add_argument('--teacher-multiplier', action='append', default=[], metavar='NAME=FLOAT')
    parser.add_argument('--output', required=True)
    parser.add_argument('--adjudication-output', default='')
    parser.add_argument('--audit-output', default='')
    parser.add_argument('--items', default='', help='item parquet for fingerprint re-audit')
    parser.add_argument('--exclude-pairs', action='append', default=[])
    parser.add_argument('--exclude-items', action='append', default=[])
    parser.add_argument('--fail-on-leakage', action='store_true')
    parser.add_argument('--deduplicate-content-pairs', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--allow-single-teacher', action='store_true')
    parser.add_argument('--allow-missing-order-gap', action='store_true')
    parser.add_argument('--allow-missing-score-meta', action='store_true')
    parser.add_argument('--min-gate-ap', type=float, default=0.58)
    parser.add_argument('--min-gate-calibration-rows', type=int, default=1000)
    parser.add_argument('--gate-weight-power', type=float, default=1.0)
    parser.add_argument('--gate-weight-floor', type=float, default=0.01)
    parser.add_argument('--probability-tolerance', type=float, default=2e-05)
    parser.add_argument('--calibration-tolerance', type=float, default=1e-07)
    parser.add_argument('--negative-threshold', type=float, default=0.28)
    parser.add_argument('--positive-threshold', type=float, default=0.72)
    parser.add_argument('--vote-negative-threshold', type=float, default=0.28)
    parser.add_argument('--vote-positive-threshold', type=float, default=0.72)
    parser.add_argument('--min-confident-teachers', type=int, default=2)
    parser.add_argument('--max-teacher-spread', type=float, default=0.25)
    parser.add_argument('--max-order-probability-gap', type=float, default=0.15)
    parser.add_argument('--order-scale', type=float, default=0.1)
    parser.add_argument('--confidence-floor', type=float, default=0.05)
    parser.add_argument('--rule-conflict-cols', default=','.join(DEFAULT_RULE_COLUMNS))
    parser.add_argument('--rule-veto-threshold', type=float, default=0.5)
    parser.add_argument('--cross-encoder-score-cols', default='', help='comma-separated CE probabilities used only to abstain on extreme disagreement')
    parser.add_argument('--ce-negative-veto-quantile', type=float, default=0.95)
    parser.add_argument('--ce-positive-veto-quantile', type=float, default=0.05)
    parser.add_argument('--max-ce-order-gap', type=float, default=0.15)
    parser.add_argument('--output-size', type=int, default=0, help='0 means as many as feasible')
    parser.add_argument('--min-output', type=int, default=10000)
    parser.add_argument('--balance-mode', choices=('category-label', 'category', 'none'), default='category', help='category preserves the natural accepted-label mix; category-label is an ablation')
    parser.add_argument('--category-quota', choices=('input', 'uniform'), default='input')
    parser.add_argument('--allow-missing-categories', action='store_true')
    parser.add_argument('--weight-scale', type=float, default=1.0)
    parser.add_argument('--min-sample-weight', type=float, default=0.2)
    parser.add_argument('--max-sample-weight', type=float, default=1.5)
    return parser

def validate_args(args: argparse.Namespace) -> None:
    probabilities = (args.negative_threshold, args.positive_threshold, args.vote_negative_threshold, args.vote_positive_threshold, args.max_teacher_spread, args.max_order_probability_gap, args.rule_veto_threshold, args.ce_negative_veto_quantile, args.ce_positive_veto_quantile, args.max_ce_order_gap)
    if any((not 0 <= value <= 1 for value in probabilities)):
        raise ValueError('all probability thresholds must be in [0,1]')
    if args.negative_threshold >= 0.5 or args.positive_threshold <= 0.5:
        raise ValueError('accept thresholds must straddle 0.5')
    if args.vote_negative_threshold >= 0.5 or args.vote_positive_threshold <= 0.5:
        raise ValueError('vote thresholds must straddle 0.5')
    if not 0 <= args.ce_positive_veto_quantile < args.ce_negative_veto_quantile <= 1:
        raise ValueError('cross-encoder veto quantiles are invalid')
    if args.order_scale <= 0 or not 0 < args.confidence_floor <= 1:
        raise ValueError('--order-scale must be positive and --confidence-floor in (0,1]')
    if args.min_confident_teachers <= 0:
        raise ValueError('--min-confident-teachers must be positive')
    if args.output_size < 0 or args.min_output < 0:
        raise ValueError('output sizes must be non-negative')
    if args.output_size and args.output_size < args.min_output:
        raise ValueError('--output-size cannot be below --min-output')
    if args.max_sample_weight < args.min_sample_weight or args.min_sample_weight <= 0:
        raise ValueError('invalid sample-weight bounds')
    if not PRODUCTION_MIN_GATE_AP <= args.min_gate_ap <= 1.0:
        raise ValueError('invalid gate quality requirements')
    if args.min_gate_calibration_rows < PRODUCTION_MIN_CALIBRATION_ROWS:
        raise ValueError(f'--min-gate-calibration-rows cannot be below {PRODUCTION_MIN_CALIBRATION_ROWS}')
    if args.vote_negative_threshold > args.negative_threshold or args.vote_positive_threshold < args.positive_threshold:
        raise ValueError('confident-vote thresholds cannot extend outside the human-validated acceptance region')
    if args.gate_weight_power <= 0 or args.gate_weight_floor <= 0:
        raise ValueError('gate weight power/floor must be positive')
    if args.probability_tolerance <= 0 or args.calibration_tolerance <= 0:
        raise ValueError('calibration/probability tolerances must be positive')

def main(argv: Sequence[str] | None=None) -> dict:
    args = build_parser().parse_args(argv)
    validate_args(args)
    specs = build_specs(args)
    if args.min_confident_teachers > len(specs):
        raise ValueError(f'--min-confident-teachers={args.min_confident_teachers} exceeds the {len(specs)} supplied teachers')
    gates = {spec.name: load_gate(spec, args) for spec in specs}
    if len({gate.model for gate in gates.values()}) != len(gates):
        raise RuntimeError('teacher gates resolve to duplicate model identities')
    priors = {round(gate.reference_positive_prior, 12) for gate in gates.values()}
    if len(priors) != 1:
        raise RuntimeError('teacher calibrated evidence uses incompatible reference priors')
    (frame, score_metadata) = align_teachers(specs, gates, args)
    probability_vectors = [frame[f'{spec.name}_probability'].to_numpy(dtype=np.float64) for spec in specs]
    for left in range(len(probability_vectors)):
        for right in range(left + 1, len(probability_vectors)):
            if np.array_equal(probability_vectors[left], probability_vectors[right]):
                raise RuntimeError(f'teacher evidence is duplicated exactly: {specs[left].name} and {specs[right].name}')
    (frame, fingerprints_verified) = attach_fingerprints(frame, args.items)
    (leak, leak_reasons, leak_audit) = leakage_mask(frame, args)
    if leak.any() and args.fail_on_leakage:
        raise RuntimeError(f'leakage re-audit found {int(leak.sum()):,} forbidden rows')
    (frame, eligible, reasons, consensus_audit) = compute_consensus(frame, specs, gates, args)
    eligible &= ~leak
    for index in range(len(frame)):
        reasons[index].extend(leak_reasons[index])
    (selected, reserve, balance_audit) = select_balanced(frame, eligible, args)
    for index in reserve:
        reasons[int(index)].append('balance_reserve')
    input_audit = {spec.name: {'score_path': str(spec.score_path), 'score_sha256': sha256_file(spec.score_path), 'score_metadata': score_metadata[spec.name], 'gate_path': str(spec.gate_path), 'gate_sha256': sha256_file(spec.gate_path), 'model': gates[spec.name].model, 'calibration': {'kind': 'global_platt_category_class_balanced_prior_explicit_v2', 'fit_role': 'calibration', 'fit_stratum': 'representative', 'rows': gates[spec.name].calibration_rows, 'slope': gates[spec.name].slope, 'intercept': gates[spec.name].intercept, 'reference_positive_prior': gates[spec.name].reference_positive_prior}, 'hard_negative_audit': gates[spec.name].report['hard_negative_audit']} for spec in specs}
    exclusion_evidence = training_contract.describe_exclusions(args.exclude_pairs, args.exclude_items)
    score_hashes = [record['score_sha256'] for record in input_audit.values()]
    gate_hashes = [record['gate_sha256'] for record in input_audit.values()]
    independent_teachers = len(specs) >= 2 and len({gate.model for gate in gates.values()}) == len(specs) and (len(set(score_hashes)) == len(specs)) and (len(set(gate_hashes)) == len(specs))
    complete_exclusions = bool(exclusion_evidence.pair_keys) and bool(exclusion_evidence.item_ids) and bool(exclusion_evidence.item_fingerprints)
    fingerprints_recomputed = bool(args.items) and fingerprints_verified
    no_weakening_flags = not args.allow_single_teacher and (not args.allow_missing_order_gap) and (not args.allow_missing_score_meta) and args.deduplicate_content_pairs
    downstream_satisfied = independent_teachers and args.min_confident_teachers >= 2 and complete_exclusions and fingerprints_recomputed and no_weakening_flags and (args.balance_mode == 'category')
    audit = {'kind': 'multi_teacher_consensus_v2', 'producer': {'script': str(Path(__file__).resolve()), 'script_sha256': sha256_file(Path(__file__).resolve())}, 'input_rows': len(frame), 'teacher_inputs': input_audit, 'exclusion_inputs': exclusion_evidence.description, 'thresholds': {'negative': args.negative_threshold, 'positive': args.positive_threshold, 'vote_negative': args.vote_negative_threshold, 'vote_positive': args.vote_positive_threshold, 'min_confident_teachers': args.min_confident_teachers, 'max_teacher_spread': args.max_teacher_spread, 'max_order_probability_gap': args.max_order_probability_gap, 'order_scale': args.order_scale, 'confidence_floor': args.confidence_floor, 'min_gate_ap': args.min_gate_ap, 'rule_veto_threshold': args.rule_veto_threshold, 'cross_encoder_score_cols': args.cross_encoder_score_cols, 'ce_negative_veto_quantile': args.ce_negative_veto_quantile, 'ce_positive_veto_quantile': args.ce_positive_veto_quantile, 'max_ce_order_gap': args.max_ce_order_gap}, 'fingerprints_available': fingerprints_verified, 'fingerprints_recomputed_from_items': fingerprints_recomputed, 'leakage_reaudit': leak_audit, 'consensus': consensus_audit, 'balance': balance_audit, 'weighting': {'scale': args.weight_scale, 'minimum': args.min_sample_weight, 'maximum': args.max_sample_weight}, 'downstream_contract': {'version': training_contract.CONTRACT_VERSION, 'consumer': 'pretrain_ce.py --extra_pairs', 'required_columns': list(training_contract.REQUIRED_COLUMNS), 'soft_target': True, 'source_target_is_diagnostic_only': True, 'fingerprints_recomputed_from_items': fingerprints_recomputed, 'independent_teacher_count': len(specs), 'pair_entity_fingerprint_exclusions_complete': complete_exclusions, 'no_weakening_flags': no_weakening_flags, 'satisfied': downstream_satisfied}}
    return write_outputs(frame, selected, reasons, args, audit)
if __name__ == '__main__':
    main()
