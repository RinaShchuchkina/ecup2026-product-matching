from __future__ import annotations
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence
import numpy as np
import pandas as pd
AUDIT_KIND = 'multi_teacher_consensus_v2'
CONTRACT_VERSION = 'consensus_training_contract_v1'
REQUIRED_COLUMNS = ('id1', 'id2', 'target', 'sample_weight')
STREAM_NAMES = ('base', 'consensus', 'human')
HEX_SHA256 = re.compile('^[0-9a-f]{64}$')

def sha256_file(path: str | Path, block_size: int=1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda : stream.read(block_size), b''):
            digest.update(block)
    return digest.hexdigest()

def _unique_columns(frame: pd.DataFrame, source: str) -> None:
    if not frame.columns.is_unique:
        duplicate = frame.columns[frame.columns.duplicated()].tolist()
        raise RuntimeError(f'{source}: duplicate columns are forbidden: {duplicate}')

def integral_ids(series: pd.Series, source: str) -> np.ndarray:
    if series.isna().any():
        raise RuntimeError(f'{source}: null IDs are forbidden')
    try:
        values = series.to_numpy(dtype=np.int64, copy=True)
        numeric = pd.to_numeric(series, errors='coerce').to_numpy(dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(f'{source}: IDs must be exact int64 values') from exc
    if not np.isfinite(numeric).all() or not np.array_equal(numeric, values.astype(np.float64)):
        raise RuntimeError(f'{source}: IDs must be exact int64 values')
    return values

def canonical_pair_keys(frame: pd.DataFrame, source: str) -> tuple[np.ndarray, np.ndarray]:
    _unique_columns(frame, source)
    missing = {'id1', 'id2'} - set(frame.columns)
    if missing:
        raise RuntimeError(f'{source}: missing pair columns {sorted(missing)}')
    left = integral_ids(frame['id1'], f'{source}:id1')
    right = integral_ids(frame['id2'], f'{source}:id2')
    if np.any(left == right):
        raise RuntimeError(f'{source}: self-pairs are forbidden')
    return (np.minimum(left, right), np.maximum(left, right))

def pair_entity_fingerprint_exclusion_mask(frame: pd.DataFrame, *, pair_keys: Iterable[tuple[int, int]], item_ids: Iterable[int], item_fingerprint_by_id: Mapping[int, str], forbidden_fingerprints: Iterable[str], source: str) -> tuple[np.ndarray, dict]:
    (lo, hi) = canonical_pair_keys(frame, source)
    left = integral_ids(frame['id1'], f'{source}:id1')
    right = integral_ids(frame['id2'], f'{source}:id2')
    canonical_exclusions = {(min(int(first), int(second)), max(int(first), int(second))) for (first, second) in pair_keys}
    forbidden_ids = {int(value) for value in item_ids}
    forbidden_fps = {str(value) for value in forbidden_fingerprints}
    exact = np.fromiter(((int(first), int(second)) in canonical_exclusions for (first, second) in zip(lo, hi)), dtype=bool, count=len(frame))
    _fid = set(forbidden_ids)
    entity = np.fromiter((x in _fid for x in left), bool, len(left)) | np.fromiter((x in _fid for x in right), bool, len(right))
    try:
        fp1 = np.asarray([str(item_fingerprint_by_id[int(value)]) for value in left], dtype=object)
        fp2 = np.asarray([str(item_fingerprint_by_id[int(value)]) for value in right], dtype=object)
    except KeyError as exc:
        raise RuntimeError(f'{source}: item catalog misses fingerprint for ID {exc.args[0]}') from exc
    _ffp = set(forbidden_fps)
    fingerprint = np.fromiter((x in _ffp for x in fp1), bool, len(fp1)) | np.fromiter((x in _ffp for x in fp2), bool, len(fp2))
    union = exact | entity | fingerprint
    return (union, {'rows': int(len(frame)), 'excluded_rows': int(union.sum()), 'exact_pair_rows': int(exact.sum()), 'entity_id_rows': int(entity.sum()), 'content_fingerprint_rows': int(fingerprint.sum())})

def _path_record(path: Path, **extra: object) -> dict:
    return {'path': str(path.resolve()), 'sha256': sha256_file(path), **extra}

@dataclass(frozen=True)
class ExclusionEvidence:
    description: dict
    pair_keys: frozenset[tuple[int, int]]
    item_ids: frozenset[int]
    item_fingerprints: frozenset[str]

def describe_exclusions(pair_paths: Sequence[str | Path], item_paths: Sequence[str | Path]) -> ExclusionEvidence:
    pair_keys: set[tuple[int, int]] = set()
    item_ids: set[int] = set()
    item_fingerprints: set[str] = set()
    pair_files: list[dict] = []
    item_files: list[dict] = []
    for raw_path in pair_paths:
        path = Path(raw_path).resolve()
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f'pair exclusion is absent or empty: {path}')
        frame = pd.read_parquet(path)
        (lo, hi) = canonical_pair_keys(frame, f'pair exclusion {path}')
        keys = set(zip(lo.astype(int), hi.astype(int)))
        if not keys:
            raise RuntimeError(f'pair exclusion has no pair keys: {path}')
        pair_keys.update(keys)
        pair_files.append(_path_record(path, rows=int(len(frame)), unordered_keys=int(len(keys))))
    for raw_path in item_paths:
        path = Path(raw_path).resolve()
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f'item exclusion is absent or empty: {path}')
        frame = pd.read_parquet(path)
        _unique_columns(frame, f'item exclusion {path}')
        if not {'id', 'fp'}.issubset(frame.columns):
            raise RuntimeError(f'item exclusion must contain both id and fp: {path}')
        ids = integral_ids(frame['id'], f'item exclusion {path}:id')
        if frame['fp'].isna().any():
            raise RuntimeError(f'item exclusion {path}: null fingerprints are forbidden')
        fps = frame['fp'].astype(str)
        if fps.str.strip().eq('').any():
            raise RuntimeError(f'item exclusion {path}: empty fingerprints are forbidden')
        current_ids = set((int(value) for value in ids))
        current_fps = set(fps)
        if not current_ids or not current_fps:
            raise RuntimeError(f'item exclusion has no entity/fingerprint evidence: {path}')
        item_ids.update(current_ids)
        item_fingerprints.update(current_fps)
        item_files.append(_path_record(path, rows=int(len(frame)), unique_ids=int(len(current_ids)), unique_fingerprints=int(len(current_fps))))
    description = {'pair_files': pair_files, 'item_files': item_files, 'pair_keys': int(len(pair_keys)), 'item_ids': int(len(item_ids)), 'item_fingerprints': int(len(item_fingerprints))}
    return ExclusionEvidence(description, frozenset(pair_keys), frozenset(item_ids), frozenset(item_fingerprints))

def _portable_exclusion_description(description: Mapping[str, object]) -> dict:
    result = {'pair_keys': description.get('pair_keys'), 'item_ids': description.get('item_ids'), 'item_fingerprints': description.get('item_fingerprints')}
    for key in ('pair_files', 'item_files'):
        values = description.get(key, [])
        if not isinstance(values, list):
            raise RuntimeError(f'audit exclusion field {key} must be a list')
        result[key] = sorted(({field: value for (field, value) in record.items() if field != 'path'} for record in values), key=lambda record: (str(record.get('sha256')), int(record.get('rows', -1))))
    return result

def value_counts_json(frame: pd.DataFrame, columns: list[str]) -> dict[str, int]:
    if frame.empty:
        return {}
    counts = frame.groupby(columns, observed=True).size()
    if len(columns) == 1:
        return {str(key): int(value) for (key, value) in counts.items()}
    return {'|'.join(map(str, key)): int(value) for (key, value) in counts.items()}

def output_schema(frame: pd.DataFrame) -> dict:
    return {'columns': [str(column) for column in frame.columns], 'dtypes': {str(column): str(dtype) for (column, dtype) in frame.dtypes.items()}}

def normalize_stream_weights(weights: np.ndarray, counts: Mapping[str, int], fractions: Mapping[str, float]) -> tuple[np.ndarray, dict]:
    if tuple(counts) != STREAM_NAMES or tuple(fractions) != STREAM_NAMES:
        raise ValueError(f'stream counts/fractions must be ordered exactly as {STREAM_NAMES}')
    try:
        stream_counts = {name: int(counts[name]) for name in STREAM_NAMES}
        stream_fractions = {name: float(fractions[name]) for name in STREAM_NAMES}
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('stream counts/fractions must be numeric') from exc
    if any((stream_counts[name] <= 0 for name in STREAM_NAMES)):
        raise ValueError(f'verified training requires all three non-empty streams: {stream_counts}')
    if sum(stream_counts.values()) != len(weights):
        raise ValueError(f'stream counts sum to {sum(stream_counts.values())}, weights contain {len(weights)} rows')
    fraction_values = np.asarray([stream_fractions[name] for name in STREAM_NAMES], dtype=np.float64)
    if not np.isfinite(fraction_values).all() or np.any(fraction_values <= 0):
        raise ValueError('verified stream fractions must be finite and strictly positive')
    if not math.isclose(float(fraction_values.sum()), 1.0, rel_tol=0, abs_tol=1e-09):
        raise ValueError(f'verified stream fractions must sum to 1, got {float(fraction_values.sum()):.12g}')
    original = np.asarray(weights)
    normalized = original.astype(np.float64, copy=True)
    if not np.isfinite(normalized).all() or np.any(normalized <= 0):
        raise ValueError('all pre-normalization weights must be finite and strictly positive')
    total_mass = float(len(normalized))
    audit = {'method': 'post_purge_contiguous_stream_mass_v1', 'total_rows': int(len(normalized)), 'target_total_mass': total_mass, 'streams': {}}
    start = 0
    for name in STREAM_NAMES:
        stop = start + stream_counts[name]
        raw_mass = float(normalized[start:stop].sum())
        if not math.isfinite(raw_mass) or raw_mass <= 0:
            raise ValueError(f'stream {name!r} has invalid raw weight mass {raw_mass!r}')
        target_mass = total_mass * stream_fractions[name]
        scale = target_mass / raw_mass
        normalized[start:stop] *= scale
        final_mass = float(normalized[start:stop].sum())
        audit['streams'][name] = {'rows': stream_counts[name], 'target_fraction': stream_fractions[name], 'raw_mass': raw_mass, 'scale': scale, 'normalized_mass': final_mass, 'normalized_fraction': final_mass / total_mass}
        start = stop
    if not math.isclose(float(normalized.sum()), total_mass, rel_tol=0, abs_tol=2e-07 * total_mass):
        raise RuntimeError('stream normalization failed total-mass invariant')
    for name in STREAM_NAMES:
        actual = audit['streams'][name]['normalized_fraction']
        if not math.isclose(actual, stream_fractions[name], rel_tol=0, abs_tol=2e-07):
            raise RuntimeError(f'stream normalization failed fraction invariant for {name}')
    return (normalized.astype(original.dtype, copy=False), audit)

@dataclass(frozen=True)
class VerifiedConsensus:
    frame: pd.DataFrame
    audit: dict
    exclusions: ExclusionEvidence

def _finite_probability(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame:
        raise RuntimeError(f'consensus parquet misses provenance column {column!r}')
    values = pd.to_numeric(frame[column], errors='coerce').to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise RuntimeError(f'{column}: expected finite probabilities in [0,1]')
    return values

def _audit_float(mapping: Mapping[str, object], key: str) -> float:
    try:
        value = float(mapping[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f'audit field {key!r} is absent or non-numeric') from exc
    if not math.isfinite(value):
        raise RuntimeError(f'audit field {key!r} is non-finite')
    return value

def _close(actual: np.ndarray, expected: np.ndarray, name: str, tolerance: float=3e-05) -> None:
    gap = float(np.max(np.abs(actual - expected), initial=0.0))
    if gap > tolerance:
        raise RuntimeError(f'consensus {name} fails recomputation; max gap={gap:.8g}')

def verify_consensus_artifact(parquet_path: str | Path, audit_path: str | Path, *, exclude_pair_paths: Sequence[str | Path], exclude_item_paths: Sequence[str | Path], producer_script_path: str | Path) -> VerifiedConsensus:
    (parquet_path, audit_path) = (Path(parquet_path).resolve(), Path(audit_path).resolve())
    if not parquet_path.is_file() or parquet_path.stat().st_size <= 0:
        raise RuntimeError(f'consensus parquet is absent or empty: {parquet_path}')
    if not audit_path.is_file() or audit_path.stat().st_size <= 0:
        raise RuntimeError(f'consensus audit is absent or empty: {audit_path}')
    with audit_path.open(encoding='utf-8') as stream:
        audit = json.load(stream)
    if not isinstance(audit, dict):
        raise RuntimeError('consensus audit root must be an object')
    audited_output = audit.get('output')
    if not isinstance(audited_output, str) or Path(audited_output).resolve() != parquet_path:
        raise RuntimeError('consensus parquet path differs from its producer audit')
    top_invariants = {'kind': AUDIT_KIND, 'source_target_used_for_truth': False, 'source_target_used_for_weight': False, 'rules_can_create_positive': False}
    bad = {key: audit.get(key) for (key, expected) in top_invariants.items() if audit.get(key) != expected}
    if bad:
        raise RuntimeError(f'invalid consensus audit invariants: {bad}')
    contract = audit.get('downstream_contract')
    if not isinstance(contract, dict):
        raise RuntimeError('consensus audit lacks downstream_contract')
    for (key, expected) in {'version': CONTRACT_VERSION, 'consumer': 'pretrain_ce.py --extra_pairs', 'required_columns': list(REQUIRED_COLUMNS), 'soft_target': True, 'satisfied': True, 'source_target_is_diagnostic_only': True, 'fingerprints_recomputed_from_items': True}.items():
        if contract.get(key) != expected:
            raise RuntimeError(f'downstream contract {key}={contract.get(key)!r}, expected {expected!r}')
    producer = audit.get('producer', {})
    expected_producer_sha = sha256_file(producer_script_path)
    if producer.get('script_sha256') != expected_producer_sha:
        raise RuntimeError('consensus audit was produced by a different teacher_consensus.py')
    first_hash = sha256_file(parquet_path)
    if not HEX_SHA256.fullmatch(str(audit.get('output_sha256', ''))):
        raise RuntimeError('consensus audit has no valid output SHA-256')
    if first_hash != audit['output_sha256']:
        raise RuntimeError('consensus parquet checksum differs from its audit')
    frame = pd.read_parquet(parquet_path)
    if sha256_file(parquet_path) != first_hash:
        raise RuntimeError('consensus parquet changed while it was being verified')
    _unique_columns(frame, 'consensus parquet')
    required = set(REQUIRED_COLUMNS)
    if (missing := (required - set(frame.columns))):
        raise RuntimeError(f'consensus parquet misses required columns {sorted(missing)}')
    if len(frame) != int(audit.get('accepted_rows', -1)) or frame.empty:
        raise RuntimeError('consensus accepted row count disagrees with audit or is zero')
    if output_schema(frame) != contract.get('output_schema'):
        raise RuntimeError('consensus parquet schema/dtypes differ from producer audit')
    (lo, hi) = canonical_pair_keys(frame, 'consensus parquet')
    if pd.MultiIndex.from_arrays([lo, hi]).duplicated().any():
        raise RuntimeError('consensus parquet contains duplicate unordered pairs')
    targets = _finite_probability(frame, 'target')
    weights = pd.to_numeric(frame['sample_weight'], errors='coerce').to_numpy(dtype=np.float64)
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise RuntimeError('consensus sample_weight must be finite and strictly positive')
    if 'consensus_probability' not in frame:
        raise RuntimeError('consensus parquet lacks consensus_probability provenance')
    _close(targets, _finite_probability(frame, 'consensus_probability'), 'target')
    if 'label' not in frame or 'consensus_label' not in frame:
        raise RuntimeError('consensus parquet lacks hard-label provenance')
    labels = pd.to_numeric(frame['label'], errors='coerce').to_numpy(dtype=np.float64)
    consensus_labels = pd.to_numeric(frame['consensus_label'], errors='coerce').to_numpy(dtype=np.float64)
    expected_labels = (targets >= 0.5).astype(np.float64)
    if not (np.array_equal(labels, expected_labels) and np.array_equal(consensus_labels, expected_labels)):
        raise RuntimeError('consensus label columns disagree with soft target')
    teacher_inputs = audit.get('teacher_inputs')
    teacher_summary = audit.get('consensus', {}).get('teachers')
    if not isinstance(teacher_inputs, dict) or not isinstance(teacher_summary, dict):
        raise RuntimeError('consensus audit lacks teacher provenance')
    names = sorted(teacher_inputs)
    if len(names) < 2 or set(names) != set(teacher_summary):
        raise RuntimeError('at least two matching teacher provenance records are required')
    for (key, expected) in {'independent_teacher_count': len(names), 'pair_entity_fingerprint_exclusions_complete': True, 'no_weakening_flags': True}.items():
        if contract.get(key) != expected:
            raise RuntimeError(f'downstream contract {key}={contract.get(key)!r}, expected {expected!r}')
    if audit.get('fingerprints_available') is not True:
        raise RuntimeError('producer did not verify item fingerprints')
    balance = audit.get('balance', {})
    if not isinstance(balance, dict) or balance.get('mode') != 'category':
        raise RuntimeError('verified consensus requires category-conditional balancing')
    models = [str(teacher_inputs[name].get('model', '')) for name in names]
    score_hashes = [str(teacher_inputs[name].get('score_sha256', '')) for name in names]
    gate_hashes = [str(teacher_inputs[name].get('gate_sha256', '')) for name in names]
    if any((not model for model in models)) or len(set(models)) != len(models):
        raise RuntimeError('teacher model identities are missing or not independent')
    for (label, values) in (('score', score_hashes), ('gate', gate_hashes)):
        if any((not HEX_SHA256.fullmatch(value) for value in values)) or len(set(values)) != len(values):
            raise RuntimeError(f'teacher {label} artifacts are missing or not independent')
    for name in names:
        record = teacher_inputs[name]
        for artifact in ('score', 'gate'):
            raw_path = record.get(f'{artifact}_path')
            if not isinstance(raw_path, str) or not raw_path:
                raise RuntimeError(f'teacher {name} has no bound {artifact} path')
            path = Path(raw_path).resolve()
            if not path.is_file() or path.stat().st_size <= 0:
                raise RuntimeError(f'teacher {name} bound {artifact} artifact is absent: {path}')
            if sha256_file(path) != record[f'{artifact}_sha256']:
                raise RuntimeError(f'teacher {name} bound {artifact} artifact changed')
        try:
            with Path(record['gate_path']).open(encoding='utf-8') as stream:
                gate = json.load(stream)
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f'teacher {name} bound gate JSON is unreadable') from exc
        if not isinstance(gate, dict) or gate.get('passed') is not True:
            raise RuntimeError(f'teacher {name} bound gate no longer passes')
        if str(gate.get('model', '')) != record['model']:
            raise RuntimeError(f'teacher {name} model identity differs from its bound gate')
    thresholds = audit.get('thresholds', {})
    minimum = int(thresholds.get('min_confident_teachers', 0))
    if minimum < 2 or minimum > len(names):
        raise RuntimeError('consensus audit weakens the minimum independent-teacher count')
    negative_threshold = _audit_float(thresholds, 'negative')
    positive_threshold = _audit_float(thresholds, 'positive')
    vote_negative = _audit_float(thresholds, 'vote_negative')
    vote_positive = _audit_float(thresholds, 'vote_positive')
    max_spread = _audit_float(thresholds, 'max_teacher_spread')
    max_order_gap = _audit_float(thresholds, 'max_order_probability_gap')
    order_scale = _audit_float(thresholds, 'order_scale')
    confidence_floor = _audit_float(thresholds, 'confidence_floor')
    min_gate_ap = _audit_float(thresholds, 'min_gate_ap')
    if not 0 <= negative_threshold < 0.5 < positive_threshold <= 1:
        raise RuntimeError('consensus acceptance thresholds are invalid')
    probabilities = np.column_stack([_finite_probability(frame, f'{name}_probability') for name in names])
    order_gaps = np.column_stack([_finite_probability(frame, f'{name}_order_probability_gap') for name in names])
    if any((np.array_equal(probabilities[:, i], probabilities[:, j]) for i in range(len(names)) for j in range(i + 1, len(names)))):
        raise RuntimeError('accepted artifact contains duplicated teacher probability vectors')
    reliability = np.asarray([_audit_float(teacher_summary[name], 'normalized_consensus_weight') for name in names])
    if np.any(reliability <= 0) or not math.isclose(float(reliability.sum()), 1.0, abs_tol=2e-06):
        raise RuntimeError('teacher consensus weights are invalid')
    for (index, name) in enumerate(names):
        column = f'{name}_consensus_weight'
        stored = pd.to_numeric(frame[column], errors='coerce').to_numpy(dtype=np.float64)
        if not np.isfinite(stored).all() or np.max(np.abs(stored - reliability[index]), initial=0) > 2e-06:
            raise RuntimeError(f'{column} differs from audit reliability')
    confidence = np.abs(2.0 * probabilities - 1.0)
    effective = reliability[None, :] * (confidence_floor + confidence) * np.exp(-order_gaps / order_scale)
    recomputed_target = np.sum(effective * probabilities, axis=1) / np.sum(effective, axis=1)
    _close(targets, recomputed_target, 'probability')
    spread = probabilities.max(axis=1) - probabilities.min(axis=1)
    stored_spread = pd.to_numeric(frame['teacher_probability_spread'], errors='coerce').to_numpy(dtype=float)
    _close(stored_spread, spread, 'teacher spread')
    row_order_gap = order_gaps.max(axis=1)
    stored_order_gap = pd.to_numeric(frame['max_order_probability_gap'], errors='coerce').to_numpy(dtype=float)
    _close(stored_order_gap, row_order_gap, 'order gap')
    confident_negative = probabilities <= vote_negative
    confident_positive = probabilities >= vote_positive
    confident_count = np.sum(confident_negative | confident_positive, axis=1)
    stored_count = pd.to_numeric(frame['confident_teacher_count'], errors='coerce').to_numpy(dtype=float)
    if not np.array_equal(stored_count, confident_count.astype(float)) or np.any(confident_count < minimum):
        raise RuntimeError('accepted rows do not satisfy independent confident-teacher count')
    if np.any(confident_negative.any(axis=1) & confident_positive.any(axis=1)):
        raise RuntimeError('accepted rows contain conflicting teacher votes')
    if np.any(spread > max_spread + 2e-06) or np.any(row_order_gap > max_order_gap + 2e-06):
        raise RuntimeError('accepted rows violate spread/order abstention')
    if np.any((targets > negative_threshold) & (targets < positive_threshold)):
        raise RuntimeError('accepted rows violate calibrated acceptance thresholds')
    rule_threshold = _audit_float(thresholds, 'rule_veto_threshold')
    rule_score = _finite_probability(frame, 'rule_conflict_score')
    if np.any((targets >= positive_threshold) & (rule_score >= rule_threshold)):
        raise RuntimeError('accepted positive rows violate rule-conflict abstention')
    weighting = audit.get('weighting', {})
    weight_scale = _audit_float(weighting, 'scale')
    min_weight = _audit_float(weighting, 'minimum')
    max_weight = _audit_float(weighting, 'maximum')
    weighted_ap = _audit_float(audit.get('consensus', {}), 'weighted_gate_ap')
    certainty = np.abs(2.0 * recomputed_target - 1.0)
    agreement = np.clip(1.0 - spread, 0.0, 1.0)
    order_stability = np.exp(-row_order_gap / order_scale)
    gate_strength = np.clip((weighted_ap - min_gate_ap) / max(1e-09, 1.0 - min_gate_ap), 0.0, 1.0)
    quality = (0.25 + 0.75 * certainty) * (0.5 + 0.5 * agreement) * order_stability * (0.75 + 0.25 * gate_strength)
    stored_quality = pd.to_numeric(frame['consensus_quality'], errors='coerce').to_numpy(dtype=float)
    _close(stored_quality, quality, 'quality')
    recomputed_weight = np.clip(weight_scale * quality, min_weight, max_weight)
    _close(weights, recomputed_weight, 'sample weight')
    if not math.isclose(float(targets.mean()), float(audit.get('accepted_mean_target')), abs_tol=2e-07):
        raise RuntimeError('accepted mean target differs from audit')
    if not math.isclose(float(weights.mean()), float(audit.get('accepted_mean_sample_weight')), abs_tol=2e-07):
        raise RuntimeError('accepted mean sample_weight differs from audit')
    if 'category' not in frame or frame['category'].isna().any():
        raise RuntimeError('consensus parquet requires complete categories')
    if value_counts_json(frame, ['category', 'label']) != audit.get('accepted_category_label_counts'):
        raise RuntimeError('accepted category/label counts differ from audit')
    exclusions = describe_exclusions(exclude_pair_paths, exclude_item_paths)
    audited_exclusions = audit.get('exclusion_inputs')
    if not isinstance(audited_exclusions, dict) or _portable_exclusion_description(audited_exclusions) != _portable_exclusion_description(exclusions.description):
        raise RuntimeError('current pair/entity/fingerprint exclusions differ from producer audit')
    if not exclusions.pair_keys or not exclusions.item_ids or (not exclusions.item_fingerprints):
        raise RuntimeError('pair, entity and fingerprint exclusions are all mandatory')
    leak_audit = audit.get('leakage_reaudit', {})
    expected_leak_counts = {'excluded_pair_keys': len(exclusions.pair_keys), 'forbidden_item_ids': len(exclusions.item_ids), 'forbidden_item_fingerprints': len(exclusions.item_fingerprints), 'fingerprints_verified': True}
    for (key, expected) in expected_leak_counts.items():
        if leak_audit.get(key) != expected:
            raise RuntimeError(f'producer leakage audit {key}={leak_audit.get(key)!r}, expected {expected!r}')
    pairs = set(zip(lo.astype(int), hi.astype(int)))
    if pairs & set(exclusions.pair_keys):
        raise RuntimeError('accepted consensus contains an excluded exact pair')
    if set(frame['id1'].astype(int)) & set(exclusions.item_ids) or set(frame['id2'].astype(int)) & set(exclusions.item_ids):
        raise RuntimeError('accepted consensus contains an excluded entity ID')
    if not {'item_fp1', 'item_fp2'}.issubset(frame.columns):
        raise RuntimeError('accepted consensus lacks recomputed item fingerprints')
    if set(frame['item_fp1'].astype(str)) & set(exclusions.item_fingerprints) or set(frame['item_fp2'].astype(str)) & set(exclusions.item_fingerprints):
        raise RuntimeError('accepted consensus contains an excluded entity fingerprint')
    return VerifiedConsensus(frame=frame, audit=audit, exclusions=exclusions)
