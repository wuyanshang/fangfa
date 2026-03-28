"""
中间结果分批写入 CSV（检查点）
"""
import csv
import json
import os
from datetime import datetime, timezone

from .config import (
    CHECKPOINT_DIR,
    PHASE2_WRITE_BATCH_SIZE,
    PHASE3_WRITE_BATCH_SIZE,
    PHASE4_WRITE_BATCH_SIZE,
)
from .models import CandidatePair, JudgeResult


PHASE2_FILE = "phase2_all_pairs.csv"
PHASE3_FILE = "phase3_judge_results.csv"
PHASE4_FILE = "phase4_supplement_results.csv"
MANIFEST_FILE = "manifest.json"


def _iter_batches(items, batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def _ensure_dir():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def _path(filename: str) -> str:
    return os.path.join(CHECKPOINT_DIR, filename)


def _append_rows(filepath: str, headers: list[str], rows: list[list]):
    file_exists = os.path.exists(filepath)
    with open(filepath, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(headers)
        writer.writerows(rows)


def _manifest_path() -> str:
    return _path(MANIFEST_FILE)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_manifest() -> dict:
    return {
        "version": 1,
        "updated_at": _now_iso(),
        "stages": {
            "phase2": {"completed": False, "updated_at": ""},
            "phase3": {"completed": False, "updated_at": ""},
            "phase4": {"completed": False, "updated_at": ""},
        },
    }


def _write_manifest(manifest: dict):
    manifest["updated_at"] = _now_iso()
    with open(_manifest_path(), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def load_manifest() -> dict:
    _ensure_dir()
    p = _manifest_path()
    if not os.path.exists(p):
        manifest = _default_manifest()
        _write_manifest(manifest)
        return manifest
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def mark_stage_completed(stage: str):
    manifest = load_manifest()
    if stage not in manifest.get("stages", {}):
        manifest["stages"][stage] = {"completed": False, "updated_at": ""}
    manifest["stages"][stage]["completed"] = True
    manifest["stages"][stage]["updated_at"] = _now_iso()
    _write_manifest(manifest)


def reset_stage(stage: str):
    manifest = load_manifest()
    if stage not in manifest.get("stages", {}):
        manifest["stages"][stage] = {"completed": False, "updated_at": ""}
    manifest["stages"][stage]["completed"] = False
    manifest["stages"][stage]["updated_at"] = _now_iso()
    _write_manifest(manifest)


def is_stage_completed(stage: str) -> bool:
    manifest = load_manifest()
    return bool(manifest.get("stages", {}).get(stage, {}).get("completed", False))


def prepare_checkpoint_outputs(force_reset: bool = False):
    """运行前初始化检查点文件。"""
    _ensure_dir()
    if force_reset:
        for name in [PHASE2_FILE, PHASE3_FILE, PHASE4_FILE]:
            p = _path(name)
            if os.path.exists(p):
                os.remove(p)
        _write_manifest(_default_manifest())
    else:
        load_manifest()


def has_phase2_pairs() -> bool:
    return os.path.exists(_path(PHASE2_FILE))


def has_phase3_results() -> bool:
    return os.path.exists(_path(PHASE3_FILE))


def has_phase4_results() -> bool:
    return os.path.exists(_path(PHASE4_FILE))


def _read_csv_rows(filepath: str) -> list[dict]:
    if not os.path.exists(filepath):
        return []
    with open(filepath, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return list(reader)


def load_phase2_pairs() -> list[CandidatePair]:
    rows = _read_csv_rows(_path(PHASE2_FILE))
    pairs = []
    for r in rows:
        pairs.append(CandidatePair(
            name_a=r["name_a"],
            name_b=r["name_b"],
            source=r.get("source", "recall") or "recall",
            score=float(r.get("score", 0) or 0),
        ))
    return pairs


def load_phase3_results() -> list[JudgeResult]:
    rows = _read_csv_rows(_path(PHASE3_FILE))
    result_map: dict[tuple[str, str], JudgeResult] = {}
    for r in rows:
        jr = JudgeResult(
            name_a=r["name_a"],
            name_b=r["name_b"],
            result=r["result"],
            source=r.get("source", "recall") or "recall",
        )
        result_map[tuple(sorted([jr.name_a, jr.name_b]))] = jr
    return list(result_map.values())


def load_phase4_results() -> list[JudgeResult]:
    rows = _read_csv_rows(_path(PHASE4_FILE))
    result_map: dict[tuple[str, str], JudgeResult] = {}
    for r in rows:
        jr = JudgeResult(
            name_a=r["name_a"],
            name_b=r["name_b"],
            result=r["result"],
            source=r.get("source", "supplement") or "supplement",
        )
        result_map[tuple(sorted([jr.name_a, jr.name_b]))] = jr
    return list(result_map.values())


def write_phase2_pairs_batched(all_pairs: list):
    """阶段2候选对分批写入。"""
    filepath = _path(PHASE2_FILE)
    if os.path.exists(filepath):
        os.remove(filepath)
    headers = ["name_a", "name_b", "source", "score"]
    for batch in _iter_batches(all_pairs, PHASE2_WRITE_BATCH_SIZE):
        rows = [[p.name_a, p.name_b, p.source, p.score] for p in batch]
        _append_rows(filepath, headers, rows)


def append_phase3_results(results: list):
    """阶段3判断结果分批追加。"""
    if not results:
        return
    filepath = _path(PHASE3_FILE)
    headers = ["name_a", "name_b", "result", "source"]
    for batch in _iter_batches(results, PHASE3_WRITE_BATCH_SIZE):
        rows = [[r.name_a, r.name_b, r.result, r.source] for r in batch]
        _append_rows(filepath, headers, rows)


def append_phase4_results(results: list):
    """阶段4补判结果分批追加。"""
    if not results:
        return
    filepath = _path(PHASE4_FILE)
    headers = ["name_a", "name_b", "result", "source"]
    for batch in _iter_batches(results, PHASE4_WRITE_BATCH_SIZE):
        rows = [[r.name_a, r.name_b, r.result, r.source] for r in batch]
        _append_rows(filepath, headers, rows)
