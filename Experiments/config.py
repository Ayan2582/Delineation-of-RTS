"""YAML config loading: `_base_` inheritance, dotted CLI overrides, stream resolution."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

CODE_DIR = Path(__file__).resolve().parent


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    """Load a YAML config, merge its `_base_` chain, then apply `a.b.c=value` overrides."""
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = CODE_DIR / path
    cfg = yaml.safe_load(path.read_text())
    if "_base_" in cfg:
        cfg = _merge(load_config(path.parent / cfg.pop("_base_")), cfg)
    for item in overrides or []:
        key, value = item.split("=", 1)
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(value)
    return cfg


def resolve_path(p: str | Path) -> Path:
    """Relative config paths are relative to this code directory, not the cwd."""
    p = Path(p)
    return p if p.is_absolute() else (CODE_DIR / p).resolve()


def resolve_streams(cfg: dict) -> dict[str, list[str]]:
    """Band names per stream after the ablation flags.

    Returns {"swin": [...], "terrain": [...]} ("terrain" is empty for single-stream experiments).
    """
    swin = list(cfg["streams"]["swin"])
    terrain = list(cfg["streams"].get("terrain") or [])
    flags = cfg.get("flags") or {}
    if flags.get("drop_ndwi"):
        swin = [b for b in swin if b != "ndwi"]
        terrain = [b for b in terrain if b != "ndwi"]
    if flags.get("ndwi_to_terrain"):
        if not terrain:
            raise ValueError("ndwi_to_terrain needs a terrain stream (E3/E4 only)")
        swin = [b for b in swin if b != "ndwi"]
        if "ndwi" not in terrain:
            terrain.append("ndwi")
    return {"swin": swin, "terrain": terrain}


def run_name(cfg: dict) -> str:
    """Experiment name plus a suffix for each active ablation flag, e.g. `e2_drop_ndwi`."""
    flags = cfg.get("flags") or {}
    return "_".join([cfg["name"], *sorted(k for k, v in flags.items() if v)])
