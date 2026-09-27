"""Export research results to JSON or CSV (a directory of tables).

Every export embeds provenance: data source, sample window, whether data is synthetic, and
the list of model assumptions, so exported numbers cannot be separated from their caveats.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from equilibrium.core.errors import ValidationError


def to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy/pandas objects into JSON-serialisable primitives."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, pd.DataFrame):
        return {str(i): to_jsonable(row.to_dict()) for i, row in obj.iterrows()}
    if isinstance(obj, pd.Series):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    if hasattr(obj, "value") and isinstance(obj.value, str):  # enums
        return obj.value
    if hasattr(obj, "model_dump"):
        return to_jsonable(obj.model_dump())
    return obj


def write_json(payload: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2))
    return path


def write_csv_bundle(tables: dict[str, pd.DataFrame], meta: dict[str, Any], directory: Path
                     ) -> list[Path]:
    """One CSV per table plus ``metadata.json`` with provenance and assumptions."""
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for name, frame in tables.items():
        if not name.replace("_", "").isalnum():
            raise ValidationError(f"Invalid table name {name!r}")
        p = directory / f"{name}.csv"
        frame.to_csv(p)
        written.append(p)
    written.append(write_json(meta, directory / "metadata.json"))
    return written
