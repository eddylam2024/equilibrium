"""Explicit, inspectable workspace state: the current universe and investor views.

Stored as JSON at ``./.equilibrium/session.json`` (override with ``$EQUILIBRIUM_SESSION``).
Nothing is kept in hidden global variables; ``equilibrium config show`` prints the path.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from pydantic import BaseModel, Field

from equilibrium.bl.views import View
from equilibrium.core.errors import ValidationError


class Session(BaseModel):
    universe: list[str] = Field(default_factory=list)
    views: list[View] = Field(default_factory=list)
    market_weights: dict[str, float] | None = None


def session_path() -> Path:
    env = os.environ.get("EQUILIBRIUM_SESSION")
    return Path(env).expanduser() if env else Path.cwd() / ".equilibrium" / "session.json"


def load_session(path: Path | None = None) -> Session:
    p = path or session_path()
    if not p.exists():
        return Session()
    try:
        return Session.model_validate(json.loads(p.read_text()))
    except (OSError, ValueError) as exc:
        raise ValidationError(f"Session file {p} is corrupt ({exc}); delete it to reset.") from exc


def save_session(session: Session, path: Path | None = None) -> Path:
    p = path or session_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(session.model_dump_json(indent=2))
    return p
