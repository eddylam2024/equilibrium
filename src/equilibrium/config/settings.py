"""Configuration: defaults < equilibrium.yaml < CLI flags.

Search order for the config file: ``--config PATH``, ``$EQUILIBRIUM_CONFIG``,
``./equilibrium.yaml``, ``~/.config/equilibrium/equilibrium.yaml``. The first one found wins.
Every default is an explicit, documented assumption and is displayed by ``equilibrium config``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from equilibrium.backtest.engine import Rebalance
from equilibrium.bl.equilibrium import DEFAULT_RISK_AVERSION, DEFAULT_TAU
from equilibrium.core.errors import ValidationError as EqValidationError
from equilibrium.core.types import Frequency
from equilibrium.optimize.constraints import Constraints
from equilibrium.risk.covariance import CovarianceEstimator
from equilibrium.risk.numerics import PSDPolicy

CONFIG_FILENAME = "equilibrium.yaml"


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=False)

    risk_free_rate: float = Field(
        0.04, ge=-0.05, le=0.25,
        description="Annual risk-free rate (decimal). Default 4% is an assumption; set it.")
    risk_aversion: float = Field(DEFAULT_RISK_AVERSION, gt=0, le=50,
                                 description="delta in Pi = delta Sigma w (He-Litterman 2.5).")
    tau: float = Field(DEFAULT_TAU, gt=0, le=1, description="Prior uncertainty scale tau.")
    frequency: Frequency = Field(Frequency.MONTHLY, description="Return sampling frequency.")
    lookback_period: str = Field("10y", description="History to download (Yahoo period).")
    covariance_estimator: CovarianceEstimator = CovarianceEstimator.SAMPLE
    psd_policy: PSDPolicy = Field(PSDPolicy.RAISE,
                                  description="Non-PSD covariance: 'raise' or 'repair'.")
    rebalance_frequency: Rebalance = Rebalance.MONTHLY
    backtest_lookback: int = Field(36, ge=3, description="Estimation window (observations).")
    transaction_cost_bps: float = Field(10.0, ge=0, lt=1000,
                                        description="Cost per unit of traded notional (bps).")
    constraints: Constraints = Field(default_factory=Constraints)
    cache_enabled: bool = True
    cache_ttl_hours: float = Field(12.0, gt=0)


def find_config(explicit: Path | None = None) -> Path | None:
    candidates: list[Path] = []
    if explicit:
        if not explicit.exists():
            raise EqValidationError(f"Config file not found: {explicit}")
        return explicit
    env = os.environ.get("EQUILIBRIUM_CONFIG")
    if env:
        candidates.append(Path(env).expanduser())
    candidates += [Path.cwd() / CONFIG_FILENAME,
                   Path.home() / ".config" / "equilibrium" / CONFIG_FILENAME]
    return next((p for p in candidates if p.is_file()), None)


def load_settings(path: Path | None = None, overrides: dict[str, Any] | None = None
                  ) -> tuple[Settings, Path | None]:
    """Load settings from YAML (safe_load only) and apply non-None CLI overrides."""
    found = find_config(path)
    data: dict[str, Any] = {}
    if found:
        try:
            loaded = yaml.safe_load(found.read_text()) or {}
        except yaml.YAMLError as exc:
            raise EqValidationError(f"Could not parse {found}: {exc}") from exc
        if not isinstance(loaded, dict):
            raise EqValidationError(f"{found} must contain a mapping of settings.")
        data = loaded
    for key, value in (overrides or {}).items():
        if value is None:
            continue
        if key in ("long_only", "min_weight", "max_weight"):
            data.setdefault("constraints", {})
            data["constraints"] = {**dict(data["constraints"]), key: value}
        else:
            data[key] = value
    try:
        return Settings.model_validate(data), found
    except ValidationError as exc:
        msgs = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
        raise EqValidationError(f"Invalid configuration — {msgs}") from exc


def default_config_yaml() -> str:
    s = Settings()
    return f"""# Equilibrium configuration. CLI flags override these values.
# All returns are annual decimals (0.04 = 4%).

risk_free_rate: {s.risk_free_rate}      # set to the current T-bill yield for your analysis
risk_aversion: {s.risk_aversion}        # delta in Pi = delta * Sigma * w_mkt (He & Litterman 1999)
tau: {s.tau}                  # scale of prior uncertainty on the mean (tau * Sigma)
frequency: {s.frequency.value}          # daily | weekly | monthly
lookback_period: {s.lookback_period}         # 1y 2y 3y 5y 10y 15y 20y max
covariance_estimator: {s.covariance_estimator.value}  # sample | ledoit-wolf
psd_policy: {s.psd_policy.value}          # raise | repair (eigenvalue clipping)

rebalance_frequency: {s.rebalance_frequency.value}  # monthly | quarterly
backtest_lookback: {s.backtest_lookback}       # estimation window, in `frequency` observations
transaction_cost_bps: {s.transaction_cost_bps}  # per unit of traded notional

constraints:
  long_only: true
  min_weight: null
  max_weight: null
"""
