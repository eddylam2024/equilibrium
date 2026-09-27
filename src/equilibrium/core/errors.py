"""Exception hierarchy. Every failure the user can cause maps to one of these."""


class EquilibriumError(Exception):
    """Base class for all expected, user-facing errors."""


class DataError(EquilibriumError):
    """Market data could not be retrieved or is unusable."""


class InvalidTickerError(DataError):
    """One or more tickers returned no data (unknown, delisted, or mistyped)."""

    def __init__(self, tickers: list[str], detail: str = "") -> None:
        self.tickers = tickers
        msg = f"No usable data for: {', '.join(tickers)}."
        if detail:
            msg += f" {detail}"
        msg += " Securities are never substituted automatically; remove or correct them."
        super().__init__(msg)


class InsufficientHistoryError(DataError):
    """Not enough overlapping observations to estimate the requested statistics."""


class ValidationError(EquilibriumError):
    """Invalid user input or inconsistent dimensions."""


class NumericalError(EquilibriumError):
    """A matrix is singular, not positive semidefinite, or otherwise numerically unusable."""


class InfeasibleError(EquilibriumError):
    """An optimisation problem has no solution under the requested constraints."""
