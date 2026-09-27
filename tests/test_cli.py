"""CLI and interactive-shell tests. Offline: every command runs on the synthetic provider."""

from __future__ import annotations

import json

import pytest
from prompt_toolkit.document import Document
from typer.testing import CliRunner

from equilibrium.cli.commands import app

runner = CliRunner()
U = ["AAA", "BBB", "CCC", "DDD"]


@pytest.fixture(autouse=True)
def offline(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("EQUILIBRIUM_PROVIDER", "synthetic")
    monkeypatch.setenv("EQUILIBRIUM_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("COLUMNS", "120")


def run(*args):
    return runner.invoke(app, list(args))


def test_help_lists_all_commands():
    r = run("--help")
    assert r.exit_code == 0
    for cmd in ("data", "analyze", "risk", "covariance", "views", "portfolio", "optimize",
                "frontier", "backtest", "stress", "export", "config"):
        assert cmd in r.output


@pytest.mark.parametrize("cmd", ["views", "optimize", "backtest", "portfolio", "stress"])
def test_subcommand_help(cmd):
    assert run(cmd, "--help").exit_code == 0


def test_portfolio_json_no_views_is_market():
    r = run("portfolio", *U, "--json")
    assert r.exit_code == 0, r.output
    out = json.loads(r.output)
    assert out["meta"]["synthetic"] is True
    for t in U:
        assert out["bl_weights"][t] == pytest.approx(out["market_weights"][t], abs=1e-5)
    assert out["portfolio"]["label"] == "model estimate"


def test_views_workflow_changes_portfolio():
    assert run("portfolio", *U, "--json").exit_code == 0  # saves universe
    r = run("views", "add", "--long", "DDD", "--short", "AAA", "--spread", "8%",
            "--confidence", "80%")
    assert r.exit_code == 0, r.output
    assert run("views", "add", "--asset", "BBB", "--return", "0.02").exit_code == 0
    listing = run("views", "list")
    assert "DDD" in listing.output and "BBB" in listing.output
    out = json.loads(run("portfolio", "--json").output)  # universe reused from session
    assert len(out["views"]) == 2
    assert out["omega_methods"] == ["idzorek", "he-litterman"]
    assert out["Q"][1] == pytest.approx(0.02 - 0.04)  # absolute view -> excess
    assert run("views", "remove", "2").exit_code == 0
    assert run("views", "clear").exit_code == 0
    assert "No views" in run("views", "list").output


def test_view_input_validation():
    assert run("views", "add", "--asset", "AAA", "--return", "8").exit_code == 1
    assert run("views", "add", "--long", "AAA", "--short", "AAA", "--spread", "1%"
               ).exit_code == 1
    assert run("views", "add", "--asset", "AAA").exit_code == 1


@pytest.mark.parametrize("args", [
    ["data", *U],
    ["analyze", *U],
    ["covariance", *U],
    ["risk", *U, "--weights", "AAA=0.4,BBB=0.3,CCC=0.2,DDD=0.1"],
    ["portfolio", *U, "--verbose"],
    ["optimize", *U, "-o", "min-variance"],
    ["optimize", *U, "-o", "target-return", "--target-return", "13%"],
    ["optimize", *U, "-r", "historical"],
    ["frontier", *U, "--points", "12"],
    ["backtest", *U, "--lookback", "24", "--cost", "5"],
    ["stress", *U, "--shock", "AAA=-0.3", "--from", "2020-02-19", "--to", "2020-03-23"],
    ["config", "show"],
])
def test_commands_render(args):
    r = run(*args)
    assert r.exit_code == 0, r.output
    assert "Traceback" not in r.output


def test_synthetic_data_is_labelled_on_screen():
    assert "SYNTHETIC" in run("analyze", *U).output


def test_plots_are_written(tmp_path):
    r = run("covariance", *U, "--plot", "--plot-dir", str(tmp_path / "p"))
    assert r.exit_code == 0, r.output
    assert (tmp_path / "p" / "correlation.png").exists()
    r = run("backtest", *U, "--lookback", "24", "--strategy", "equal", "--plot",
            "--plot-dir", str(tmp_path / "p"), "--plot-format", "svg")
    assert r.exit_code == 0, r.output
    assert (tmp_path / "p" / "growth.svg").exists()


def test_export_json_and_csv(tmp_path):
    r = run("export", *U, "--out", str(tmp_path / "bundle.json"), "--no-backtest")
    assert r.exit_code == 0, r.output
    data = json.loads((tmp_path / "bundle.json").read_text())
    assert {"bl_weights", "covariance", "correlation", "assumptions"} <= data.keys()
    r = run("export", *U, "--format", "csv", "--out", str(tmp_path / "csv"))
    assert r.exit_code == 0, r.output
    for name in ("weights", "covariance", "correlation", "expected_returns", "risk_metrics",
                 "backtest", "metadata"):
        assert any(p.stem == name for p in (tmp_path / "csv").iterdir())


def test_errors_are_clean():
    r = run("analyze", "AAA;rm")
    assert r.exit_code == 1 and "Invalid ticker" in r.output
    r = run("optimize", *U, "--max-weight", "0.1")
    assert r.exit_code == 1 and "infeasible" in r.output
    r = run("portfolio", *U, "--rf", "4")
    assert r.exit_code == 1 and "ambiguous" in r.output
    r = run("analyze", *U, "--json", "--period", "7y")
    assert r.exit_code == 1 and json.loads(r.output)["error"] == "ValidationError"


def test_missing_universe_message():
    r = run("portfolio")
    assert r.exit_code == 1 and "No tickers given" in r.output


def test_config_init_and_override(tmp_path):
    assert run("config", "init").exit_code == 0
    assert (tmp_path / "equilibrium.yaml").exists()
    assert run("config", "init").exit_code == 1  # refuses to overwrite
    out = json.loads(run("portfolio", *U, "--json", "--rf", "3%").output)
    assert out["meta"]["risk_free_rate"] == pytest.approx(0.03)


# ------------------------------------------------------------------ interactive shell
def test_shell_dispatch_runs_commands_and_builtins(capsys):
    from equilibrium.cli.shell import _command_tree, dispatch

    group = _command_tree()
    assert dispatch(group, "/portfolio AAA BBB CCC DDD --json") is True
    assert dispatch(group, "help") is True
    assert dispatch(group, "portfolio --not-an-option") is True  # usage error, no crash
    assert dispatch(group, 'views add --asset "unterminated') is True  # shlex error handled
    assert dispatch(group, "/exit") is False


def test_shell_never_executes_shell_syntax(tmp_path):
    from equilibrium.cli.shell import _command_tree, dispatch

    marker = tmp_path / "pwned"
    dispatch(_command_tree(), f"analyze AAA; touch {marker}")
    dispatch(_command_tree(), f"analyze $(touch {marker})")
    assert not marker.exists()


def test_shell_completer():
    from equilibrium.cli.shell import EquilibriumCompleter, _command_tree

    comp = EquilibriumCompleter(_command_tree())

    def complete(text):
        return [c.text for c in comp.get_completions(Document(text), None)]

    assert "/portfolio" in complete("/port")
    assert "add" in complete("views a")
    assert "--confidence" in complete("views add --conf")
    assert "--objective" in complete("optimize --obj")
    run("portfolio", *U, "--json")
    assert "BBB" in complete("analyze AAA B")
