"""Interactive research shell with command palette, completion and a status bar.

Input is tokenised with ``shlex`` and dispatched to the same Click/Typer command tree the
one-shot CLI uses. Nothing typed here is ever evaluated as Python or passed to a system
shell.
"""

from __future__ import annotations

import shlex
from pathlib import Path

import typer
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from typer.core import TyperGroup, TyperOption

from equilibrium import __version__
from equilibrium.cli.commands import app
from equilibrium.cli.common import render_error
from equilibrium.cli.theme import ACCENT, ELBOW, STAR, console
from equilibrium.config.session import load_session
from equilibrium.config.settings import load_settings
from equilibrium.core.errors import EquilibriumError
from equilibrium.data.cache import default_cache_dir

BUILTINS = {
    "help": "Show commands",
    "clear": "Clear the screen",
    "exit": "Leave the shell",
}

PROMPT_STYLE = Style.from_dict({
    "prompt": f"{ACCENT} bold",
    "placeholder": "#6b6b6b italic",
    "bottom-toolbar": "noreverse bg:default #7a7a7a",
    "bottom-toolbar.key": "#a0a0a0",
    "bottom-toolbar.accent": ACCENT,
    "completion-menu": "bg:#1f1f1f #c8c8c8",
    "completion-menu.completion.current": f"bg:#2d2d2d {ACCENT} bold",
    "completion-menu.meta.completion": "bg:#1f1f1f #7a7a7a",
    "completion-menu.meta.completion.current": "bg:#2d2d2d #b0b0b0",
    "auto-suggestion": "#5c5c5c",
    "scrollbar.background": "bg:#1f1f1f",
    "scrollbar.button": "bg:#444444",
})


def _command_tree() -> TyperGroup:
    cmd = typer.main.get_command(app)
    if not isinstance(cmd, TyperGroup):
        raise TypeError("Expected the root command to be a group.")
    return cmd


class EquilibriumCompleter(Completer):
    """Completes commands (with descriptions), subcommands, flags and universe tickers."""

    def __init__(self, group: TyperGroup) -> None:
        self.group = group
        self.entries = {name: (c.get_short_help_str(60)) for name, c in group.commands.items()}
        self.entries.update(BUILTINS)

    def _options(self, cmd) -> list[tuple[str, str]]:
        out = []
        for p in cmd.params:
            if isinstance(p, TyperOption):
                for opt in p.opts + p.secondary_opts:
                    if opt.startswith("--"):
                        out.append((opt, (p.help or "")[:60]))
        return out

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        words = text.split()
        at_new_word = text.endswith(" ") or not words
        current = "" if at_new_word else words[-1]
        if len(words) == 0 or (len(words) == 1 and not at_new_word):
            slash = current.startswith("/")
            stem = current.lstrip("/")
            for name, meta in sorted(self.entries.items()):
                if name.startswith(stem):
                    yield Completion(("/" if slash else "") + name, start_position=-len(current),
                                     display="/" + name, display_meta=meta)
            return
        cmd = self.group.commands.get(words[0].lstrip("/"))
        if isinstance(cmd, TyperGroup):
            if len(words) == 1 or (len(words) == 2 and not at_new_word):
                for name, sub in sorted(cmd.commands.items()):
                    if name.startswith(current):
                        yield Completion(name, start_position=-len(current),
                                         display_meta=sub.get_short_help_str(60))
                return
            cmd = cmd.commands.get(words[1])
        if cmd is None:
            return
        if current.startswith("-"):
            for opt, meta in self._options(cmd):
                if opt.startswith(current):
                    yield Completion(opt, start_position=-len(current), display_meta=meta)
            return
        for t in load_session().universe:
            if t.startswith(current.upper()) and t not in words:
                yield Completion(t, start_position=-len(current), display_meta="universe")


def _toolbar() -> HTML:
    try:
        session = load_session()
        settings, _ = load_settings()
        uni = " ".join(session.universe) if session.universe else "no universe"
        views = len(session.views)
        info = (f" <accent>{STAR}</accent> <b>{_esc(uni)}</b>  ·  {views} view"
                f"{'s' if views != 1 else ''}  ·  r_f {settings.risk_free_rate:.2%}  ·  "
                f"δ {settings.risk_aversion:g}  ·  τ {settings.tau:g}")
    except EquilibriumError:
        info = " <accent>!</accent> config/session could not be read"
    keys = "   <key>/</key> commands  <key>tab</key> complete  <key>↑↓</key> history  " \
           "<key>ctrl-d</key> exit"
    return HTML(info + keys)


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def welcome() -> None:
    cwd = str(Path.cwd()).replace(str(Path.home()), "~")
    if len(cwd) > 48:
        cwd = "…" + cwd[-47:]
    body = Text()
    body.append(f"{STAR} ", style="accent.b")
    body.append("Welcome to ", style="white")
    body.append("Equilibrium", style="bold white")
    body.append(f"  v{__version__}\n\n", style="faint")
    body.append("  Black-Litterman portfolio research terminal\n", style="muted")
    body.append(f"  cwd: {cwd}", style="faint")
    console.print()
    console.print(Panel(body, border_style="accent", expand=False, padding=(0, 2)))
    console.print()
    console.print(Text(" Tips for getting started:", style="muted"))
    console.print()
    tips = [
        ("Build a portfolio", "portfolio AAPL MSFT NVDA AMZN GOOGL"),
        ("Add a view", "views add --long NVDA --short MSFT --spread 5% --confidence 70%"),
        ("Stress it", "stress --shock NVDA=-30%"),
        ("Test it", "backtest --rebalance monthly"),
    ]
    for i, (what, cmd) in enumerate(tips, 1):
        t = Text(f" {i}. ", style="muted")
        t.append(f"{what:<20}", style="white")
        t.append(cmd, style="accent")
        console.print(t)
    console.print()
    console.print(Text(" Research and education only — not investment advice. "
                       "Offline? add --synthetic.", style="faint"))
    console.print()


def show_help(group: TyperGroup) -> None:
    console.print()
    panels: dict[str, list[tuple[str, str]]] = {}
    for name, cmd in group.commands.items():
        panel = getattr(cmd, "rich_help_panel", None) or "Other"
        panels.setdefault(panel, []).append((name, cmd.get_short_help_str(70)))
    for panel in ("Data", "Research", "Black-Litterman", "Evaluation", "Output", "Setup"):
        if panel not in panels:
            continue
        console.print(Text(f" {panel}", style="muted"))
        t = Table(box=None, show_header=False, padding=(0, 2), pad_edge=False)
        t.add_column(style="accent", no_wrap=True)
        t.add_column(style="white")
        for name, helptext in sorted(panels[panel]):
            t.add_row(f"   /{name}", helptext)
        console.print(t)
        console.print()
    console.print(Text("   Add --help to any command for its options, e.g. ", style="faint")
                  + Text("portfolio --help", style="accent"))
    console.print()


def dispatch(group: TyperGroup, line: str) -> bool:
    """Run one input line. Returns False when the shell should exit."""
    try:
        args = shlex.split(line)
    except ValueError as exc:
        console.print(Text(f"  {ELBOW}  ", style="faint") + Text(f"Could not parse: {exc}",
                                                                 style="err"))
        return True
    if not args:
        return True
    if args[0] in ("equilibrium", "eq"):
        args = args[1:] or ["help"]
    args[0] = args[0].lstrip("/")
    head = args[0].lower()
    if head in ("exit", "quit", "q"):
        return False
    if head == "clear":
        console.clear()
        return True
    if head in ("help", "?"):
        if len(args) > 1:
            args = [*args[1:], "--help"]
        else:
            show_help(group)
            return True
    try:
        group.main(args=args, prog_name="equilibrium", standalone_mode=False)
    except typer.Exit:
        pass
    except typer.TyperException as exc:  # usage errors: bad option, missing value, ...
        console.print()
        fmt = getattr(exc, "format_message", None)
        console.print(Text("✗ ", style="err") + Text(fmt() if fmt else str(exc), style="err"))
        ctx = getattr(exc, "ctx", None)
        if ctx is not None:
            console.print(Text(f"  {ELBOW}  ", style="faint") +
                          Text(f"Try: {ctx.command_path.replace('equilibrium ', '')} --help",
                               style="muted"))
    except typer.Abort:
        console.print(Text("  Interrupted.", style="muted"))
    except EquilibriumError as exc:
        render_error(exc)
    except KeyboardInterrupt:
        console.print(Text("  Interrupted.", style="muted"))
    return True


def run_shell() -> None:
    group = _command_tree()
    history_dir = default_cache_dir()
    history_dir.mkdir(parents=True, exist_ok=True)
    session: PromptSession[str] = PromptSession(
        history=FileHistory(str(history_dir / "shell_history")),
        auto_suggest=AutoSuggestFromHistory(),
        completer=EquilibriumCompleter(group),
        complete_while_typing=True,
        bottom_toolbar=_toolbar,
        style=PROMPT_STYLE,
        placeholder=HTML('<placeholder>Try "portfolio AAPL MSFT NVDA" or /help</placeholder>'),
        reserve_space_for_menu=6,
    )
    welcome()
    while True:
        console.rule(style="grey23")
        try:
            line = session.prompt([("class:prompt", "› ")])
        except KeyboardInterrupt:
            continue
        except EOFError:
            break
        if not dispatch(group, line.strip()):
            break
    console.print(Text(f"  {STAR} ", style="accent") + Text("Goodbye.", style="muted"))
