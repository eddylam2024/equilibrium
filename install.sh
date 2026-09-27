#!/bin/sh
# Equilibrium installer.
#
#   curl -LsSf https://raw.githubusercontent.com/eddylam2024/equilibrium/main/install.sh | sh
#
# What it does (read before piping to sh):
#   1. Installs uv (Astral's Python package manager) if it is not already on PATH.
#   2. Installs Equilibrium as an isolated CLI tool with `uv tool install`, using
#      Python >= 3.12 (uv downloads one if your system lacks it).
#   3. Makes sure uv's tool directory is on your PATH.
# It never uses sudo and only writes to ~/.local (uv) and uv's tool directory.

set -eu

REPO="${EQUILIBRIUM_REPO:-https://github.com/eddylam2024/equilibrium}"
REF="${EQUILIBRIUM_REF:-main}"

accent() { printf '\033[38;2;217;119;87m%s\033[0m' "$1"; }
muted() { printf '\033[38;5;244m%s\033[0m\n' "$1"; }

printf '\n %s %s\n\n' "$(accent '✻')" "Installing Equilibrium — Black-Litterman research terminal"

if ! command -v uv >/dev/null 2>&1; then
  muted "  ⎿  uv not found; installing it from https://astral.sh/uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
  export PATH
fi

muted "  ⎿  uv $(uv --version | cut -d' ' -f2)"
muted "  ⎿  installing from ${REPO}@${REF}"
uv tool install --force --python ">=3.12" "git+${REPO}@${REF}"
uv tool update-shell >/dev/null 2>&1 || true

printf '\n %s %s\n' "$(accent '✓')" "Done. Open a new terminal (or run: export PATH=\"\$HOME/.local/bin:\$PATH\"), then:"
printf '\n     %s        %s\n' "$(accent 'equilibrium')" "interactive shell"
printf '     %s   %s\n' "$(accent 'eq portfolio AAPL MSFT')" "one-shot command"
printf '     %s   %s\n\n' "$(accent 'equilibrium --help')" "all commands"
muted "  Research and education only — not investment advice."
printf '\n'
