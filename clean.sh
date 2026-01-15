#!/bin/bash
# Clean Python project artifacts and OS cruft (safe-ish).
# Usage: DRY_RUN=1 ./clean.sh  # to preview

set -euo pipefail

echo "✨ Cleaning up Python project clutter... ✨"
DRY_RUN="${DRY_RUN:-0}"

rmrf() {
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '🧹 (dry-run) %s\n' "$*"
  else
    printf '🧹 %s\n' "$*"
    rm -rf "$@"
  fi
}

# Helpers to run find safely (skip .git)
FIND_ROOT="."
SKIP_GIT='-not -path "./.git/*"'

# Python bytecode & caches
find "$FIND_ROOT" $SKIP_GIT -type d -name "__pycache__" -prune -exec bash -c 'rmrf "$@"' _ {} + 2>/dev/null
find "$FIND_ROOT" $SKIP_GIT -type f \( -name "*.py[co]" -o -name "*.pyd" \) -print -delete

# build/dist/egg-info
for d in ./build ./dist ./docs/_build ./__pypackages__ ./pip-wheel-metadata; do
  [[ -e "$d" ]] && rmrf "$d"
done
find "$FIND_ROOT" $SKIP_GIT -type d -name "*.egg-info" -prune -exec bash -c 'rmrf "$@"' _ {} + 2>/dev/null

# test & typing caches
for d in ./.pytest_cache ./.mypy_cache ./.ruff_cache ./htmlcov ./.coverage; do
  [[ -e "$d" ]] && rmrf "$d"
done
# coverage files
find "$FIND_ROOT" $SKIP_GIT -type f \( -name ".coverage" -o -name ".coverage.*" \) -print -delete

# virtual envs / env caches (comment out if you want to keep them)
for d in ./.venv ./.tox ./.nox ./.cache; do
  [[ -e "$d" ]] && rmrf "$d"
done

# Jupyter checkpoints
find "$FIND_ROOT" $SKIP_GIT -type d -name ".ipynb_checkpoints" -prune -exec bash -c 'rmrf "$@"' _ {} + 2>/dev/null

# Sphinx doctrees (if outside _build)
find ./docs $SKIP_GIT -type d -name ".doctrees" -prune -exec bash -c 'rmrf "$@"' _ {} + 2>/dev/null || true
find ./docs $SKIP_GIT -type f -name "*.doctree" -print -delete 2>/dev/null || true

# OS cruft
find "$FIND_ROOT" $SKIP_GIT -type f -name ".DS_Store" -print -delete
find "$FIND_ROOT" $SKIP_GIT -type f -name "Thumbs.db" -print -delete
find "$FIND_ROOT" $SKIP_GIT -type f -name "._*" -print -delete
find "$FIND_ROOT" $SKIP_GIT -type d -name ".Trashes" -prune -exec bash -c 'rmrf "$@"' _ {} + 2>/dev/null

# backup/editor swap files
find "$FIND_ROOT" $SKIP_GIT -type f \( -name "*~" -o -name "*.swp" -o -name "*.bak" \) -print -delete

# compiled extensions (optional)
find "$FIND_ROOT" $SKIP_GIT -type f -name "*.so" -print -delete

echo "✅ Everything Done Under the Sun."
