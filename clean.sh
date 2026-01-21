#!/usr/bin/env bash
# ✨ clean.sh — HYPERGLAM repo detox (macOS + Python) ✨
# Usage:
#   DRY_RUN=1 ./clean.sh    # preview
#   ./clean.sh              # actually delete
#
# Notes:
# - Safe-ish: skips .git and never touches outside project root.
# - Fixes your main bug: rmrf isn't visible inside `find -exec bash -c ...` subshells.

set -Eeuo pipefail

DRY_RUN="${DRY_RUN:-0}"
FIND_ROOT="."

say()  { printf '%b\n' "$*"; }
ok()   { say "✅ $*"; }
info() { say "✨ $*"; }
warn() { say "⚠️  $*"; }

rmrf() {
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '🧹 (dry-run) rm -rf %q\n' "$@"
  else
    printf '🧹 rm -rf %q\n' "$@"
    rm -rf -- "$@"
  fi
}

# Delete files (via find) with glam output + dry-run support
find_del_files() {
  local root="$1"; shift
  if [[ ! -e "$root" ]]; then
    return 0
  fi

  if [[ "$DRY_RUN" == "1" ]]; then
    # print what would be deleted
    find "$root" -path "./.git" -prune -o -path "./.git/*" -prune -o \
      -type f "$@" -print 2>/dev/null \
      | sed 's/^/🧹 (dry-run) rm -f /'
  else
    # print then delete (portable: no -delete surprises)
    find "$root" -path "./.git" -prune -o -path "./.git/*" -prune -o \
      -type f "$@" -print -exec rm -f -- {} + 2>/dev/null
  fi
}

# Delete directories (via find) with glam output + dry-run support
find_del_dirs() {
  local root="$1"; shift
  if [[ ! -e "$root" ]]; then
    return 0
  fi

  if [[ "$DRY_RUN" == "1" ]]; then
    find "$root" -path "./.git" -prune -o -path "./.git/*" -prune -o \
      -type d "$@" -print 2>/dev/null \
      | sed 's/^/🧹 (dry-run) rm -rf /'
  else
    # Use -prune + -exec rm -rf to reliably delete matching dirs
    find "$root" -path "./.git" -prune -o -path "./.git/*" -prune -o \
      -type d "$@" -print -prune -exec rm -rf -- {} + 2>/dev/null
  fi
}

info "Cleaning up Python project clutter… (DRY_RUN=$DRY_RUN)"

# --- Python bytecode & caches -------------------------------------------------
info "Bytecode & caches"
find_del_dirs "$FIND_ROOT" -name "__pycache__"
find_del_files "$FIND_ROOT" \( -name "*.py[co]" -o -name "*.pyd" \)

# --- build/dist/egg-info ------------------------------------------------------
info "Build artifacts"
for d in ./build ./dist ./docs/_build ./__pypackages__ ./pip-wheel-metadata; do
  [[ -e "$d" ]] && rmrf "$d"
done
find_del_dirs "$FIND_ROOT" -name "*.egg-info"

# --- test & typing caches -----------------------------------------------------
info "Test/typing caches"
for d in ./.pytest_cache ./.mypy_cache ./.ruff_cache ./htmlcov; do
  [[ -e "$d" ]] && rmrf "$d"
done
# .coverage can be a file OR a dir; handle both
[[ -e ./.coverage ]] && rmrf ./.coverage
find_del_files "$FIND_ROOT" \( -name ".coverage" -o -name ".coverage.*" \)

# --- virtual envs / env caches (optional) ------------------------------------
info "Virtual envs / caches (optional)"
for d in ./.venv ./.tox ./.nox ./.cache; do
  [[ -e "$d" ]] && rmrf "$d"
done

# --- Jupyter checkpoints ------------------------------------------------------
info "Jupyter checkpoints"
find_del_dirs "$FIND_ROOT" -name ".ipynb_checkpoints"

# --- Sphinx doctrees ----------------------------------------------------------
info "Sphinx doctrees"
find_del_dirs ./docs -name ".doctrees"
find_del_files ./docs -name "*.doctree"

# --- OS cruft -----------------------------------------------------------------
info "OS cruft"
find_del_files "$FIND_ROOT" -name ".DS_Store"
find_del_files "$FIND_ROOT" -name "Thumbs.db"
find_del_files "$FIND_ROOT" -name "._*"
find_del_dirs  "$FIND_ROOT" -name ".Trashes"

# --- backup/editor swap files -------------------------------------------------
info "Backup/swap files"
find_del_files "$FIND_ROOT" \( -name "*~" -o -name "*.swp" -o -name "*.bak" \)

# --- compiled extensions (optional) ------------------------------------------
info "Compiled extensions (optional)"
find_del_files "$FIND_ROOT" -name "*.so"

ok "Everything Done Under the Sun."
