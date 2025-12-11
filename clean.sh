#!/bin/bash

# Message
echo "Cleaning system-specific artifacts..."

# Removendo os arquivos .DS_Store recursivamente.
find . -name ".DS_Store" -type f -delete

# Removendo qualquer outro arquivo de metadados do MAC (arquivos que começam com "._") recursivamente.
find . -name "._*" -type f -delete

# Removendo .Trashes and outros que, às vezes, o MAC gera, quando quer.
find . -name ".Trashes" -type d -exec rm -rf {} +

# Message
echo "Cleaning Python build artifacts..."

# Remove build directories
rm -rf build/ dist/ *.egg-info/ .eggs/

# Remove Python cache and compiled files
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
find . -type f -name "*.pyo" -delete
find . -type f -name "*~" -delete

# Remove test and tool caches
rm -rf .pytest_cache/ .mypy_cache/ .tox/ .coverage .cache/

echo "Cleanup complete."


