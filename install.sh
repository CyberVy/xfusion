#!/usr/bin/env bash
set -e

echo "Installing xfusion Backend dependencies..."
if command -v uv > /dev/null 2>&1; then
    uv sync
else
    pip install -e .
fi
