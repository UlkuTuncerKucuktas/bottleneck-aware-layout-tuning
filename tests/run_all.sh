#!/bin/bash
# Every check that does not need Lustre. Run before pushing.
set -e
cd "$(dirname "$0")/.."

echo "== import and registry =="
PYTHONPATH=. python3 -m layout_tuning.run list > /dev/null
echo "ok"

echo "== every experiment, real code paths =="
python3 tests/smoke.py | tail -3

echo "== the README's example, extracted from the README =="
python3 tests/readme_example.py | tail -1
