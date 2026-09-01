#!/bin/bash
set -e
PY=$(command -v python || command -v python3)
"$PY" - <<'PYEOF'
import glob
import sys
import zipfile

tag = f"cp{sys.version_info[0]}{sys.version_info[1]}"
cands = glob.glob(f"wheels/*-{tag}-*manylinux*x86_64.whl")
if not cands:
    cands = glob.glob("wheels/*manylinux*x86_64.whl")
print("extracting", cands[0])
zipfile.ZipFile(cands[0]).extractall("libs")
PYEOF
export PYTHONPATH="libs:${PYTHONPATH:-}"
"$PY" -c "import rapidfuzz; print('rapidfuzz', rapidfuzz.__version__)"
"$PY" -u run.py "$@"
