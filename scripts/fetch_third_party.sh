#!/usr/bin/env bash
# Kirkley's official MDL hub code (A. Kirkley, Identifying hubs in directed networks, PRE 109, 034310, 2024).
# Not redistributed in this repository (no licence file); used by models/mdl.py and tests/test_mdl.py.
set -eu
cd "$(dirname "$0")/.."
mkdir -p third_party
[ -f third_party/kirkley_hubs/functions.py ] || git clone --depth 1 https://github.com/aleckirkley/Network-hubs third_party/kirkley_hubs
echo "third_party/kirkley_hubs ready"
