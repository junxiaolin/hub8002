#!/usr/bin/env bash
# Download StatsBomb event + lineup files for ids in data/match_ids.txt (8 parallel curls, .part then rename).
set -u
cd "$(dirname "$0")/.."
BASE=https://raw.githubusercontent.com/statsbomb/open-data/master/data
fetch() {
  kind=$1; id=$2; out=data/raw/$kind/$id.json
  [ -s "$out" ] && return 0
  curl -sS --retry 4 --retry-delay 2 -o "$out.part" "$BASE/$kind/$id.json" && mv "$out.part" "$out"
}
export -f fetch; export BASE
for kind in lineups events; do
  xargs -P 8 -I{} bash -c "fetch $kind {}" < data/match_ids.txt
done
echo "events: $(ls data/raw/events/*.json | wc -l)  lineups: $(ls data/raw/lineups/*.json | wc -l)"
