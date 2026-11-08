#!/usr/bin/env bash
# Second downloader working from the end of the id list (own .part2 suffix so the two never share a temp file).
set -u
cd "$(dirname "$0")/.."
BASE=https://raw.githubusercontent.com/statsbomb/open-data/master/data
fetch() {
  kind=$1; id=$2; out=data/raw/$kind/$id.json
  [ -s "$out" ] && return 0
  curl -sS --retry 4 --retry-delay 2 -o "$out.part2" "$BASE/$kind/$id.json" && mv "$out.part2" "$out"
}
export -f fetch; export BASE
xargs -P 8 -I{} bash -c "fetch events {}" < data/match_ids_rev.txt
echo "rev done: events $(ls data/raw/events/*.json | wc -l)"
