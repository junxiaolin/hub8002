# make data / calib / sim / real / figs / all     (PY defaults to the project venv)
PY ?= .venv/bin/python -I
W ?= 12

.PHONY: all data calib sim real figs v2 net v3 test clean-results

all: data calib sim real figs v2 net v3

data:
	./scripts/download.sh
	$(PY) scripts/build_tables.py 8
	$(PY) scripts/build_values.py
	$(PY) scripts/build_periods.py
	$(PY) scripts/data_check.py

calib:
	$(PY) scripts/league_rates.py
	$(PY) scripts/calibrate_alpha.py

sim: export HUB8002_LAYERS = org,box
sim:
	$(PY) -m hub8002.experiments.e1_null_power 500 $(W)
	$(PY) -m hub8002.experiments.e2_matches_needed 300 $(W)

real: export HUB8002_LAYERS = org,box
real:
	$(PY) -m hub8002.experiments.e45_real_fits $(W)
	$(PY) -m hub8002.experiments.e3_half_season $(W)
	$(PY) -m hub8002.experiments.e6_permutation 200 $(W)
	$(PY) -m hub8002.experiments.e7_sequential 8
	$(PY) -m hub8002.experiments.sensitivity $(W)
	$(PY) scripts/check_generator.py

figs:
	$(PY) scripts/make_figures.py

# v2: core = most valuable to the team (attack / organisation / defence)
v2:   # needs `make data calib` (player_match_values is built in `data`)
	$(PY) scripts/league_rates.py
	$(PY) scripts/calibrate_cores.py
	$(PY) -m hub8002.experiments.v2_real
	$(PY) -m hub8002.experiments.v2_sim 2000 $(W)
	$(PY) -m hub8002.experiments.v2_checks all
	$(PY) scripts/make_figures_v2.py
	$(PY) scripts/hook_numbers.py

# network extension: linking core (value on the possession network); needs v2 results for the comparison tables
net:
	$(PY) scripts/build_chains.py $(W)
	$(PY) scripts/calibrate_net.py
	$(PY) -m hub8002.experiments.net_real
	$(PY) -m hub8002.experiments.net_sim 100 $(W)
	$(PY) -m hub8002.experiments.net_checks all
	$(PY) scripts/make_figures_net.py

# 2026-11-08 improvements: replacement level, data-selected positions, absence check, consensus check
v3:
	$(PY) scripts/calibrate_bench.py
	$(PY) -m hub8002.experiments.v2_sim 2000 $(W) bench50
	$(PY) -m hub8002.experiments.net_sim 100 $(W) bench50
	$(PY) -m hub8002.experiments.rep_real
	$(PY) scripts/build_position_minutes.py 8
	$(PY) -m hub8002.experiments.partitions 7
	$(PY) -m hub8002.experiments.partition_real
	$(PY) -m hub8002.experiments.absence
	$(PY) -m hub8002.experiments.consensus
	$(PY) scripts/make_tables_v3.py
	$(PY) scripts/data_review.py
	$(PY) scripts/make_figures_v3.py

test:
	$(PY) -m pytest -q

clean-results:   # keeps the hand-written results/SUMMARY*.md
	find results -maxdepth 1 -type f ! -name 'SUMMARY*.md' -delete
	rm -f figures/*
