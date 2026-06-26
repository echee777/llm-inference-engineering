# phase-a/Makefile — Reproducible execution for admission control experiments
#
# Local targets: sync, gpu-start, gpu-stop
# GPU host targets (run in interactive SSH session):
#   setup, start-vllm, start-gateway, start-prometheus,
#   stop-gateway, stop-all, smoke-test
# Experiment targets (self-contained, work over non-interactive SSH):
#   run-baseline, run-controlled, run-exp1, run-exp3

SHELL := /bin/bash

# Paths (on GPU host, after rsync)
VENV_VLLM    := ~/venv-vllm
VENV_LOCUST  := ~/venv-locust
GATEWAY_PY   := day19/gateway.py
LOCUSTFILE   := day18/locustfile.py
EXP1_RUNNER  := day19/exp1_runner.py
EXP3_LOCUST  := day19/exp3_locustfile.py

# Gateway defaults (override with env vars)
GATEWAY_PORT         ?= 8001
VLLM_PORT            ?= 8000
ADMISSION_ENABLED    ?= true
BUDGET_TOKENS        ?= 141252
MAX_WAIT_SECONDS     ?= 5.0

# vLLM defaults
VLLM_MODEL  ?= Qwen/Qwen2.5-1.5B-Instruct
VLLM_DTYPE  ?= half
VLLM_MAXLEN ?= 4096
VLLM_MAXSEQ ?= 32

# Locust defaults
LOCUST_USERS      ?= 50
LOCUST_SPAWN_RATE ?= 2
LOCUST_RUNTIME    ?= 10m

# Infra repo (for gpu-start/stop from laptop)
INFRA_REPO := /Users/ctwins/code/infra_kuberay

# ─── Local (laptop) targets ────────────────────────────────────────

.PHONY: sync gpu-start gpu-stop

sync:
	rsync -avz --exclude='*.pyc' --exclude='__pycache__' \
		./ gpu-ec2:~/phase-a/

gpu-start:
	$(MAKE) -C $(INFRA_REPO) gpu-start

gpu-stop:
	$(MAKE) -C $(INFRA_REPO) gpu-stop

# ─── GPU host targets ──────────────────────────────────────────────

.PHONY: setup start-vllm start-gateway start-prometheus stop-gateway stop-all smoke-test

setup:
	$(VENV_VLLM)/bin/pip install -r requirements.txt
	$(VENV_LOCUST)/bin/pip install locust

start-vllm:
	@echo "=== Starting vLLM ($(VLLM_MODEL)) ==="
	source $(VENV_VLLM)/bin/activate && \
	nohup python -m vllm.entrypoints.openai.api_server \
		--model $(VLLM_MODEL) \
		--dtype $(VLLM_DTYPE) \
		--max-model-len $(VLLM_MAXLEN) \
		--max-num-seqs $(VLLM_MAXSEQ) \
		--port $(VLLM_PORT) \
		> vllm.log 2>&1 &
	@echo "vLLM starting in background. Tail vllm.log to monitor."

start-gateway:
	@echo "=== Starting gateway (admission=$(ADMISSION_ENABLED)) ==="
	@pkill -f "[u]vicorn gateway.*$(GATEWAY_PORT)" 2>/dev/null && sleep 2 || true
	source $(VENV_VLLM)/bin/activate && \
	ADMISSION_ENABLED=$(ADMISSION_ENABLED) \
	BUDGET_TOKENS=$(BUDGET_TOKENS) \
	MAX_WAIT_SECONDS=$(MAX_WAIT_SECONDS) \
	nohup uvicorn gateway:app \
		--host 0.0.0.0 --port $(GATEWAY_PORT) \
		--app-dir $(dir $(GATEWAY_PY)) \
		> gateway.log 2>&1 &
	@echo "Gateway starting in background. Tail gateway.log to monitor."

start-prometheus:
	@echo "=== Starting Prometheus ==="
	nohup prometheus --config.file=prometheus.yml \
		--storage.tsdb.path=prom-data \
		--web.listen-address=:9090 \
		> prometheus.log 2>&1 &
	@echo "Prometheus running on :9090"

stop-gateway:
	@pkill -f "[u]vicorn gateway.*$(GATEWAY_PORT)" 2>/dev/null \
		&& echo "Gateway stopped." || echo "No gateway process found."

stop-all: stop-gateway
	@pkill -f "[p]rometheus.*prometheus.yml" 2>/dev/null \
		&& echo "Prometheus stopped." || echo "No prometheus process found."

smoke-test:
	@echo "--- vLLM health ---"
	@curl -sf http://localhost:$(VLLM_PORT)/health && echo " OK" || echo " FAIL"
	@echo "--- Gateway health ---"
	@curl -sf http://localhost:$(GATEWAY_PORT)/debug/stats \
		| python3 -m json.tool && echo " OK" || echo " FAIL"

# ─── Experiment targets (run on GPU host) ───────────────────────────

.PHONY: run-baseline run-controlled run-exp1 run-exp3

RESULTS_DIR := results/$(shell date +%Y%m%d-%H%M%S)

run-baseline: _ensure-gateway-down
	@echo "=== Day 18 Baseline (AC off, $(LOCUST_USERS) users, $(LOCUST_RUNTIME)) ==="
	mkdir -p $(RESULTS_DIR)
	source $(VENV_VLLM)/bin/activate && \
	ADMISSION_ENABLED=false \
	uvicorn gateway:app \
		--host 0.0.0.0 --port $(GATEWAY_PORT) \
		--app-dir $(dir $(GATEWAY_PY)) \
		> $(RESULTS_DIR)/gateway_baseline.log 2>&1 &
	@sleep 3
	source $(VENV_LOCUST)/bin/activate && \
	locust -f $(LOCUSTFILE) \
		--host http://localhost:$(GATEWAY_PORT) \
		--users $(LOCUST_USERS) \
		--spawn-rate $(LOCUST_SPAWN_RATE) \
		--run-time $(LOCUST_RUNTIME) \
		--headless \
		--csv $(RESULTS_DIR)/baseline \
		--html $(RESULTS_DIR)/baseline_report.html \
		--print-stats
	@echo "=== Baseline complete. Results in $(RESULTS_DIR)/ ==="
	@pkill -f "[u]vicorn gateway.*$(GATEWAY_PORT)" 2>/dev/null || true

run-controlled: _ensure-gateway-down
	@echo "=== Day 18 Controlled (AC on, $(LOCUST_USERS) users, $(LOCUST_RUNTIME)) ==="
	mkdir -p $(RESULTS_DIR)
	source $(VENV_VLLM)/bin/activate && \
	ADMISSION_ENABLED=true \
	uvicorn gateway:app \
		--host 0.0.0.0 --port $(GATEWAY_PORT) \
		--app-dir $(dir $(GATEWAY_PY)) \
		> $(RESULTS_DIR)/gateway_controlled.log 2>&1 &
	@sleep 3
	source $(VENV_LOCUST)/bin/activate && \
	locust -f $(LOCUSTFILE) \
		--host http://localhost:$(GATEWAY_PORT) \
		--users $(LOCUST_USERS) \
		--spawn-rate $(LOCUST_SPAWN_RATE) \
		--run-time $(LOCUST_RUNTIME) \
		--headless \
		--csv $(RESULTS_DIR)/controlled \
		--html $(RESULTS_DIR)/controlled_report.html \
		--print-stats
	@echo "=== Controlled complete. Results in $(RESULTS_DIR)/ ==="
	@pkill -f "[u]vicorn gateway.*$(GATEWAY_PORT)" 2>/dev/null || true

run-exp1: _ensure-gateway-down
	@echo "=== Day 19 Exp1: Adversarial Starvation ==="
	mkdir -p $(RESULTS_DIR)
	source $(VENV_VLLM)/bin/activate && \
	ADMISSION_ENABLED=true \
	BUDGET_TOKENS=15000 \
	MAX_WAIT_SECONDS=3 \
	RATE_LIMIT_REQUESTS=100000 \
	RATE_LIMIT_TOKENS=100000000 \
	uvicorn gateway:app \
		--host 0.0.0.0 --port $(GATEWAY_PORT) \
		--app-dir $(dir $(GATEWAY_PY)) \
		> $(RESULTS_DIR)/gateway_exp1.log 2>&1 &
	@sleep 3
	@echo "Waiting for gateway..."
	@for i in $$(seq 1 30); do \
		curl -s http://localhost:$(GATEWAY_PORT)/debug/stats > /dev/null 2>&1 && break; \
		sleep 1; \
	done
	source $(VENV_VLLM)/bin/activate && \
	cd day19 && python exp1_runner.py
	@echo "=== Exp1 complete. Results in day19/results/ ==="
	@pkill -f "[u]vicorn gateway.*$(GATEWAY_PORT)" 2>/dev/null || true

# Exp3: Hockey stick sweep. Runs all 7 concurrency levels sequentially.
# Each level runs 6 min. Total wall time: ~45 min.
EXP3_LEVELS := 12:25pct 20:40pct 25:50pct 30:60pct 35:70pct 40:80pct 50:100pct

run-exp3: _ensure-gateway-down
	@echo "=== Day 19 Exp3: Hockey Stick Sweep (7 levels) ==="
	mkdir -p $(RESULTS_DIR)
	source $(VENV_VLLM)/bin/activate && \
	ADMISSION_ENABLED=true \
	uvicorn gateway:app \
		--host 0.0.0.0 --port $(GATEWAY_PORT) \
		--app-dir $(dir $(GATEWAY_PY)) \
		> $(RESULTS_DIR)/gateway_exp3.log 2>&1 &
	@sleep 3
	@echo "Waiting for gateway..."
	@for i in $$(seq 1 30); do \
		curl -s http://localhost:$(GATEWAY_PORT)/debug/stats > /dev/null 2>&1 && break; \
		sleep 1; \
	done
	@for level in $(EXP3_LEVELS); do \
		users=$${level%%:*}; \
		label=$${level##*:}; \
		echo ""; \
		echo "--- Exp3: $$label ($$users users, 6 min) ---"; \
		source $(VENV_LOCUST)/bin/activate && \
		locust -f $(EXP3_LOCUST) \
			--host http://localhost:$(GATEWAY_PORT) \
			--users $$users \
			--spawn-rate 5 \
			--run-time 6m \
			--headless \
			--csv $(RESULTS_DIR)/exp3_$$label \
			--html $(RESULTS_DIR)/exp3_$${label}_report.html \
			--print-stats; \
		echo "--- Metrics after $$label ---"; \
		curl -s http://localhost:$(GATEWAY_PORT)/debug/stats | python3 -m json.tool; \
		curl -s http://localhost:$(VLLM_PORT)/metrics \
			| grep -E "num_requests_waiting|gpu_cache_usage|time_to_first_token" || true; \
		echo ""; \
	done
	@echo "=== Exp3 complete. Results in $(RESULTS_DIR)/ ==="
	@pkill -f "[u]vicorn gateway.*$(GATEWAY_PORT)" 2>/dev/null || true

# ─── Internal helpers ───────────────────────────────────────────────

.PHONY: _ensure-gateway-down

_ensure-gateway-down:
	@pkill -f "[u]vicorn gateway.*$(GATEWAY_PORT)" 2>/dev/null && sleep 2 || true
