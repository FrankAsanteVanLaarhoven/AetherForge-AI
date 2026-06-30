ENV ?= conda run -n aetherforge-train

# ── Quick checks ──────────────────────────────────────────────────────────
.PHONY: test-aetherforge test-finetune test-eval help

test-aetherforge:
	$(ENV) python aetherforge/model.py

test-finetune:
	$(ENV) python scripts/finetune_qwen25_vl.py --mode text --test-run

test-eval:
	$(ENV) python scripts/evaluate_model.py --benchmark text

test-all: test-aetherforge test-finetune test-eval
	@echo "\nAll smoke-tests passed."

# ── Data ──────────────────────────────────────────────────────────────────
.PHONY: data data-multimodal diagram

data:
	$(ENV) python scripts/generate_synthetic_data.py 2>/dev/null || \
	$(ENV) python multimodal_example/generate_dummy_data.py

data-multimodal:
	$(ENV) python multimodal_example/generate_dummy_data.py

diagram:
	$(ENV) python scripts/generate_architecture_diagram.py

# ── AetherForge pretraining ───────────────────────────────────────────────
.PHONY: train-aetherforge train-aetherforge-test

train-aetherforge:
	$(ENV) python scripts/train_aetherforge.py \
		--data data/synthetic_data.jsonl \
		--steps 2000 --batch-size 4 --seq-len 256 --warmup-steps 100

train-aetherforge-gc:
	$(ENV) python scripts/train_aetherforge.py \
		--stream --config 1B --gradient-checkpointing --8bit-adam \
		--steps 15000 --seq-len 256 --batch-size 1

train-fineweb:
	$(ENV) python scripts/train_aetherforge.py \
		--stream --config 128M \
		--steps 10000 --seq-len 256

train-fineweb-ddp:
	torchrun --nproc_per_node=$(or $(NGPU),2) scripts/train_aetherforge.py \
		--stream --config 1B --gradient-checkpointing \
		--steps 50000 --seq-len 512

train-aetherforge-test:
	$(ENV) python scripts/train_aetherforge.py --test-run

# ── Qwen2.5-VL fine-tuning ────────────────────────────────────────────────
.PHONY: finetune finetune-multimodal finetune-test

finetune:
	$(ENV) python scripts/finetune_qwen25_vl.py \
		--mode text --data data/synthetic_data.jsonl

finetune-multimodal:
	$(ENV) python scripts/finetune_qwen25_vl.py \
		--mode multimodal \
		--data multimodal_example/multimodal_data.jsonl

finetune-test:
	$(ENV) python scripts/finetune_qwen25_vl.py --mode text --test-run

# ── Inference ─────────────────────────────────────────────────────────────
.PHONY: infer infer-4bit

infer:
	$(ENV) python scripts/inference_qwen25_vl.py --interactive

infer-4bit:
	$(ENV) python scripts/run_4bit.py --interactive

# ── Evaluation ────────────────────────────────────────────────────────────
.PHONY: eval eval-vision eval-all eval-checkpoints

eval:
	$(ENV) python scripts/evaluate_model.py --benchmark text

eval-checkpoints:
	$(ENV) python scripts/eval_checkpoints.py \
		--n-chunks 300 --n-alpaca 100 --output outputs/eval_results

eval-vision:
	$(ENV) python scripts/evaluate_model.py \
		--benchmark all \
		--image-dir multimodal_example/images

eval-lora:
	$(ENV) python scripts/evaluate_model.py \
		--lora-path outputs/qwen25_vl_lora/final \
		--benchmark all \
		--image-dir multimodal_example/images

eval-compare:
	$(ENV) python scripts/evaluate_model.py \
		--lora-path outputs/qwen25_vl_lora/final \
		--benchmark all \
		--image-dir multimodal_example/images \
		--compare-base

# ── LoRA merge ────────────────────────────────────────────────────────────
.PHONY: merge-lora serve serve-qwen distill distill-test

merge-lora:
	$(ENV) python scripts/merge_lora.py \
		--lora-path outputs/qwen25_vl_lora/final \
		--output-dir outputs/qwen25_vl_merged

# ── Code agent ────────────────────────────────────────────────────────────
.PHONY: data-code data-agent-only finetune-code-agent agent-loop \
        finetune-qwen-code-agent finetune-qwen-code-agent-test \
        finetune-qwen-code-agent-agent-only \
        agent-loop-qwen agent-loop-qwen-benchmark \
        eval-code-agent eval-code-agent-compare eval-code-agent-best-of-n

data-code:
	$(ENV) python scripts/generate_code_data.py --n-code 5000 --n-react 1500
	$(ENV) python scripts/generate_execution_traces.py
	cat data/execution_traces.jsonl >> data/code_agent_data.jsonl

data-agent-only:
	$(ENV) python scripts/generate_code_data.py --agent-only --n-react 2000
	$(ENV) python scripts/generate_execution_traces.py

finetune-code-agent:
	$(ENV) python scripts/finetune_code_agent.py \
		--checkpoint outputs/aetherforge_distill_5k/final/model.pt \
		--config     outputs/aetherforge_distill_5k/final/config.json \
		--data       data/code_agent_data.jsonl \
		--steps 3000 --lr 2e-5 --batch-size 1 --grad-accum 16 --max-length 256

finetune-1B-code-agent:
	PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
	$(ENV) python scripts/finetune_code_agent.py \
		--checkpoint outputs/aetherforge_1B_bpe/init/model.pt \
		--config     outputs/aetherforge_1B_bpe/init/config.json \
		--data       data/code_agent_data.jsonl \
		--output     outputs/aetherforge_1B_code_agent \
		--steps 5000 --lr 3e-5 --batch-size 1 --grad-accum 8 \
		--max-length 256 --warmup-steps 200 \
		--8bit-adam --gradient-checkpointing --amp

adapt-tokenizer-1B:
	$(ENV) python scripts/adapt_tokenizer.py \
		--checkpoint outputs/aetherforge_1B_pretrain/final/model.pt \
		--config     1B \
		--output     outputs/aetherforge_1B_bpe/init

agent-loop:
	$(ENV) python scripts/agent_loop.py \
		--checkpoint outputs/aetherforge_1B_code_agent/final/model.pt \
		--config     outputs/aetherforge_1B_bpe/init/config.json \
		--interactive

agent-benchmark:
	$(ENV) python scripts/agent_loop.py \
		--checkpoint outputs/aetherforge_1B_code_agent/final/model.pt \
		--config     outputs/aetherforge_1B_bpe/init/config.json \
		--benchmark

# ── Fast path: Qwen2.5-0.5B-Instruct + LoRA code-agent ───────────────────
finetune-qwen-code-agent-test:
	$(ENV) python scripts/finetune_qwen_code_agent.py --test-run

finetune-qwen-code-agent-test-agent-only:
	$(ENV) python scripts/finetune_qwen_code_agent.py --test-run \
		--agent-only --agent-contract strict

finetune-qwen-code-agent:
	$(ENV) python scripts/finetune_qwen_code_agent.py \
		--steps 3000 --lr 2e-4 \
		--batch-size 2 --grad-accum 8 --max-length 1024

finetune-qwen-code-agent-agent-only:
	$(ENV) python scripts/finetune_qwen_code_agent.py \
		--steps 3000 --lr 2e-4 \
		--batch-size 2 --grad-accum 8 --max-length 1024 \
		--agent-only --agent-contract strict

finetune-qwen-code-agent-wandb:
	$(ENV) python scripts/finetune_qwen_code_agent.py \
		--steps 3000 --lr 2e-4 \
		--batch-size 2 --grad-accum 8 --max-length 1024 --wandb

agent-loop-qwen:
	$(ENV) python scripts/agent_loop.py \
		--hf-model outputs/qwen_code_agent/final \
		--interactive

agent-loop-qwen-benchmark:
	$(ENV) python scripts/agent_loop.py \
		--hf-model outputs/qwen_code_agent/final \
		--benchmark

# Single-pass evaluation (16-task benchmark)
eval-code-agent:
	$(ENV) python scripts/evaluate_code_agent.py \
		--hf-model outputs/qwen_code_agent/final \
		--mode single \
		--output outputs/eval_code_agent

# Best-of-3 evaluation
eval-code-agent-best-of-n:
	$(ENV) python scripts/evaluate_code_agent.py \
		--hf-model outputs/qwen_code_agent/final \
		--mode best_of_n --n 3 \
		--output outputs/eval_code_agent

# Full comparison: base Qwen vs fine-tuned, single vs best-of-3
eval-code-agent-compare:
	$(ENV) python scripts/evaluate_code_agent.py \
		--hf-model outputs/qwen_code_agent/final \
		--mode compare --n 3 \
		--compare-base \
		--output outputs/eval_code_agent

# ── Qwen2.5-Coder-1.5B base model evaluation (no LoRA) ───────────────────
# Run before fine-tuning 1.5B. Compare against 0.5B strict baseline first.
eval-coder-1b5-base:
	$(ENV) python scripts/evaluate_code_agent.py \
		--hf-model Qwen/Qwen2.5-Coder-1.5B-Instruct \
		--mode single \
		--scoring-mode verified_agent \
		--agent-contract strict \
		--output outputs/eval_coder_1b5_base

eval-coder-1b5-base-compare:
	$(ENV) python scripts/evaluate_code_agent.py \
		--hf-model Qwen/Qwen2.5-Coder-1.5B-Instruct \
		--mode compare --n 3 \
		--scoring-mode verified_agent \
		--agent-contract strict \
		--output outputs/eval_coder_1b5_base

# ── Offline vector memory ────────────────────────────────────────────────────
.PHONY: memory-extract memory-build memory-audit memory-retrieve \
        memory-build-full memory-validate \
        eval-base-only eval-lora-only eval-memory-only \
        eval-lora-memory eval-lora-memory-best-of-3 \
        test-memory test-extract

# Extract verified records from eval outputs → memory/raw/extracted_memories.jsonl
memory-extract:
	$(ENV) python scripts/extract_memory_from_evals.py \
		--outputs-dir outputs \
		--out memory/raw/extracted_memories.jsonl \
		--max-records 500

# Build the memory index from memory/raw/*.jsonl
memory-build:
	$(ENV) python scripts/build_vector_memory.py \
		--raw-dir memory/raw \
		--index-dir memory/index

# Dry-run: validate records only, do not write index
memory-validate:
	$(ENV) python scripts/build_vector_memory.py \
		--raw-dir memory/raw --dry-run

# Audit raw records + index consistency
memory-audit:
	$(ENV) python scripts/audit_memory.py \
		--raw-dir memory/raw \
		--index-dir memory/index

# Quick retrieval test from CLI
memory-retrieve:
	$(ENV) python scripts/retrieve_memory.py \
		--query "word_count frequency dict case-insensitive" \
		--top-k 3 --format text

# Full pipeline: extract → audit → build
memory-build-full: memory-extract memory-audit memory-build

# Run the memory test suite
test-memory:
	$(ENV) python -m pytest tests/test_vector_memory.py -v

# Run the extraction test suite
test-extract:
	$(ENV) python -m pytest tests/test_extract_memory.py -v

# Run all memory tests
test-memory-all:
	$(ENV) python -m pytest tests/test_vector_memory.py tests/test_extract_memory.py -v

# ── Evaluation modes: base / LoRA / memory / combined ─────────────────────
# Uses Qwen2.5-Coder-1.5B-Instruct with verified_agent strict scoring.
LORA_PATH := outputs/qwen15b_memory_300steps/final
MEM_INDEX := memory/index

eval-base-only:
	$(ENV) python scripts/evaluate_code_agent.py \
		--hf-model Qwen/Qwen2.5-Coder-1.5B-Instruct \
		--mode single --scoring-mode verified_agent \
		--agent-contract strict \
		--output outputs/eval_base_only

eval-lora-only:
	$(ENV) python scripts/evaluate_code_agent.py \
		--hf-model $(LORA_PATH) \
		--mode single --scoring-mode verified_agent \
		--agent-contract strict \
		--output outputs/eval_lora_only

eval-memory-only:
	$(ENV) python scripts/evaluate_code_agent.py \
		--hf-model Qwen/Qwen2.5-Coder-1.5B-Instruct \
		--mode single --scoring-mode verified_agent \
		--agent-contract strict \
		--memory-enabled --memory-index $(MEM_INDEX) \
		--output outputs/eval_memory_only

eval-lora-memory:
	$(ENV) python scripts/evaluate_code_agent.py \
		--hf-model $(LORA_PATH) \
		--mode single --scoring-mode verified_agent \
		--agent-contract strict \
		--memory-enabled --memory-index $(MEM_INDEX) \
		--output outputs/eval_lora_memory

eval-lora-memory-best-of-3:
	$(ENV) python scripts/evaluate_code_agent.py \
		--hf-model $(LORA_PATH) \
		--mode best_of_n --n 3 --scoring-mode verified_agent \
		--agent-contract strict \
		--memory-enabled --memory-index $(MEM_INDEX) \
		--output outputs/eval_lora_memory_bon3

# ── Held-out and recovery-stress evaluation ────────────────────────────────
.PHONY: eval-heldout-base eval-heldout-memory \
        eval-heldout-lora-memory-single eval-heldout-lora-memory-best3 \
        eval-recovery-stress-base eval-recovery-stress-memory \
        eval-recovery-stress-lora-memory-single eval-recovery-stress-lora-memory-best3

HELDOUT_FILE  := data/heldout_code_agent_tasks.jsonl
RECOVERY_FILE := data/recovery_stress_tasks.jsonl

eval-heldout-base:
	$(ENV) python scripts/evaluate_code_agent.py \
		--hf-model Qwen/Qwen2.5-Coder-1.5B-Instruct \
		--tasks-file $(HELDOUT_FILE) \
		--mode single --scoring-mode verified_agent \
		--agent-contract strict \
		--stop-after-pass \
		--output outputs/eval_heldout_base

eval-heldout-memory:
	$(ENV) python scripts/evaluate_code_agent.py \
		--hf-model Qwen/Qwen2.5-Coder-1.5B-Instruct \
		--tasks-file $(HELDOUT_FILE) \
		--mode single --scoring-mode verified_agent \
		--agent-contract strict \
		--stop-after-pass \
		--memory-enabled --memory-index $(MEM_INDEX) \
		--output outputs/eval_heldout_memory

eval-heldout-lora-memory-single:
	$(ENV) python scripts/evaluate_code_agent.py \
		--hf-model $(LORA_PATH) \
		--tasks-file $(HELDOUT_FILE) \
		--mode single --scoring-mode verified_agent \
		--agent-contract strict \
		--stop-after-pass \
		--memory-enabled --memory-index $(MEM_INDEX) \
		--output outputs/eval_heldout_lora_memory

eval-heldout-lora-memory-best3:
	$(ENV) python scripts/evaluate_code_agent.py \
		--hf-model $(LORA_PATH) \
		--tasks-file $(HELDOUT_FILE) \
		--mode best_of_n --n 3 --scoring-mode verified_agent \
		--agent-contract strict \
		--stop-after-pass \
		--memory-enabled --memory-index $(MEM_INDEX) \
		--output outputs/eval_heldout_lora_memory_bon3

eval-recovery-stress-base:
	$(ENV) python scripts/evaluate_code_agent.py \
		--hf-model Qwen/Qwen2.5-Coder-1.5B-Instruct \
		--tasks-file $(RECOVERY_FILE) \
		--mode single --scoring-mode verified_agent \
		--agent-contract strict \
		--stop-after-pass \
		--output outputs/eval_recovery_base

eval-recovery-stress-memory:
	$(ENV) python scripts/evaluate_code_agent.py \
		--hf-model Qwen/Qwen2.5-Coder-1.5B-Instruct \
		--tasks-file $(RECOVERY_FILE) \
		--mode single --scoring-mode verified_agent \
		--agent-contract strict \
		--stop-after-pass \
		--memory-enabled --memory-index $(MEM_INDEX) \
		--output outputs/eval_recovery_memory

eval-recovery-stress-lora-memory-single:
	$(ENV) python scripts/evaluate_code_agent.py \
		--hf-model $(LORA_PATH) \
		--tasks-file $(RECOVERY_FILE) \
		--mode single --scoring-mode verified_agent \
		--agent-contract strict \
		--stop-after-pass \
		--memory-enabled --memory-index $(MEM_INDEX) \
		--output outputs/eval_recovery_lora_memory

eval-recovery-stress-lora-memory-best3:
	$(ENV) python scripts/evaluate_code_agent.py \
		--hf-model $(LORA_PATH) \
		--tasks-file $(RECOVERY_FILE) \
		--mode best_of_n --n 3 --scoring-mode verified_agent \
		--agent-contract strict \
		--stop-after-pass \
		--memory-enabled --memory-index $(MEM_INDEX) \
		--output outputs/eval_recovery_lora_memory_bon3

# ── Memory-augmented training ──────────────────────────────────────────────
.PHONY: train-memory-smoke train-memory-25pilot eval-memory-25pilot train-memory-300

MEMORY_PILOT_OUT := outputs/qwen15b_memory_25pilot
MEMORY_SMOKE_OUT := outputs/qwen15b_memory_smoke

train-memory-smoke:
	$(ENV) python scripts/finetune_qwen_code_agent.py \
		--hf-model Qwen/Qwen2.5-Coder-1.5B-Instruct \
		--test-run \
		--steps 5 \
		--lr 6e-6 \
		--batch-size 1 \
		--grad-accum 16 \
		--max-length 1024 \
		--agent-only \
		--agent-contract strict \
		--memory-enabled \
		--memory-index $(MEM_INDEX) \
		--memory-top-k 4 \
		--output-dir $(MEMORY_SMOKE_OUT)

train-memory-25pilot:
	$(ENV) python scripts/finetune_qwen_code_agent.py \
		--hf-model Qwen/Qwen2.5-Coder-1.5B-Instruct \
		--steps 25 \
		--lr 6e-6 \
		--batch-size 1 \
		--grad-accum 16 \
		--max-length 1024 \
		--agent-only \
		--agent-contract strict \
		--memory-enabled \
		--memory-index $(MEM_INDEX) \
		--memory-top-k 4 \
		--output-dir $(MEMORY_PILOT_OUT)

eval-memory-25pilot:
	$(ENV) python scripts/evaluate_code_agent.py \
		--hf-model $(MEMORY_PILOT_OUT)/final \
		--mode single --scoring-mode verified_agent \
		--agent-contract strict \
		--memory-enabled --memory-index $(MEM_INDEX) \
		--output outputs/eval_memory_25pilot

train-memory-300:
	$(ENV) python scripts/finetune_qwen_code_agent.py \
		--hf-model Qwen/Qwen2.5-Coder-1.5B-Instruct \
		--steps 300 \
		--lr 6e-6 \
		--batch-size 1 \
		--grad-accum 16 \
		--max-length 1024 \
		--agent-only \
		--agent-contract strict \
		--memory-enabled \
		--memory-index $(MEM_INDEX) \
		--memory-top-k 4 \
		--output-dir outputs/qwen15b_memory_300

# ── Targeted development set (5 persistent frozen held-out failure categories) ─
# Scientific hygiene:
#   - Tasks in data/targeted_failure_dev_tasks.jsonl are similar-but-not-identical
#     to frozen held-out tasks — they are NOT copies.
#   - Train a targeted pilot on the dev-set traces; re-evaluate on the frozen
#     held-out benchmark with the ORIGINAL 82-record memory index only.
#   - The adapted-memory result (82.1%) is a separate experiment; do not mix them.
.PHONY: test-targeted-dev-tasks eval-targeted-dev-base eval-targeted-dev-memory \
        eval-targeted-dev-lora-memory generate-targeted-dev-traces

TARGETED_DEV_FILE := data/targeted_failure_dev_tasks.jsonl
TARGETED_DEV_OUT  := outputs/qwen15b_targeted_failure_dev_150steps

test-targeted-dev-tasks:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $(ENV) python -m pytest \
		tests/test_targeted_failure_dev_tasks.py -v

eval-targeted-dev-base:
	$(ENV) python scripts/evaluate_code_agent.py \
		--hf-model Qwen/Qwen2.5-Coder-1.5B-Instruct \
		--tasks-file $(TARGETED_DEV_FILE) \
		--mode single --scoring-mode verified_agent \
		--agent-contract strict \
		--stop-after-pass \
		--output outputs/eval_targeted_dev_base

eval-targeted-dev-memory:
	$(ENV) python scripts/evaluate_code_agent.py \
		--hf-model Qwen/Qwen2.5-Coder-1.5B-Instruct \
		--tasks-file $(TARGETED_DEV_FILE) \
		--mode single --scoring-mode verified_agent \
		--agent-contract strict \
		--stop-after-pass \
		--memory-enabled --memory-index $(MEM_INDEX) \
		--output outputs/eval_targeted_dev_memory

eval-targeted-dev-lora-memory:
	$(ENV) python scripts/evaluate_code_agent.py \
		--hf-model $(LORA_PATH) \
		--tasks-file $(TARGETED_DEV_FILE) \
		--mode single --scoring-mode verified_agent \
		--agent-contract strict \
		--stop-after-pass \
		--memory-enabled --memory-index $(MEM_INDEX) \
		--output outputs/eval_targeted_dev_lora_memory

generate-targeted-dev-traces:
	$(ENV) python scripts/generate_code_data.py --dev-set-only
	@echo "Generated dev-set failure→fix traces: data/dev_set_data.jsonl"
	@echo "Next: make train-targeted-failure-pilot"

# ── Targeted failure pilot training (150 steps from existing LoRA checkpoint) ──
.PHONY: train-targeted-failure-pilot eval-frozen-heldout-after-targeted

train-targeted-failure-pilot: generate-targeted-dev-traces
	$(ENV) python scripts/finetune_qwen_code_agent.py \
		--hf-model $(LORA_PATH) \
		--training-file data/dev_set_data.jsonl \
		--steps 150 \
		--lr 3e-6 \
		--batch-size 1 \
		--grad-accum 16 \
		--max-length 1024 \
		--agent-only \
		--agent-contract strict \
		--memory-enabled \
		--memory-index $(MEM_INDEX) \
		--memory-top-k 4 \
		--output-dir $(TARGETED_DEV_OUT)

eval-frozen-heldout-after-targeted:
	$(ENV) python scripts/evaluate_code_agent.py \
		--hf-model $(TARGETED_DEV_OUT)/final \
		--tasks-file $(HELDOUT_FILE) \
		--mode best_of_n --n 3 --scoring-mode verified_agent \
		--agent-contract strict \
		--stop-after-pass \
		--memory-enabled --memory-index $(MEM_INDEX) \
		--output outputs/eval_frozen_heldout_after_targeted_dev_150steps \
		--verbose

# ── Option A: merge 300-step LoRA → train one fresh adapter ──────────────
# Goal: beat 75.0% frozen held-out without LoRA-on-LoRA stacking.
# Always use memory/index (clean baseline), never memory/index_adapted.
# Result: rejected — 64.3% vs 75.0% baseline (negative experiment).

OPTION_A_BASE    := outputs/qwen15b_merged_base
OPTION_A_OUT     := outputs/qwen15b_fresh_blended_350
OPTION_A_EVAL    := outputs/eval_frozen_heldout_option_a_350
BLENDED_DATA     := data/dev_set_blended.jsonl

.PHONY: merge-option-a-lora build-option-a-blended-data train-fresh-blended \
        eval-frozen-heldout-option-a summarise-option-a

# Step 1: merge champion LoRA into a plain HF model
merge-option-a-lora:
	$(ENV) python scripts/merge_lora.py \
		--lora-path $(LORA_PATH) \
		--output-dir $(OPTION_A_BASE) \
		--dtype bfloat16
	@echo "Merged model saved to $(OPTION_A_BASE). Verify it loads before training."

# Step 2: build blended training file (general + targeted traces)
build-option-a-blended-data:
	$(ENV) python scripts/build_option_a_blended_data.py \
		--general-file data/agent_only_data.jsonl \
		--targeted-file data/dev_set_failing_5.jsonl \
		--fallback-targeted data/dev_set_data.jsonl \
		--output $(BLENDED_DATA)
	wc -l $(BLENDED_DATA)

# Step 3: train ONE fresh LoRA from the merged model (no LoRA-on-LoRA)
train-fresh-blended:
	@test -d $(OPTION_A_BASE) || (echo "ERROR: $(OPTION_A_BASE) not found. Run make merge-option-a-lora first." && exit 1)
	@test -s $(BLENDED_DATA)  || (echo "ERROR: $(BLENDED_DATA) empty/missing. Run make build-option-a-blended-data first." && exit 1)
	mkdir -p outputs/option_a_logs
	$(ENV) python scripts/finetune_qwen_code_agent.py \
		--hf-model $(OPTION_A_BASE) \
		--training-file $(BLENDED_DATA) \
		--steps 350 \
		--lr 5e-6 \
		--batch-size 1 \
		--grad-accum 16 \
		--max-length 1024 \
		--agent-only \
		--agent-contract strict \
		--memory-enabled \
		--memory-index $(MEM_INDEX) \
		--memory-top-k 4 \
		--output-dir $(OPTION_A_OUT) \
		2>&1 | tee outputs/option_a_logs/train_fresh_blended_350.log

# Step 4: evaluate on clean frozen held-out using original memory/index only
eval-frozen-heldout-option-a:
	@test -d $(OPTION_A_OUT)/final || (echo "ERROR: $(OPTION_A_OUT)/final not found. Run make train-fresh-blended first." && exit 1)
	$(ENV) python scripts/evaluate_code_agent.py \
		--hf-model $(OPTION_A_OUT)/final \
		--tasks-file data/heldout_code_agent_tasks.jsonl \
		--mode best_of_n --n 3 \
		--scoring-mode verified_agent \
		--agent-contract strict \
		--stop-after-pass \
		--memory-enabled --memory-index $(MEM_INDEX) \
		--output $(OPTION_A_EVAL) \
		--verbose

# Step 5: compare Option A against the 75.0% clean baseline
summarise-option-a:
	$(ENV) python scripts/summarise_option_a.py \
		--baseline outputs/current_rerun_20260620_075110/eval_heldout_lora_memory_bon3/best_of_3.csv \
		--option-a $(OPTION_A_EVAL)/best_of_3.csv

# ── v2.6 Data Mixture and Trace-Gating Audit ─────────────────────────────
# Ablation: find the optimal execution-trace fraction in the blended dataset.
# v2.5 used 100% of traces (57% of blend) and regressed to 53.6%.
# Hard tasks improved (+24pp) but string/basic collapsed (-50pp, -100pp).
# This ablation tests 0%, 10%, 25%, 50%, 100% trace fractions.
# Base pool: 2000 general + 200 failure + 82 memory = 2282 fixed examples.
# All runs start from outputs/qwen15b_merged_base (no LoRA-on-LoRA).
# Evaluate each on frozen held-out. Promote only if >= 75.0%.

V26_BASE     := outputs/qwen15b_merged_base
V26_BASELINE := outputs/current_rerun_20260620_075110/eval_heldout_lora_memory_bon3/best_of_3.csv

.PHONY: build-v26-blends \
        train-v26-traces000 train-v26-traces010 train-v26-traces025 \
        train-v26-traces050 train-v26-traces100 \
        eval-v26-traces000 eval-v26-traces010 eval-v26-traces025 \
        eval-v26-traces050 eval-v26-traces100 \
        summarise-v26

build-v26-blends:
	$(ENV) python scripts/build_v26_trace_blend.py --trace-ratio 0.00
	$(ENV) python scripts/build_v26_trace_blend.py --trace-ratio 0.10
	$(ENV) python scripts/build_v26_trace_blend.py --trace-ratio 0.25
	$(ENV) python scripts/build_v26_trace_blend.py --trace-ratio 0.50
	$(ENV) python scripts/build_v26_trace_blend.py --trace-ratio 1.00
	@echo "All 5 blends written to data/v26_blend_traces*.jsonl"
	wc -l data/v26_blend_traces*.jsonl

define V26_TRAIN_RULE
train-v26-traces$(1):
	@test -d $(V26_BASE) || (echo "ERROR: $(V26_BASE) missing — run make merge-option-a-lora" && exit 1)
	@test -s data/v26_blend_traces$(1)pct.jsonl || (echo "ERROR: data/v26_blend_traces$(1)pct.jsonl missing — run make build-v26-blends" && exit 1)
	$$(ENV) python scripts/finetune_qwen_code_agent.py \
		--hf-model      $(V26_BASE) \
		--training-file data/v26_blend_traces$(1)pct.jsonl \
		--steps         300 \
		--lr            2e-5 \
		--batch-size    1 \
		--grad-accum    16 \
		--max-length    1024 \
		--agent-only \
		--agent-contract strict \
		--memory-enabled \
		--memory-index  $(MEM_INDEX) \
		--memory-top-k  4 \
		--output-dir    outputs/qwen15b_v26_traces$(1)pct_300

eval-v26-traces$(1):
	@test -d outputs/qwen15b_v26_traces$(1)pct_300/final || (echo "ERROR: train traces$(1) first" && exit 1)
	$$(ENV) python scripts/evaluate_code_agent.py \
		--hf-model      outputs/qwen15b_v26_traces$(1)pct_300/final \
		--tasks-file    data/heldout_code_agent_tasks.jsonl \
		--mode          best_of_n --n 3 \
		--scoring-mode  verified_agent \
		--agent-contract strict \
		--stop-after-pass \
		--memory-enabled --memory-index $(MEM_INDEX) \
		--output        outputs/eval_frozen_heldout_v26_traces$(1)pct \
		--verbose
endef

$(eval $(call V26_TRAIN_RULE,000))
$(eval $(call V26_TRAIN_RULE,010))
$(eval $(call V26_TRAIN_RULE,025))
$(eval $(call V26_TRAIN_RULE,050))
$(eval $(call V26_TRAIN_RULE,100))

summarise-v26:
	@echo "=== v2.6 Trace-Gating Ablation Results ==="
	@for pct in 000 010 025 050 100; do \
	  csv=outputs/eval_frozen_heldout_v26_traces$${pct}pct/best_of_3.csv; \
	  if [ -f "$$csv" ]; then \
	    result=$$($(ENV) python -c "import csv; rows=list(csv.DictReader(open('$$csv'))); p=sum(1 for r in rows if r.get('passed','').strip() in ('True','true','1')); print(f'{p}/{len(rows)}={p*100//len(rows)}%')"); \
	    echo "  traces $${pct}%:  $$result"; \
	  else \
	    echo "  traces $${pct}%:  not yet evaluated"; \
	  fi; \
	done
	@echo "  champion (v0.1): 21/28 = 75.0%"
	@echo "  v2.5 (100% traces): 15/28 = 53.6%  [rejected]"

# ── v2.7 Champion Preservation Audit ─────────────────────────────────────
# Goal: find why the 75.0% champion is not preserved by merge-and-retrain.
# DO NOT train a new model in this milestone — audit only.
#
# Champion: outputs/qwen15b_memory_300steps/final (300-step LoRA, Qwen2.5-Coder-1.5B)
# Known results:
#   Champion 300-step LoRA + memory : 21/28 = 75.0%
#   v2.6 traces000                  : 16/28 = 57.1%
#   v2.6 traces010                  : 14/28 = 50.0%
#   v2.6 traces025                  : 15/28 = 53.6%
#
# Run order:
#   1. make eval-v27-champion-adapter       (does the champion still reproduce 75.0%?)
#   2. make merge-v27-champion              (create merged standalone model)
#   3. make eval-v27-merged-champion        (does merge_and_unload damage the model?)
#   4. make eval-v27-champion-no-memory     (how much does memory contribute?)
#   5. make summarise-v27-preservation      (produces the audit report)
#
# Optional environment cross-checks:
#   make eval-v27-champion-adapter-mltorch      (ml-torch env)
#   make eval-v27-champion-adapter-aetherforge  (aetherforge-train env, same as default)

V27_CHAMPION_LORA   := outputs/qwen15b_memory_300steps/final
V27_CHAMPION_MERGED := outputs/qwen15b_v27_champion_merged
V27_HELDOUT         := data/heldout_code_agent_tasks.jsonl
V27_OUT_DIR         := results/v27_champion_preservation

.PHONY: eval-v27-champion-adapter \
        eval-v27-champion-adapter-mltorch \
        eval-v27-champion-adapter-aetherforge \
        merge-v27-champion \
        eval-v27-merged-champion \
        eval-v27-champion-no-memory \
        eval-v27-champion-original-memory \
        summarise-v27-preservation

# Control 1: original champion adapter (unmerged), best-of-3, with memory/index.
# This is the canonical re-evaluation.  Expected: 21/28 = 75.0%.
eval-v27-champion-adapter:
	@test -d $(V27_CHAMPION_LORA) || (echo "ERROR: $(V27_CHAMPION_LORA) not found." && exit 1)
	$(ENV) python scripts/evaluate_code_agent.py \
		--hf-model      $(V27_CHAMPION_LORA) \
		--tasks-file    $(V27_HELDOUT) \
		--mode          best_of_n --n 3 \
		--scoring-mode  verified_agent \
		--agent-contract strict \
		--stop-after-pass \
		--memory-enabled --memory-index $(MEM_INDEX) \
		--memory-top-k   4 \
		--output         outputs/eval_v27_champion_adapter \
		--verbose

# Control 2a: same adapter, forced ml-torch environment.
# Tests whether env/CUDA/PyTorch version changes the result.
eval-v27-champion-adapter-mltorch:
	@test -d $(V27_CHAMPION_LORA) || (echo "ERROR: $(V27_CHAMPION_LORA) not found." && exit 1)
	conda run -n ml-torch python scripts/evaluate_code_agent.py \
		--hf-model      $(V27_CHAMPION_LORA) \
		--tasks-file    $(V27_HELDOUT) \
		--mode          best_of_n --n 3 \
		--scoring-mode  verified_agent \
		--agent-contract strict \
		--stop-after-pass \
		--memory-enabled --memory-index $(MEM_INDEX) \
		--memory-top-k   4 \
		--output         outputs/eval_v27_champion_adapter_mltorch \
		--verbose

# Control 2b: same adapter, explicit aetherforge-train environment.
eval-v27-champion-adapter-aetherforge:
	@test -d $(V27_CHAMPION_LORA) || (echo "ERROR: $(V27_CHAMPION_LORA) not found." && exit 1)
	conda run -n aetherforge-train python scripts/evaluate_code_agent.py \
		--hf-model      $(V27_CHAMPION_LORA) \
		--tasks-file    $(V27_HELDOUT) \
		--mode          best_of_n --n 3 \
		--scoring-mode  verified_agent \
		--agent-contract strict \
		--stop-after-pass \
		--memory-enabled --memory-index $(MEM_INDEX) \
		--memory-top-k   4 \
		--output         outputs/eval_v27_champion_adapter_aetherforge \
		--verbose

# Control 3: merge champion LoRA into a standalone HF model (no retraining).
# If eval-v27-merged-champion matches the adapter result, merge is safe.
# If it drops, the root cause is merge precision / tokenizer / config drift.
merge-v27-champion:
	@test -d $(V27_CHAMPION_LORA) || (echo "ERROR: $(V27_CHAMPION_LORA) not found." && exit 1)
	$(ENV) python scripts/merge_lora.py \
		--lora-path  $(V27_CHAMPION_LORA) \
		--output-dir $(V27_CHAMPION_MERGED) \
		--dtype      bfloat16
	@echo "Merged model saved to $(V27_CHAMPION_MERGED). Verify it loads before eval."

# Control 4: evaluate merged champion WITHOUT retraining.
# Uses the same eval settings as the champion adapter run.
eval-v27-merged-champion:
	@test -d $(V27_CHAMPION_MERGED) || \
		(echo "ERROR: $(V27_CHAMPION_MERGED) not found — run make merge-v27-champion first." && exit 1)
	$(ENV) python scripts/evaluate_code_agent.py \
		--hf-model      $(V27_CHAMPION_MERGED) \
		--tasks-file    $(V27_HELDOUT) \
		--mode          best_of_n --n 3 \
		--scoring-mode  verified_agent \
		--agent-contract strict \
		--stop-after-pass \
		--memory-enabled --memory-index $(MEM_INDEX) \
		--memory-top-k   4 \
		--output         outputs/eval_v27_merged_champion \
		--verbose

# Control 5: champion adapter WITH MEMORY DISABLED.
# If this drops significantly below 75.0%, the 75.0% result is
# a model+memory system result, not pure adapter generalisation.
eval-v27-champion-no-memory:
	@test -d $(V27_CHAMPION_LORA) || (echo "ERROR: $(V27_CHAMPION_LORA) not found." && exit 1)
	$(ENV) python scripts/evaluate_code_agent.py \
		--hf-model      $(V27_CHAMPION_LORA) \
		--tasks-file    $(V27_HELDOUT) \
		--mode          best_of_n --n 3 \
		--scoring-mode  verified_agent \
		--agent-contract strict \
		--stop-after-pass \
		--output         outputs/eval_v27_champion_no_memory \
		--verbose

# Control 6: champion adapter with original memory/index (explicit).
# Same as eval-v27-champion-adapter but named to show the index is the
# original clean index, not memory/index_adapted.
eval-v27-champion-original-memory:
	@test -d $(V27_CHAMPION_LORA) || (echo "ERROR: $(V27_CHAMPION_LORA) not found." && exit 1)
	$(ENV) python scripts/evaluate_code_agent.py \
		--hf-model      $(V27_CHAMPION_LORA) \
		--tasks-file    $(V27_HELDOUT) \
		--mode          best_of_n --n 3 \
		--scoring-mode  verified_agent \
		--agent-contract strict \
		--stop-after-pass \
		--memory-enabled --memory-index memory/index \
		--memory-top-k   4 \
		--output         outputs/eval_v27_champion_original_memory \
		--verbose

# Produce the preservation report from all available eval CSVs.
# Safe to run at any point — missing CSVs are reported as "not yet evaluated".
summarise-v27-preservation:
	mkdir -p $(V27_OUT_DIR)
	$(ENV) python scripts/summarise_v27_preservation.py \
		--champion-csv        outputs/eval_v27_champion_adapter/best_of_3.csv \
		--mltorch-csv         outputs/eval_v27_champion_adapter_mltorch/best_of_3.csv \
		--aetherforge-csv     outputs/eval_v27_champion_adapter_aetherforge/best_of_3.csv \
		--merged-csv          outputs/eval_v27_merged_champion/best_of_3.csv \
		--no-memory-csv       outputs/eval_v27_champion_no_memory/best_of_3.csv \
		--original-memory-csv outputs/eval_v27_champion_original_memory/best_of_3.csv \
		--output-dir          $(V27_OUT_DIR)
	@echo ""
	@echo "Audit report: $(V27_OUT_DIR)/summary.md"
	@echo "Per-task CSV: $(V27_OUT_DIR)/per_task_comparison.csv"
	@echo "Failure diff: $(V27_OUT_DIR)/failure_diff.md"

# ── v2.8 Champion System Enhancement ─────────────────────────────────────
# Goal: improve 82.1% (23/28) merged champion + memory system without retraining.
# Promotion rule: >= 24/28 = new champion; == 23/28 = tie; < 23/28 = reject.
#
# Current best: outputs/qwen15b_v27_champion_merged + memory/index_adapted
# Failing 5: group_anagrams, merge_intervals, count_islands,
#             median_two_sorted, tree_depth_tuple
#
# Run order:
#   1. make eval-v28-current-champion       (reproduce 23/28 baseline)
#   2. make eval-v28-no-memory              (control: memory lift)
#   3. make eval-v28-memory-topk1           |
#      make eval-v28-memory-topk3           | top-k ablation
#      make eval-v28-memory-topk5           |
#   4. make eval-v28-filtered-memory        (curated hard/medium-only index)
#   5. make eval-v28-direct-answer-prompt   (DIRECT_ANSWER_SYSTEM prompt)
#   6. make eval-v28-continuation-logic     (best-of-5 on 5 failing tasks)
#   7. make summarise-v28

V28_CHAMPION  := outputs/qwen15b_v27_champion_merged
V28_MEM_INDEX := memory/index_adapted
V28_HELDOUT   := data/heldout_code_agent_tasks.jsonl
V28_OUT_DIR   := results/v28_champion_system

.PHONY: eval-v28-current-champion \
        eval-v28-no-memory \
        eval-v28-memory-topk1 \
        eval-v28-memory-topk3 \
        eval-v28-memory-topk5 \
        eval-v28-filtered-memory \
        eval-v28-direct-answer-prompt \
        eval-v28-continuation-logic \
        summarise-v28

# Baseline: reproduce the 23/28 merged champion + memory result.
eval-v28-current-champion:
	@test -d $(V28_CHAMPION) || \
		(echo "ERROR: $(V28_CHAMPION) not found — run make merge-v27-champion first" && exit 1)
	$(ENV) python scripts/evaluate_code_agent.py \
		--hf-model      $(V28_CHAMPION) \
		--tasks-file    $(V28_HELDOUT) \
		--mode          best_of_n --n 3 \
		--scoring-mode  verified_agent \
		--agent-contract strict \
		--stop-after-pass \
		--memory-enabled --memory-index $(V28_MEM_INDEX) \
		--memory-top-k   4 \
		--output         outputs/eval_v28_current_champion \
		--verbose

# Control: merged champion WITHOUT memory.
eval-v28-no-memory:
	@test -d $(V28_CHAMPION) || \
		(echo "ERROR: $(V28_CHAMPION) not found — run make merge-v27-champion first" && exit 1)
	$(ENV) python scripts/evaluate_code_agent.py \
		--hf-model      $(V28_CHAMPION) \
		--tasks-file    $(V28_HELDOUT) \
		--mode          best_of_n --n 3 \
		--scoring-mode  verified_agent \
		--agent-contract strict \
		--stop-after-pass \
		--output         outputs/eval_v28_no_memory \
		--verbose

# top-k ablation: 1, 3, 5 retrieved examples.
eval-v28-memory-topk1:
	@test -d $(V28_CHAMPION) || \
		(echo "ERROR: $(V28_CHAMPION) not found — run make merge-v27-champion first" && exit 1)
	$(ENV) python scripts/evaluate_code_agent.py \
		--hf-model      $(V28_CHAMPION) \
		--tasks-file    $(V28_HELDOUT) \
		--mode          best_of_n --n 3 \
		--scoring-mode  verified_agent \
		--agent-contract strict \
		--stop-after-pass \
		--memory-enabled --memory-index $(V28_MEM_INDEX) \
		--memory-top-k   1 \
		--output         outputs/eval_v28_memory_topk1 \
		--verbose

eval-v28-memory-topk3:
	@test -d $(V28_CHAMPION) || \
		(echo "ERROR: $(V28_CHAMPION) not found — run make merge-v27-champion first" && exit 1)
	$(ENV) python scripts/evaluate_code_agent.py \
		--hf-model      $(V28_CHAMPION) \
		--tasks-file    $(V28_HELDOUT) \
		--mode          best_of_n --n 3 \
		--scoring-mode  verified_agent \
		--agent-contract strict \
		--stop-after-pass \
		--memory-enabled --memory-index $(V28_MEM_INDEX) \
		--memory-top-k   3 \
		--output         outputs/eval_v28_memory_topk3 \
		--verbose

eval-v28-memory-topk5:
	@test -d $(V28_CHAMPION) || \
		(echo "ERROR: $(V28_CHAMPION) not found — run make merge-v27-champion first" && exit 1)
	$(ENV) python scripts/evaluate_code_agent.py \
		--hf-model      $(V28_CHAMPION) \
		--tasks-file    $(V28_HELDOUT) \
		--mode          best_of_n --n 3 \
		--scoring-mode  verified_agent \
		--agent-contract strict \
		--stop-after-pass \
		--memory-enabled --memory-index $(V28_MEM_INDEX) \
		--memory-top-k   5 \
		--output         outputs/eval_v28_memory_topk5 \
		--verbose

# Filtered memory: curated hard/medium-only index (see configs/v28_memory_retrieval.yaml).
# Build index before running: python scripts/build_vector_memory.py
#   --raw-dir memory/raw_adapted --index-dir memory/index_v28_filtered
eval-v28-filtered-memory:
	@test -d $(V28_CHAMPION) || \
		(echo "ERROR: $(V28_CHAMPION) not found — run make merge-v27-champion first" && exit 1)
	@test -d memory/index_v28_filtered || \
		(echo "ERROR: memory/index_v28_filtered not found — build the filtered index first" && exit 1)
	$(ENV) python scripts/evaluate_code_agent.py \
		--hf-model      $(V28_CHAMPION) \
		--tasks-file    $(V28_HELDOUT) \
		--mode          best_of_n --n 3 \
		--scoring-mode  verified_agent \
		--agent-contract strict \
		--stop-after-pass \
		--memory-enabled --memory-index memory/index_v28_filtered \
		--memory-top-k   4 \
		--output         outputs/eval_v28_filtered_memory \
		--verbose

# Direct-answer prompt: uses DIRECT_ANSWER_SYSTEM instead of STRICT_SYSTEM.
eval-v28-direct-answer-prompt:
	@test -d $(V28_CHAMPION) || \
		(echo "ERROR: $(V28_CHAMPION) not found — run make merge-v27-champion first" && exit 1)
	$(ENV) python scripts/evaluate_code_agent.py \
		--hf-model      $(V28_CHAMPION) \
		--tasks-file    $(V28_HELDOUT) \
		--mode          best_of_n --n 3 \
		--scoring-mode  verified_agent \
		--agent-contract strict \
		--stop-after-pass \
		--memory-enabled --memory-index $(V28_MEM_INDEX) \
		--memory-top-k   4 \
		--prompt-variant direct_answer \
		--output         outputs/eval_v28_direct_answer_prompt \
		--verbose

# Continuation logic: best-of-5 sampling focused on the 5 failing tasks.
# Tests whether increased sampling resolves the capability gap.
eval-v28-continuation-logic:
	@test -d $(V28_CHAMPION) || \
		(echo "ERROR: $(V28_CHAMPION) not found — run make merge-v27-champion first" && exit 1)
	$(ENV) python scripts/evaluate_code_agent.py \
		--hf-model      $(V28_CHAMPION) \
		--tasks-file    $(V28_HELDOUT) \
		--task-ids      group_anagrams merge_intervals count_islands \
		                median_two_sorted tree_depth_tuple \
		--mode          best_of_n --n 5 \
		--scoring-mode  verified_agent \
		--agent-contract strict \
		--stop-after-pass \
		--memory-enabled --memory-index $(V28_MEM_INDEX) \
		--memory-top-k   4 \
		--output         outputs/eval_v28_continuation_logic \
		--verbose

# Produce the v2.8 enhancement report.
summarise-v28:
	mkdir -p $(V28_OUT_DIR)
	$(ENV) python scripts/summarise_v28_champion_system.py \
		--output-dir $(V28_OUT_DIR)
	@echo ""
	@echo "Summary : $(V28_OUT_DIR)/summary.md"
	@echo "Per-task: $(V28_OUT_DIR)/per_task_comparison.csv"
	@echo "Failures: $(V28_OUT_DIR)/failure_analysis.md"

# ── v2.9 Memory Repair Split ──────────────────────────────────────────────
# Architecture:
#   Champion index  : memory/index_adapted          (99 records, PROTECTED)
#   Repair raw      : memory/raw_v29_repair          (4 new verified records)
#   Repair index    : memory/index_v29_repair        (champion + repair, 103 records)
#   Diagnostic eval : same 28-task benchmark + repair index  (NOT a clean result)
#   Clean eval      : data/v29_clean_generalisation_tasks.jsonl + repair index
#
# Promotion rule:
#   Diagnostic score (any) on original benchmark  → DIAGNOSTIC label only
#   Clean generalisation score                     → valid generalisation claim
#   New 28-task champion                           → requires merging repair records
#                                                    into index_adapted AND fresh eval
#
# tree_depth_tuple note: the task prompt has a broken assertion
#   (claims ==3 for a case where the correct answer is 4).
#   Repair record uses the correct value. Eval results for that task may vary.

.PHONY: inspect-v29-retrieval build-v29-repair-memory \
        eval-v29-repair-memory-diagnostic eval-v29-clean-memory-generalisation \
        summarise-v29

V29_CHAMPION  := outputs/qwen15b_v27_champion_merged
V29_MEM_INDEX := memory/index_adapted
V29_REPAIR_RAW  := memory/raw_v29_repair
V29_REPAIR_IDX  := memory/index_v29_repair
V29_HELDOUT   := data/heldout_code_agent_tasks.jsonl
V29_CLEAN_SET := data/v29_clean_generalisation_tasks.jsonl
V29_OUT_DIR   := results/v29_memory_repair

# ── inspect-v29-retrieval ─────────────────────────────────────────────────
# Show k=4 hits for the 4 failing tasks from both champion and repair indexes.
# Writes results/v29_memory_repair/retrieval_inspection.md
inspect-v29-retrieval: $(V29_REPAIR_RAW)/repair_records.jsonl
	@test -d $(V29_MEM_INDEX) || (echo "ERROR: champion index not found at $(V29_MEM_INDEX)" && exit 1)
	@mkdir -p $(V29_OUT_DIR)
	$(ENV) python scripts/inspect_v29_retrieval.py \
		--champion-index $(V29_MEM_INDEX) \
		--repair-raw-dir $(V29_REPAIR_RAW) \
		--repair-index   $(V29_REPAIR_IDX) \
		--output-md      $(V29_OUT_DIR)/retrieval_inspection.md \
		--top-k 4
	@echo "Inspection: $(V29_OUT_DIR)/retrieval_inspection.md"

# ── build-v29-repair-memory ───────────────────────────────────────────────
# Validate repair records and build memory/index_v29_repair
# (champion records + 4 repair examples, champion index untouched)
build-v29-repair-memory: $(V29_REPAIR_RAW)/repair_records.jsonl
	$(ENV) python scripts/inspect_v29_retrieval.py \
		--champion-index $(V29_MEM_INDEX) \
		--repair-raw-dir $(V29_REPAIR_RAW) \
		--repair-index   $(V29_REPAIR_IDX) \
		--output-md      $(V29_OUT_DIR)/retrieval_inspection.md \
		--rebuild-repair-index \
		--top-k 4
	@echo "Repair index: $(V29_REPAIR_IDX)"
	@echo "Inspection  : $(V29_OUT_DIR)/retrieval_inspection.md"

# ── eval-v29-repair-memory-diagnostic ────────────────────────────────────
# Evaluate merged champion + repair index on the original 28-task benchmark.
# DIAGNOSTIC LABEL: adds task-specific repair records for known failing tasks.
# Score on this run is NOT a clean champion promotion.
eval-v29-repair-memory-diagnostic: build-v29-repair-memory
	@test -d $(V29_CHAMPION) || (echo "ERROR: champion model not found at $(V29_CHAMPION)" && exit 1)
	@test -f $(V29_HELDOUT)  || (echo "ERROR: heldout tasks not found at $(V29_HELDOUT)" && exit 1)
	@mkdir -p $(V29_OUT_DIR)
	$(ENV) python scripts/evaluate_code_agent.py \
		--hf-model $(V29_CHAMPION) \
		--tasks-file $(V29_HELDOUT) \
		--mode best_of_n --n 3 \
		--scoring-mode verified_agent \
		--agent-contract strict \
		--stop-after-pass \
		--memory-enabled \
		--memory-index $(V29_REPAIR_IDX) \
		--memory-top-k 4 \
		--output outputs/eval_v29_repair_memory_diagnostic \
		--verbose
	@echo "DIAGNOSTIC — score on original 28-task benchmark with repair records added."
	@echo "NOT a clean champion promotion. See $(V29_OUT_DIR)/claim_boundary.md"

# ── eval-v29-clean-memory-generalisation ─────────────────────────────────
# Evaluate merged champion + repair index on the clean generalisation set.
# These 5 tasks are similar to the 4 failing tasks but have different names
# and examples — never seen during training or memory repair.
eval-v29-clean-memory-generalisation: build-v29-repair-memory
	@test -d $(V29_CHAMPION) || (echo "ERROR: champion model not found at $(V29_CHAMPION)" && exit 1)
	@test -f $(V29_CLEAN_SET) || (echo "ERROR: clean task set not found at $(V29_CLEAN_SET)" && exit 1)
	@mkdir -p $(V29_OUT_DIR)
	$(ENV) python scripts/evaluate_code_agent.py \
		--hf-model $(V29_CHAMPION) \
		--tasks-file $(V29_CLEAN_SET) \
		--mode best_of_n --n 3 \
		--scoring-mode verified_agent \
		--agent-contract strict \
		--stop-after-pass \
		--memory-enabled \
		--memory-index $(V29_REPAIR_IDX) \
		--memory-top-k 4 \
		--output outputs/eval_v29_clean_generalisation \
		--verbose
	@echo "CLEAN — score on separate untouched test set. Valid generalisation claim."

# ── summarise-v29 ─────────────────────────────────────────────────────────
summarise-v29:
	@mkdir -p $(V29_OUT_DIR)
	$(ENV) python scripts/summarise_v29_memory_repair.py \
		--diagnostic-csv $(V29_OUT_DIR)/diagnostic_repair_results.csv \
		--clean-csv      $(V29_OUT_DIR)/clean_generalisation_results.csv \
		--output-dir     $(V29_OUT_DIR)
	@echo "Summary      : $(V29_OUT_DIR)/summary.md"
	@echo "Claim boundary: $(V29_OUT_DIR)/claim_boundary.md"

# ── v2.9 Repair-Index Promotion (index_adapted_v29) ───────────────────────
# Builds memory/index_adapted_v29 = champion records + 4 repair examples,
# then runs A/B comparison against the clean champion (memory/index_adapted).
#
# Architecture:
#   Lane 1: memory/index_adapted      (99 rec, clean champion baseline)
#   Lane 2: memory/index_adapted_v29  (103 rec, repair-enhanced)
#
# Claim boundary:
#   Lane 2 on 28-task benchmark = REPAIR-INDEX DIAGNOSTIC (not clean champion)
#   Lane 2 on clean gen tasks   = valid generalisation signal

.PHONY: build-v29-adapted-repair-index \
        eval-v29-adapted-repair-index \
        eval-v29-adapted-repair-clean-generalisation \
        summarise-v29-promotion

V29_ADAPTED_IDX := memory/index_adapted_v29
V29_ADAPTED_RAW := memory/raw_adapted_v29

# ── build-v29-adapted-repair-index ───────────────────────────────────────
# Combine memory/raw_adapted + memory/raw_v29_repair/repair_records.jsonl
# into memory/raw_adapted_v29/, then build memory/index_adapted_v29.
# Keeps memory/index_adapted completely untouched.
build-v29-adapted-repair-index:
	@test -d memory/raw_adapted || (echo "ERROR: memory/raw_adapted not found (locally-excluded dir)" && exit 1)
	@test -f $(V29_REPAIR_RAW)/repair_records.jsonl || \
		(echo "ERROR: repair records not found at $(V29_REPAIR_RAW)/repair_records.jsonl" && exit 1)
	@mkdir -p $(V29_ADAPTED_RAW)
	@cp memory/raw_adapted/*.jsonl $(V29_ADAPTED_RAW)/
	@cp $(V29_REPAIR_RAW)/repair_records.jsonl $(V29_ADAPTED_RAW)/v29_repair_records.jsonl
	$(ENV) python scripts/build_vector_memory.py \
		--raw-dir  $(V29_ADAPTED_RAW) \
		--index-dir $(V29_ADAPTED_IDX)
	@echo "Repair-enhanced index: $(V29_ADAPTED_IDX) (103 records)"
	@echo "Champion index stays: $(V29_MEM_INDEX) (UNTOUCHED)"

# ── eval-v29-adapted-repair-index ────────────────────────────────────────
# Run merged champion + repair-enhanced index on the full 28-task benchmark.
# REPAIR-INDEX DIAGNOSTIC: NOT a clean held-out champion.
eval-v29-adapted-repair-index: build-v29-adapted-repair-index
	@test -d $(V29_CHAMPION) || (echo "ERROR: champion model not found at $(V29_CHAMPION)" && exit 1)
	@test -f $(V29_HELDOUT)  || (echo "ERROR: heldout tasks not found at $(V29_HELDOUT)" && exit 1)
	@mkdir -p $(V29_OUT_DIR)
	$(ENV) python scripts/evaluate_code_agent.py \
		--hf-model $(V29_CHAMPION) \
		--tasks-file $(V29_HELDOUT) \
		--mode best_of_n --n 3 \
		--scoring-mode verified_agent \
		--agent-contract strict \
		--stop-after-pass \
		--memory-enabled \
		--memory-index $(V29_ADAPTED_IDX) \
		--memory-top-k 4 \
		--output outputs/eval_v29_adapted_repair_index \
		--verbose
	@echo "REPAIR-INDEX DIAGNOSTIC — not a clean champion. See claim_boundary.md"

# ── eval-v29-adapted-repair-clean-generalisation ─────────────────────────
# Run merged champion + repair-enhanced index on the clean generalisation set.
eval-v29-adapted-repair-clean-generalisation: build-v29-adapted-repair-index
	@test -d $(V29_CHAMPION) || (echo "ERROR: champion model not found at $(V29_CHAMPION)" && exit 1)
	@test -f $(V29_CLEAN_SET) || (echo "ERROR: clean task set not found at $(V29_CLEAN_SET)" && exit 1)
	@mkdir -p $(V29_OUT_DIR)
	$(ENV) python scripts/evaluate_code_agent.py \
		--hf-model $(V29_CHAMPION) \
		--tasks-file $(V29_CLEAN_SET) \
		--mode best_of_n --n 3 \
		--scoring-mode verified_agent \
		--agent-contract strict \
		--stop-after-pass \
		--memory-enabled \
		--memory-index $(V29_ADAPTED_IDX) \
		--memory-top-k 4 \
		--output outputs/eval_v29_adapted_clean_generalisation \
		--verbose
	@echo "CLEAN — valid generalisation signal on separate untouched test set."

# ── summarise-v29-promotion ───────────────────────────────────────────────
summarise-v29-promotion:
	@mkdir -p $(V29_OUT_DIR)
	$(ENV) python scripts/summarise_v29_promotion.py \
		--adapted-v29-28task-csv $(V29_OUT_DIR)/adapted_v29_28task_results.csv \
		--adapted-v29-clean-csv  $(V29_OUT_DIR)/adapted_v29_clean_results.csv \
		--output-dir             $(V29_OUT_DIR)
	@echo "Promotion summary: $(V29_OUT_DIR)/promotion_summary.md"

# ── v2.10 Clean Repair-Generalisation Benchmark ───────────────────────────
# 32 untouched tasks across 5 families, no overlap with frozen 28-task benchmark.
# A/B comparison of champion index vs repair-enhanced index on clean tasks.
#
# Families:
#   interval_merge    — interval merging and scheduling
#   sorted_selection  — sorted-array / median / kth / merge variants
#   nested_dict       — nested dict access / update / traversal
#   tuple_tree        — tuple-tree recursion and structural traversal
#   rle_encoding      — run-length encoding and structural string
#
# Promotion rule:
#   Repair index >= 75% and > champion index on 32 clean tasks
#   = strong evidence repair memory generalises.

.PHONY: eval-v210-clean-champion eval-v210-repair-index summarise-v210

V210_TASKS   := data/v210_clean_repair_generalisation_tasks.jsonl
V210_OUT_DIR := results/v210_clean_repair_generalisation
V210_CHAMP   := outputs/qwen15b_v27_champion_merged

# ── eval-v210-clean-champion ─────────────────────────────────────────────
eval-v210-clean-champion:
	@test -d $(V210_CHAMP)   || (echo "ERROR: champion model not found at $(V210_CHAMP)" && exit 1)
	@test -f $(V210_TASKS)   || (echo "ERROR: v2.10 task file not found at $(V210_TASKS)" && exit 1)
	@mkdir -p $(V210_OUT_DIR)
	$(ENV) python scripts/evaluate_code_agent.py \
		--hf-model $(V210_CHAMP) \
		--tasks-file $(V210_TASKS) \
		--mode best_of_n --n 3 \
		--scoring-mode verified_agent \
		--agent-contract strict \
		--stop-after-pass \
		--memory-enabled \
		--memory-index $(V29_MEM_INDEX) \
		--memory-top-k 4 \
		--output outputs/eval_v210_clean_champion \
		--verbose
	@echo "Champion eval complete. See outputs/eval_v210_clean_champion/"

# ── eval-v210-repair-index ───────────────────────────────────────────────
eval-v210-repair-index: build-v29-adapted-repair-index
	@test -d $(V210_CHAMP)   || (echo "ERROR: champion model not found at $(V210_CHAMP)" && exit 1)
	@test -f $(V210_TASKS)   || (echo "ERROR: v2.10 task file not found at $(V210_TASKS)" && exit 1)
	@mkdir -p $(V210_OUT_DIR)
	$(ENV) python scripts/evaluate_code_agent.py \
		--hf-model $(V210_CHAMP) \
		--tasks-file $(V210_TASKS) \
		--mode best_of_n --n 3 \
		--scoring-mode verified_agent \
		--agent-contract strict \
		--stop-after-pass \
		--memory-enabled \
		--memory-index $(V29_ADAPTED_IDX) \
		--memory-top-k 4 \
		--output outputs/eval_v210_repair_index \
		--verbose
	@echo "Repair-index eval complete. See outputs/eval_v210_repair_index/"

# ── summarise-v210 ───────────────────────────────────────────────────────
summarise-v210:
	@mkdir -p $(V210_OUT_DIR)
	$(ENV) python scripts/summarise_v210.py \
		--champion-csv $(V210_OUT_DIR)/champion_results.csv \
		--repair-csv   $(V210_OUT_DIR)/repair_results.csv \
		--tasks-file   $(V210_TASKS) \
		--output-dir   $(V210_OUT_DIR)
	@echo "Summary       : $(V210_OUT_DIR)/summary.md"
	@echo "Family breakdown: $(V210_OUT_DIR)/per_family_breakdown.md"
	@echo "Claim boundary: $(V210_OUT_DIR)/claim_boundary.md"

# ── v2.11 Retrieval Routing and Gating Audit ──────────────────────────────
# Re-uses the 32 v2.10 clean tasks. Tests three routing strategies:
#   family-router    — repair index only for interval_merge tasks
#   confidence-router — repair index when repair top-1 score > threshold
#   oracle-router    — diagnostic ceiling: choose index that passes per task
#
# Promotion rule:
#   family or confidence router beats champion by >= 5 pp on 32 tasks
#   = strong evidence for retrieval gating.

.PHONY: route-v211 eval-v211-family-router eval-v211-confidence-router \
        eval-v211-oracle-router summarise-v211

V211_TASKS       := data/v210_clean_repair_generalisation_tasks.jsonl
V211_OUT_DIR     := results/v211_retrieval_routing
V211_CHAMP_IDX   := $(V29_MEM_INDEX)
V211_REPAIR_IDX  := $(V29_ADAPTED_IDX)
V211_CHAMP_MODEL := outputs/qwen15b_v27_champion_merged
V211_V210_CHAMP  := results/v210_clean_repair_generalisation/champion_results.csv
V211_V210_REPAIR := results/v210_clean_repair_generalisation/repair_results.csv

# ── route-v211 ───────────────────────────────────────────────────────────
# Generate routing decisions and task sub-files for all routers.
# Depends on repair index being built; champion index is always present.
route-v211: build-v29-adapted-repair-index
	@test -f $(V211_TASKS) || (echo "ERROR: v2.10 tasks not found" && exit 1)
	@test -f $(V211_V210_CHAMP) || (echo "ERROR: v2.10 champion CSV not found — run eval-v210-clean-champion first" && exit 1)
	@test -f $(V211_V210_REPAIR) || (echo "ERROR: v2.10 repair CSV not found — run eval-v210-repair-index first" && exit 1)
	@mkdir -p $(V211_OUT_DIR)
	$(ENV) python scripts/route_v211.py \
		--tasks-file             $(V211_TASKS) \
		--champion-index         $(V211_CHAMP_IDX) \
		--repair-index           $(V211_REPAIR_IDX) \
		--champion-v210-csv      $(V211_V210_CHAMP) \
		--repair-v210-csv        $(V211_V210_REPAIR) \
		--output-dir             $(V211_OUT_DIR) \
		--confidence-threshold   0.35 \
		--margin-threshold       0.05
	@echo "Routing decisions: $(V211_OUT_DIR)/routing_decisions.json"
	@echo "Routing scores:    $(V211_OUT_DIR)/routing_scores.csv"
	@echo "Task sub-files:    $(V211_OUT_DIR)/tasks_family_*.jsonl"
	@echo "                   $(V211_OUT_DIR)/tasks_conf_*.jsonl"

# ── eval-v211-family-router ──────────────────────────────────────────────
eval-v211-family-router: route-v211
	@test -d $(V211_CHAMP_MODEL) || (echo "ERROR: champion model not found" && exit 1)
	@mkdir -p $(V211_OUT_DIR)
	$(ENV) python scripts/evaluate_code_agent.py \
		--hf-model $(V211_CHAMP_MODEL) \
		--tasks-file $(V211_OUT_DIR)/tasks_family_champion.jsonl \
		--mode best_of_n --n 3 \
		--scoring-mode verified_agent \
		--agent-contract strict \
		--stop-after-pass \
		--memory-enabled \
		--memory-index $(V211_CHAMP_IDX) \
		--memory-top-k 4 \
		--output outputs/eval_v211_family_champion \
		--verbose
	$(ENV) python scripts/evaluate_code_agent.py \
		--hf-model $(V211_CHAMP_MODEL) \
		--tasks-file $(V211_OUT_DIR)/tasks_family_repair.jsonl \
		--mode best_of_n --n 3 \
		--scoring-mode verified_agent \
		--agent-contract strict \
		--stop-after-pass \
		--memory-enabled \
		--memory-index $(V211_REPAIR_IDX) \
		--memory-top-k 4 \
		--output outputs/eval_v211_family_repair \
		--verbose
	@echo "Family-router eval complete."
	@echo "  Champion sub-eval: outputs/eval_v211_family_champion/"
	@echo "  Repair sub-eval:   outputs/eval_v211_family_repair/"

# ── eval-v211-confidence-router ──────────────────────────────────────────
eval-v211-confidence-router: route-v211
	@test -d $(V211_CHAMP_MODEL) || (echo "ERROR: champion model not found" && exit 1)
	@mkdir -p $(V211_OUT_DIR)
	$(ENV) python scripts/evaluate_code_agent.py \
		--hf-model $(V211_CHAMP_MODEL) \
		--tasks-file $(V211_OUT_DIR)/tasks_conf_champion.jsonl \
		--mode best_of_n --n 3 \
		--scoring-mode verified_agent \
		--agent-contract strict \
		--stop-after-pass \
		--memory-enabled \
		--memory-index $(V211_CHAMP_IDX) \
		--memory-top-k 4 \
		--output outputs/eval_v211_conf_champion \
		--verbose
	$(ENV) python scripts/evaluate_code_agent.py \
		--hf-model $(V211_CHAMP_MODEL) \
		--tasks-file $(V211_OUT_DIR)/tasks_conf_repair.jsonl \
		--mode best_of_n --n 3 \
		--scoring-mode verified_agent \
		--agent-contract strict \
		--stop-after-pass \
		--memory-enabled \
		--memory-index $(V211_REPAIR_IDX) \
		--memory-top-k 4 \
		--output outputs/eval_v211_conf_repair \
		--verbose
	@echo "Confidence-router eval complete."
	@echo "  Champion sub-eval: outputs/eval_v211_conf_champion/"
	@echo "  Repair sub-eval:   outputs/eval_v211_conf_repair/"

# ── eval-v211-oracle-router ──────────────────────────────────────────────
# Oracle routing uses existing v2.10 results — no new model inference needed.
eval-v211-oracle-router: route-v211
	@echo "Oracle router uses v2.10 results — no new eval needed."
	@echo "Oracle scores are computed in summarise-v211 from routing_scores.csv."
	@echo "Run: make summarise-v211"

# ── summarise-v211 ───────────────────────────────────────────────────────
summarise-v211:
	@mkdir -p $(V211_OUT_DIR)
	$(ENV) python scripts/summarise_v211.py \
		--routing-scores-csv  $(V211_OUT_DIR)/routing_scores.csv \
		--fam-champion-csv    outputs/eval_v211_family_champion/best_of_3.csv \
		--fam-repair-csv      outputs/eval_v211_family_repair/best_of_3.csv \
		--conf-champion-csv   outputs/eval_v211_conf_champion/best_of_3.csv \
		--conf-repair-csv     outputs/eval_v211_conf_repair/best_of_3.csv \
		--tasks-file          $(V211_TASKS) \
		--output-dir          $(V211_OUT_DIR)
	@echo "Summary:         $(V211_OUT_DIR)/summary.md"
	@echo "Family breakdown:$(V211_OUT_DIR)/per_family_breakdown.md"
	@echo "Per-task routing:$(V211_OUT_DIR)/per_task_routing.csv"
	@echo "Claim boundary:  $(V211_OUT_DIR)/claim_boundary.md"

# ── SWE-bench Lite evaluations ────────────────────────────────────────────
# Phase 1: stub format validation.
# Phase 2: real repo-level patch generation.
# Official Docker harness required for verified scores.
# See docs/SWEBENCH_LITE_PLAN.md and docs/SWEBENCH_PHASE2_PLAN.md
.PHONY: eval-swebench-lite-smoke eval-swebench-lite-stub \
        eval-swebench-lite-patchgen-one inspect-swebench-lite-prediction

eval-swebench-lite-smoke:
	$(ENV) python scripts/eval_swebench_lite_smoke.py \
		--model $(LORA_PATH) \
		--limit 3 \
		--output-dir outputs/swebench_lite_smoke
	@echo "Patch generation only; official harness evaluation still required."
	@echo "See docs/SWEBENCH_LITE_PLAN.md"

# Phase 2 stub check — no cloning, no model, validates script and output format.
eval-swebench-lite-stub:
	$(ENV) python scripts/eval_swebench_lite_patchgen.py \
		--limit 1 \
		--stub-only \
		--output-dir outputs/swebench_lite_phase2_stub
	@echo "Stub check complete. See outputs/swebench_lite_phase2_stub/"

# Phase 2 real run — clone repo, run agent with repo tools, produce patch.
eval-swebench-lite-patchgen-one:
	$(ENV) python scripts/eval_swebench_lite_patchgen.py \
		--limit 1 \
		--model $(LORA_PATH) \
		--memory-enabled \
		--memory-index $(MEM_INDEX) \
		--output-dir outputs/swebench_lite_phase2
	@echo "Patch generation only; official harness evaluation still required."
	@echo "See docs/SWEBENCH_PHASE2_PLAN.md"

# Inspect the most recent Phase 2 prediction.
inspect-swebench-lite-prediction:
	$(ENV) python scripts/inspect_swebench_prediction.py

# ── Distillation ──────────────────────────────────────────────────────────
distill-test:
	$(ENV) python scripts/distill_aetherforge.py --test-run

distill:
	$(ENV) python scripts/distill_aetherforge.py \
		--config 128M \
		--temperature 3.0 \
		--alpha 0.7 \
		--steps 5000

# ── Inference server ──────────────────────────────────────────────────────
.PHONY: serve serve-qwen serve-1b chat chat-qwen chat-server

serve:
	$(ENV) python scripts/serve.py \
		--model aetherforge \
		--checkpoint outputs/aetherforge_pretrain/final/model.pt \
		--config 128M

serve-1b:
	$(ENV) python scripts/serve.py \
		--model aetherforge \
		--checkpoint outputs/aetherforge_pretrain/final/model.pt \
		--config 1B-8k

serve-qwen:
	$(ENV) python scripts/serve.py \
		--model qwen \
		--lora-path outputs/qwen25_vl_lora/final

# ── Interactive chat ──────────────────────────────────────────────────────
chat:
	$(ENV) python scripts/chat.py \
		--checkpoint outputs/aetherforge_pretrain/final/model.pt \
		--config 128M

chat-qwen:
	$(ENV) python scripts/chat.py --model qwen

chat-server:
	$(ENV) python scripts/chat.py --server http://localhost:8000

# ── Environment ───────────────────────────────────────────────────────────
.PHONY: env-create env-update

env-create:
	conda env create -f environment.yml

env-update:
	conda env update -f environment.yml --prune

# ── v2.17 Dense Retrieval Pilot ───────────────────────────────────────────
# Addresses v2.11 finding: TF-IDF similarity measures vocabulary co-occurrence,
# not algorithmic relevance. Three failure types (lexical collision, structural
# overlap, repair leak) are tested against dense + hybrid retrieval.
#
# Promotion rule: dense/hybrid must beat champion on 32-task clean benchmark
# by +2 tasks (62.5% → 68.8%) to be considered a clean improvement.
#
# Usage:
#   make build-v217-dense-index      # requires sentence-transformers
#   make eval-v217-tfidf-28          # TF-IDF baseline (28 frozen tasks)
#   make eval-v217-dense-28          # Dense (28 frozen tasks)
#   make eval-v217-hybrid-28         # Hybrid (28 frozen tasks)
#   make eval-v217-tfidf-32          # TF-IDF baseline (32 clean tasks)
#   make eval-v217-dense-32          # Dense (32 clean tasks)
#   make eval-v217-hybrid-32         # Hybrid (32 clean tasks)
#   make summarise-v217              # Tabulate all results

.PHONY: build-v217-dense-index \
        eval-v217-tfidf-28 eval-v217-dense-28 eval-v217-hybrid-28 \
        eval-v217-tfidf-32 eval-v217-dense-32 eval-v217-hybrid-32 \
        summarise-v217

V217_MODEL        := outputs/qwen15b_v27_champion_merged
V217_TFIDF_INDEX  := memory/index_adapted
V217_DENSE_INDEX  := memory/dense_index_adapted
V217_DENSE_MODEL  := sentence-transformers/all-MiniLM-L6-v2
V217_TASKS_28     :=
V217_TASKS_32     := data/v210_clean_repair_generalisation_tasks.jsonl
V217_OUT_DIR      := results/v217_dense_retrieval
V217_RERANK_N     := 20

# Build dense index from frozen champion TF-IDF index
build-v217-dense-index:
	@test -d $(V217_TFIDF_INDEX) || (echo "ERROR: TF-IDF index not found at $(V217_TFIDF_INDEX)" && exit 1)
	$(ENV) python scripts/build_dense_memory_index.py \
		--source-index $(V217_TFIDF_INDEX) \
		--output-dir $(V217_DENSE_INDEX) \
		--dense-model $(V217_DENSE_MODEL) \
		--batch-size 32 \
		--device auto
	@echo "Dense index built: $(V217_DENSE_INDEX)"

# ── 28-task frozen benchmark evaluations ─────────────────────────────────

eval-v217-tfidf-28:
	@test -d $(V217_MODEL) || (echo "ERROR: model not found at $(V217_MODEL)" && exit 1)
	@mkdir -p $(V217_OUT_DIR)
	$(ENV) python scripts/evaluate_code_agent.py \
		--hf-model $(V217_MODEL) \
		--mode best_of_n --n 3 \
		--scoring-mode verified_agent \
		--agent-contract strict \
		--stop-after-pass \
		--memory-enabled \
		--memory-index $(V217_TFIDF_INDEX) \
		--memory-top-k 4 \
		--retrieval-mode tfidf \
		--output outputs/eval_v217_tfidf_28 \
		--verbose
	@echo "v2.17 TF-IDF 28-task complete."

eval-v217-dense-28:
	@test -d $(V217_MODEL)       || (echo "ERROR: model not found at $(V217_MODEL)" && exit 1)
	@test -d $(V217_DENSE_INDEX) || (echo "ERROR: dense index not found — run make build-v217-dense-index first" && exit 1)
	@mkdir -p $(V217_OUT_DIR)
	$(ENV) python scripts/evaluate_code_agent.py \
		--hf-model $(V217_MODEL) \
		--mode best_of_n --n 3 \
		--scoring-mode verified_agent \
		--agent-contract strict \
		--stop-after-pass \
		--memory-enabled \
		--retrieval-mode dense \
		--dense-index $(V217_DENSE_INDEX) \
		--dense-model $(V217_DENSE_MODEL) \
		--memory-top-k 4 \
		--output outputs/eval_v217_dense_28 \
		--verbose
	@echo "v2.17 dense 28-task complete."

eval-v217-hybrid-28:
	@test -d $(V217_MODEL)       || (echo "ERROR: model not found at $(V217_MODEL)" && exit 1)
	@test -d $(V217_DENSE_INDEX) || (echo "ERROR: dense index not found — run make build-v217-dense-index first" && exit 1)
	@mkdir -p $(V217_OUT_DIR)
	$(ENV) python scripts/evaluate_code_agent.py \
		--hf-model $(V217_MODEL) \
		--mode best_of_n --n 3 \
		--scoring-mode verified_agent \
		--agent-contract strict \
		--stop-after-pass \
		--memory-enabled \
		--memory-index $(V217_TFIDF_INDEX) \
		--retrieval-mode hybrid \
		--dense-index $(V217_DENSE_INDEX) \
		--dense-model $(V217_DENSE_MODEL) \
		--rerank-top-n $(V217_RERANK_N) \
		--memory-top-k 4 \
		--output outputs/eval_v217_hybrid_28 \
		--verbose
	@echo "v2.17 hybrid 28-task complete."

# ── 32-task clean generalisation benchmark evaluations ───────────────────

eval-v217-tfidf-32:
	@test -d $(V217_MODEL)    || (echo "ERROR: model not found at $(V217_MODEL)" && exit 1)
	@test -f $(V217_TASKS_32) || (echo "ERROR: 32-task file not found at $(V217_TASKS_32)" && exit 1)
	@mkdir -p $(V217_OUT_DIR)
	$(ENV) python scripts/evaluate_code_agent.py \
		--hf-model $(V217_MODEL) \
		--tasks-file $(V217_TASKS_32) \
		--mode best_of_n --n 3 \
		--scoring-mode verified_agent \
		--agent-contract strict \
		--stop-after-pass \
		--memory-enabled \
		--memory-index $(V217_TFIDF_INDEX) \
		--memory-top-k 4 \
		--retrieval-mode tfidf \
		--output outputs/eval_v217_tfidf_32 \
		--verbose
	@echo "v2.17 TF-IDF 32-task complete."

eval-v217-dense-32:
	@test -d $(V217_MODEL)       || (echo "ERROR: model not found at $(V217_MODEL)" && exit 1)
	@test -f $(V217_TASKS_32)    || (echo "ERROR: 32-task file not found at $(V217_TASKS_32)" && exit 1)
	@test -d $(V217_DENSE_INDEX) || (echo "ERROR: dense index not found — run make build-v217-dense-index first" && exit 1)
	@mkdir -p $(V217_OUT_DIR)
	$(ENV) python scripts/evaluate_code_agent.py \
		--hf-model $(V217_MODEL) \
		--tasks-file $(V217_TASKS_32) \
		--mode best_of_n --n 3 \
		--scoring-mode verified_agent \
		--agent-contract strict \
		--stop-after-pass \
		--memory-enabled \
		--retrieval-mode dense \
		--dense-index $(V217_DENSE_INDEX) \
		--dense-model $(V217_DENSE_MODEL) \
		--memory-top-k 4 \
		--output outputs/eval_v217_dense_32 \
		--verbose
	@echo "v2.17 dense 32-task complete."

eval-v217-hybrid-32:
	@test -d $(V217_MODEL)       || (echo "ERROR: model not found at $(V217_MODEL)" && exit 1)
	@test -f $(V217_TASKS_32)    || (echo "ERROR: 32-task file not found at $(V217_TASKS_32)" && exit 1)
	@test -d $(V217_DENSE_INDEX) || (echo "ERROR: dense index not found — run make build-v217-dense-index first" && exit 1)
	@mkdir -p $(V217_OUT_DIR)
	$(ENV) python scripts/evaluate_code_agent.py \
		--hf-model $(V217_MODEL) \
		--tasks-file $(V217_TASKS_32) \
		--mode best_of_n --n 3 \
		--scoring-mode verified_agent \
		--agent-contract strict \
		--stop-after-pass \
		--memory-enabled \
		--memory-index $(V217_TFIDF_INDEX) \
		--retrieval-mode hybrid \
		--dense-index $(V217_DENSE_INDEX) \
		--dense-model $(V217_DENSE_MODEL) \
		--rerank-top-n $(V217_RERANK_N) \
		--memory-top-k 4 \
		--output outputs/eval_v217_hybrid_32 \
		--verbose
	@echo "v2.17 hybrid 32-task complete."

# ── summarise-v217 ────────────────────────────────────────────────────────
summarise-v217:
	@echo "=== v2.17 Dense Retrieval Pilot — Summary ==="
	@for d in \
	  outputs/eval_v217_tfidf_28 \
	  outputs/eval_v217_dense_28 \
	  outputs/eval_v217_hybrid_28 \
	  outputs/eval_v217_tfidf_32 \
	  outputs/eval_v217_dense_32 \
	  outputs/eval_v217_hybrid_32; do \
	    if [ -d "$$d" ]; then \
	      echo "  FOUND: $$d"; \
	      ls "$$d"/*.csv 2>/dev/null | head -3; \
	    else \
	      echo "  MISSING: $$d (run eval target first)"; \
	    fi; \
	done
	@echo ""
	@echo "Promotion rule: dense/hybrid must reach >=22/32 (68.8%) on 32-task to beat champion."
	@echo "Champion baseline: 20/32 = 62.5%"
	@echo ""
	@echo "See results/v217_dense_retrieval/ for full analysis."

# ── v2.18 Retrieval Baseline Stabilisation + Code-Dense Retrieval ─────────
#
# Phase A: Run TF-IDF 32-task benchmark three times to measure sampling variance.
#          Identifies stable-pass, stable-fail, and flip tasks.
#
# Phase B: Test code-specialised dense retrieval after baseline is understood.
#          V218_CODE_DENSE_MODEL can be a local path or installed model name.
#
# Usage (Phase A):
#   make eval-v218-tfidf-32-run1
#   make eval-v218-tfidf-32-run2
#   make eval-v218-tfidf-32-run3
#   make summarise-v218-tfidf-stability
#
# Usage (Phase B — after a code-dense model is available):
#   make build-v218-code-dense-index
#   make eval-v218-code-dense-32
#   make eval-v218-code-hybrid-32
#   make summarise-v218

.PHONY: eval-v218-tfidf-32-run1 eval-v218-tfidf-32-run2 eval-v218-tfidf-32-run3 \
        summarise-v218-tfidf-stability \
        build-v218-code-dense-index \
        eval-v218-code-dense-32 eval-v218-code-hybrid-32 \
        eval-v218-code-dense-28 eval-v218-code-hybrid-28 \
        summarise-v218

V218_MODEL        := outputs/qwen15b_v27_champion_merged
V218_TFIDF_INDEX  := memory/index_adapted
V218_TASKS_32     := data/v210_clean_repair_generalisation_tasks.jsonl
V218_CODE_DENSE_INDEX := memory/dense_index_v218_code
# Override V218_CODE_DENSE_MODEL at the command line or here with a local path.
# Example: make build-v218-code-dense-index V218_CODE_DENSE_MODEL=models/embeddings/codebert
V218_CODE_DENSE_MODEL ?= microsoft/codebert-base
V218_RERANK_N     := 20
V218_OUT_DIR      := results/v218_retrieval_stability

# ── Phase A: TF-IDF stability runs ───────────────────────────────────────

_v218_tfidf_eval = $(ENV) python scripts/evaluate_code_agent.py \
	--hf-model $(V218_MODEL) \
	--tasks-file $(V218_TASKS_32) \
	--mode best_of_n --n 3 \
	--scoring-mode verified_agent \
	--agent-contract strict \
	--stop-after-pass \
	--memory-enabled \
	--memory-index $(V218_TFIDF_INDEX) \
	--memory-top-k 4 \
	--retrieval-mode tfidf \
	--verbose

eval-v218-tfidf-32-run1:
	@test -d $(V218_MODEL)    || (echo "ERROR: model not found at $(V218_MODEL)" && exit 1)
	@test -f $(V218_TASKS_32) || (echo "ERROR: task file not found at $(V218_TASKS_32)" && exit 1)
	@mkdir -p $(V218_OUT_DIR)
	$(_v218_tfidf_eval) --output outputs/eval_v218_tfidf_32_run1
	@echo "v2.18 TF-IDF run 1 complete."

eval-v218-tfidf-32-run2:
	@test -d $(V218_MODEL)    || (echo "ERROR: model not found at $(V218_MODEL)" && exit 1)
	@test -f $(V218_TASKS_32) || (echo "ERROR: task file not found at $(V218_TASKS_32)" && exit 1)
	@mkdir -p $(V218_OUT_DIR)
	$(_v218_tfidf_eval) --output outputs/eval_v218_tfidf_32_run2
	@echo "v2.18 TF-IDF run 2 complete."

eval-v218-tfidf-32-run3:
	@test -d $(V218_MODEL)    || (echo "ERROR: model not found at $(V218_MODEL)" && exit 1)
	@test -f $(V218_TASKS_32) || (echo "ERROR: task file not found at $(V218_TASKS_32)" && exit 1)
	@mkdir -p $(V218_OUT_DIR)
	$(_v218_tfidf_eval) --output outputs/eval_v218_tfidf_32_run3
	@echo "v2.18 TF-IDF run 3 complete."

summarise-v218-tfidf-stability:
	@mkdir -p $(V218_OUT_DIR)
	$(ENV) python scripts/summarise_v218_tfidf_stability.py
	@echo ""
	@cat results/v218_retrieval_stability/tfidf_baseline_report.md

# ── Phase B: Code-specialised dense retrieval ─────────────────────────────

build-v218-code-dense-index:
	@test -d $(V218_TFIDF_INDEX) || (echo "ERROR: TF-IDF index not found at $(V218_TFIDF_INDEX)" && exit 1)
	$(ENV) python scripts/build_dense_memory_index.py \
		--source-index $(V218_TFIDF_INDEX) \
		--output-dir $(V218_CODE_DENSE_INDEX) \
		--dense-model $(V218_CODE_DENSE_MODEL) \
		--batch-size 32 \
		--device auto
	@echo "Code-dense index built: $(V218_CODE_DENSE_INDEX)"

eval-v218-code-dense-32:
	@test -d $(V218_MODEL)            || (echo "ERROR: model not found at $(V218_MODEL)" && exit 1)
	@test -f $(V218_TASKS_32)         || (echo "ERROR: task file not found" && exit 1)
	@test -d $(V218_CODE_DENSE_INDEX) || (echo "ERROR: dense index not found — run make build-v218-code-dense-index" && exit 1)
	@mkdir -p $(V218_OUT_DIR)
	$(ENV) python scripts/evaluate_code_agent.py \
		--hf-model $(V218_MODEL) \
		--tasks-file $(V218_TASKS_32) \
		--mode best_of_n --n 3 \
		--scoring-mode verified_agent \
		--agent-contract strict \
		--stop-after-pass \
		--memory-enabled \
		--retrieval-mode dense \
		--dense-index $(V218_CODE_DENSE_INDEX) \
		--dense-model $(V218_CODE_DENSE_MODEL) \
		--memory-top-k 4 \
		--output outputs/eval_v218_code_dense_32 \
		--verbose
	@echo "v2.18 code-dense 32-task complete."

eval-v218-code-hybrid-32:
	@test -d $(V218_MODEL)            || (echo "ERROR: model not found at $(V218_MODEL)" && exit 1)
	@test -f $(V218_TASKS_32)         || (echo "ERROR: task file not found" && exit 1)
	@test -d $(V218_CODE_DENSE_INDEX) || (echo "ERROR: dense index not found — run make build-v218-code-dense-index" && exit 1)
	@mkdir -p $(V218_OUT_DIR)
	$(ENV) python scripts/evaluate_code_agent.py \
		--hf-model $(V218_MODEL) \
		--tasks-file $(V218_TASKS_32) \
		--mode best_of_n --n 3 \
		--scoring-mode verified_agent \
		--agent-contract strict \
		--stop-after-pass \
		--memory-enabled \
		--memory-index $(V218_TFIDF_INDEX) \
		--retrieval-mode hybrid \
		--dense-index $(V218_CODE_DENSE_INDEX) \
		--dense-model $(V218_CODE_DENSE_MODEL) \
		--rerank-top-n $(V218_RERANK_N) \
		--memory-top-k 4 \
		--output outputs/eval_v218_code_hybrid_32 \
		--verbose
	@echo "v2.18 code-hybrid 32-task complete."

eval-v218-code-dense-28:
	@test -d $(V218_MODEL)            || (echo "ERROR: model not found at $(V218_MODEL)" && exit 1)
	@test -d $(V218_CODE_DENSE_INDEX) || (echo "ERROR: dense index not found — run make build-v218-code-dense-index" && exit 1)
	@mkdir -p $(V218_OUT_DIR)
	$(ENV) python scripts/evaluate_code_agent.py \
		--hf-model $(V218_MODEL) \
		--mode best_of_n --n 3 \
		--scoring-mode verified_agent \
		--agent-contract strict \
		--stop-after-pass \
		--memory-enabled \
		--retrieval-mode dense \
		--dense-index $(V218_CODE_DENSE_INDEX) \
		--dense-model $(V218_CODE_DENSE_MODEL) \
		--memory-top-k 4 \
		--output outputs/eval_v218_code_dense_28 \
		--verbose
	@echo "v2.18 code-dense 28-task complete."

eval-v218-code-hybrid-28:
	@test -d $(V218_MODEL)            || (echo "ERROR: model not found at $(V218_MODEL)" && exit 1)
	@test -d $(V218_CODE_DENSE_INDEX) || (echo "ERROR: dense index not found — run make build-v218-code-dense-index" && exit 1)
	@mkdir -p $(V218_OUT_DIR)
	$(ENV) python scripts/evaluate_code_agent.py \
		--hf-model $(V218_MODEL) \
		--mode best_of_n --n 3 \
		--scoring-mode verified_agent \
		--agent-contract strict \
		--stop-after-pass \
		--memory-enabled \
		--memory-index $(V218_TFIDF_INDEX) \
		--retrieval-mode hybrid \
		--dense-index $(V218_CODE_DENSE_INDEX) \
		--dense-model $(V218_CODE_DENSE_MODEL) \
		--rerank-top-n $(V218_RERANK_N) \
		--memory-top-k 4 \
		--output outputs/eval_v218_code_hybrid_28 \
		--verbose
	@echo "v2.18 code-hybrid 28-task complete."

summarise-v218:
	@echo "=== v2.18 Full Summary ==="
	@echo ""
	@echo "--- Phase A: TF-IDF stability ---"
	@for d in \
	  outputs/eval_v218_tfidf_32_run1 \
	  outputs/eval_v218_tfidf_32_run2 \
	  outputs/eval_v218_tfidf_32_run3; do \
	    if [ -d "$$d" ]; then echo "  FOUND: $$d"; \
	    else echo "  MISSING: $$d"; fi; \
	done
	@echo ""
	@echo "--- Phase B: Code-dense results ---"
	@for d in \
	  outputs/eval_v218_code_dense_32 \
	  outputs/eval_v218_code_hybrid_32 \
	  outputs/eval_v218_code_dense_28 \
	  outputs/eval_v218_code_hybrid_28; do \
	    if [ -d "$$d" ]; then echo "  FOUND: $$d"; \
	    else echo "  MISSING: $$d (run eval target first)"; fi; \
	done
	@echo ""
	@echo "Promotion rule: code-dense/hybrid must beat stabilised TF-IDF baseline on 32-task."
	@echo "Strong result: >=22/32. Noise floor determined by Phase A runs."
	@echo "See results/v218_retrieval_stability/ for full analysis."

# ── v2.18 Phase B: 3-run code-dense audit (UniXcoder primary) ──────────────
# Tests a different-architecture code-pretrained encoder (768d) against the
# stabilised code-aware MiniLM dense baseline (384d, mean 16.3/32).
# NOTE: in hybrid mode, stage-1 shortlist uses memory/index_adapted, which is
# code-aware MiniLM dense (per Phase A finding) — NOT TF-IDF. Stage-2 reranks
# with the Phase B encoder.
#
# Usage:
#   make build-v218-phaseb-code-index
#   make eval-v218-phaseb-code-dense-32-run1    (run2, run3)
#   make eval-v218-phaseb-code-hybrid-32-run1   (run2, run3)
#   make summarise-v218-phaseb
#
# Fallback to CodeBERT (override both vars):
#   make build-v218-phaseb-code-index \
#       V218_PHASEB_MODEL=microsoft/codebert-base \
#       V218_PHASEB_INDEX=memory/dense_index_codebert

V218_PHASEB_MODEL ?= microsoft/unixcoder-base
V218_PHASEB_INDEX ?= memory/dense_index_unixcoder

.PHONY: build-v218-phaseb-code-index \
        eval-v218-phaseb-code-dense-32-run1 eval-v218-phaseb-code-dense-32-run2 \
        eval-v218-phaseb-code-dense-32-run3 \
        eval-v218-phaseb-code-hybrid-32-run1 eval-v218-phaseb-code-hybrid-32-run2 \
        eval-v218-phaseb-code-hybrid-32-run3 \
        summarise-v218-phaseb

build-v218-phaseb-code-index:
	@test -d $(V218_TFIDF_INDEX) || (echo "ERROR: source index not found at $(V218_TFIDF_INDEX)" && exit 1)
	$(ENV) python scripts/build_dense_memory_index.py \
		--source-index $(V218_TFIDF_INDEX) \
		--output-dir $(V218_PHASEB_INDEX) \
		--dense-model $(V218_PHASEB_MODEL) \
		--batch-size 32 \
		--device auto
	@echo "Phase B code-dense index built: $(V218_PHASEB_INDEX) (model: $(V218_PHASEB_MODEL))"

# Template: generate dense + hybrid eval targets for run $(1)
define V218_PHASEB_EVAL_RULE
eval-v218-phaseb-code-dense-32-run$(1):
	@test -d $(V218_MODEL)        || (echo "ERROR: model not found at $(V218_MODEL)" && exit 1)
	@test -f $(V218_TASKS_32)     || (echo "ERROR: task file not found at $(V218_TASKS_32)" && exit 1)
	@test -d $(V218_PHASEB_INDEX) || (echo "ERROR: dense index not found — run make build-v218-phaseb-code-index" && exit 1)
	@mkdir -p $(V218_OUT_DIR)
	$$(ENV) python scripts/evaluate_code_agent.py \
		--hf-model $(V218_MODEL) \
		--tasks-file $(V218_TASKS_32) \
		--mode best_of_n --n 3 \
		--scoring-mode verified_agent \
		--agent-contract strict \
		--stop-after-pass \
		--memory-enabled \
		--retrieval-mode dense \
		--dense-index $(V218_PHASEB_INDEX) \
		--dense-model $(V218_PHASEB_MODEL) \
		--memory-top-k 4 \
		--output outputs/eval_v218_phaseb_code_dense_32_run$(1) \
		--verbose
	@echo "v2.18 Phase B code-dense run $(1) complete."

eval-v218-phaseb-code-hybrid-32-run$(1):
	@test -d $(V218_MODEL)        || (echo "ERROR: model not found at $(V218_MODEL)" && exit 1)
	@test -f $(V218_TASKS_32)     || (echo "ERROR: task file not found at $(V218_TASKS_32)" && exit 1)
	@test -d $(V218_PHASEB_INDEX) || (echo "ERROR: dense index not found — run make build-v218-phaseb-code-index" && exit 1)
	@mkdir -p $(V218_OUT_DIR)
	$$(ENV) python scripts/evaluate_code_agent.py \
		--hf-model $(V218_MODEL) \
		--tasks-file $(V218_TASKS_32) \
		--mode best_of_n --n 3 \
		--scoring-mode verified_agent \
		--agent-contract strict \
		--stop-after-pass \
		--memory-enabled \
		--memory-index $(V218_TFIDF_INDEX) \
		--retrieval-mode hybrid \
		--dense-index $(V218_PHASEB_INDEX) \
		--dense-model $(V218_PHASEB_MODEL) \
		--rerank-top-n $(V218_RERANK_N) \
		--memory-top-k 4 \
		--output outputs/eval_v218_phaseb_code_hybrid_32_run$(1) \
		--verbose
	@echo "v2.18 Phase B code-hybrid run $(1) complete."
endef

$(eval $(call V218_PHASEB_EVAL_RULE,1))
$(eval $(call V218_PHASEB_EVAL_RULE,2))
$(eval $(call V218_PHASEB_EVAL_RULE,3))

summarise-v218-phaseb:
	@mkdir -p $(V218_OUT_DIR)
	$(ENV) python scripts/summarise_v218_phaseb.py
	@echo ""
	@echo "Phase B summary: $(V218_OUT_DIR)/phase_b_code_dense_summary.md"
	@echo "Comparison CSV : $(V218_OUT_DIR)/phase_b_code_dense_comparison.csv"
	@echo "Claim boundary : $(V218_OUT_DIR)/phase_b_claim_boundary.md"

# ── v2.19 Structured Memory Records + Query Reranking ─────────────────────
# Holds the encoder FIXED (baseline code-aware MiniLM) to isolate the effect of
# record structure + multi-view query + deterministic reranking against the
# stabilised baseline (mean 16.3/32). Same protected memory pool; only embedding,
# retrieval, and ranking change. Protected indexes are never overwritten.
#
# Usage:
#   make build-v219-structured-memory-index
#   make eval-v219-structured-dense-32-run1    (run2, run3)
#   make eval-v219-structured-hybrid-32-run1   (run2, run3)
#   make summarise-v219-structured

V219_MODEL          := $(V218_MODEL)
V219_TASKS_32       := $(V218_TASKS_32)
V219_BASELINE_INDEX := memory/index_adapted
V219_STRUCT_RECORDS := memory/structured_v219/records.jsonl
V219_STRUCT_INDEX   := memory/dense_index_v219_structured
V219_ENCODER        := models/embeddings/code-memory-embedder
V219_RERANK_N       := 20
V219_OUT_DIR        := results/v219_structured_memory

.PHONY: build-v219-structured-memory-index \
        eval-v219-structured-dense-32-run1 eval-v219-structured-dense-32-run2 \
        eval-v219-structured-dense-32-run3 \
        eval-v219-structured-hybrid-32-run1 eval-v219-structured-hybrid-32-run2 \
        eval-v219-structured-hybrid-32-run3 \
        summarise-v219-structured

build-v219-structured-memory-index:
	@test -d $(V219_BASELINE_INDEX) || (echo "ERROR: protected index not found at $(V219_BASELINE_INDEX)" && exit 1)
	@test -d $(V219_ENCODER)        || (echo "ERROR: baseline encoder not found at $(V219_ENCODER)" && exit 1)
	$(ENV) python scripts/build_structured_memory_records.py \
		--source-index $(V219_BASELINE_INDEX) \
		--output $(V219_STRUCT_RECORDS)
	$(ENV) python scripts/build_dense_memory_index.py \
		--memory-jsonl $(V219_STRUCT_RECORDS) \
		--output-dir $(V219_STRUCT_INDEX) \
		--dense-model $(V219_ENCODER) \
		--batch-size 32 --device auto
	@echo "v2.19 structured index built: $(V219_STRUCT_INDEX) (encoder: $(V219_ENCODER))"

# Template: generate structured dense + hybrid eval targets for run $(1)
define V219_EVAL_RULE
eval-v219-structured-dense-32-run$(1):
	@test -d $(V219_MODEL)        || (echo "ERROR: model not found at $(V219_MODEL)" && exit 1)
	@test -f $(V219_TASKS_32)     || (echo "ERROR: task file not found at $(V219_TASKS_32)" && exit 1)
	@test -d $(V219_STRUCT_INDEX) || (echo "ERROR: structured index not found — run make build-v219-structured-memory-index" && exit 1)
	@mkdir -p $(V219_OUT_DIR)
	$$(ENV) python scripts/evaluate_code_agent.py \
		--hf-model $(V219_MODEL) \
		--tasks-file $(V219_TASKS_32) \
		--mode best_of_n --n 3 \
		--scoring-mode verified_agent \
		--agent-contract strict \
		--stop-after-pass \
		--memory-enabled \
		--retrieval-mode structured \
		--structured-index $(V219_STRUCT_INDEX) \
		--dense-model $(V219_ENCODER) \
		--rerank-top-n $(V219_RERANK_N) \
		--memory-top-k 4 \
		--output outputs/eval_v219_structured_dense_32_run$(1) \
		--verbose
	@echo "v2.19 structured-dense run $(1) complete."

eval-v219-structured-hybrid-32-run$(1):
	@test -d $(V219_MODEL)        || (echo "ERROR: model not found at $(V219_MODEL)" && exit 1)
	@test -f $(V219_TASKS_32)     || (echo "ERROR: task file not found at $(V219_TASKS_32)" && exit 1)
	@test -d $(V219_STRUCT_INDEX) || (echo "ERROR: structured index not found — run make build-v219-structured-memory-index" && exit 1)
	@mkdir -p $(V219_OUT_DIR)
	$$(ENV) python scripts/evaluate_code_agent.py \
		--hf-model $(V219_MODEL) \
		--tasks-file $(V219_TASKS_32) \
		--mode best_of_n --n 3 \
		--scoring-mode verified_agent \
		--agent-contract strict \
		--stop-after-pass \
		--memory-enabled \
		--memory-index $(V219_BASELINE_INDEX) \
		--retrieval-mode structured-hybrid \
		--structured-index $(V219_STRUCT_INDEX) \
		--dense-model $(V219_ENCODER) \
		--rerank-top-n $(V219_RERANK_N) \
		--memory-top-k 4 \
		--output outputs/eval_v219_structured_hybrid_32_run$(1) \
		--verbose
	@echo "v2.19 structured-hybrid run $(1) complete."
endef

$(eval $(call V219_EVAL_RULE,1))
$(eval $(call V219_EVAL_RULE,2))
$(eval $(call V219_EVAL_RULE,3))

summarise-v219-structured:
	@mkdir -p $(V219_OUT_DIR)
	$(ENV) python scripts/summarise_v219_structured_memory.py
	@echo ""
	@echo "v2.19 summary : $(V219_OUT_DIR)/summary.md"
	@echo "Claim boundary: $(V219_OUT_DIR)/claim_boundary.md"

# ── v2.19b Family-Targeted Memory Coverage ────────────────────────────────
# Tests the v2.19 finding that the bottleneck is memory COVERAGE. Adds 16 verified
# same-family-DIFFERENT-task repair records (interval/tree/rle/dict; contamination-
# guarded — names disjoint from the 32 benchmark tasks) to the structured pool, holds
# the encoder fixed, and re-runs the structured-dense audit. Protected indexes untouched.
# structured-hybrid is omitted: its shortlist gate is the original 99-record pool, so it
# cannot surface the new records (testing it fairly needs a combined store-index; deferred).
#
# Usage:
#   make build-v219b-family-memory-index
#   make eval-v219b-structured-dense-32-run1    (run2, run3)
#   make summarise-v219b-family

V219B_FAMILY_RECORDS := data/v219b_family_repair_records.jsonl
V219B_STRUCT_RECORDS := memory/structured_v219b/records.jsonl
V219B_STRUCT_INDEX   := memory/dense_index_v219b_structured
V219B_OUT_DIR        := results/v219b_family_memory

.PHONY: build-v219b-family-memory-index \
        eval-v219b-structured-dense-32-run1 eval-v219b-structured-dense-32-run2 \
        eval-v219b-structured-dense-32-run3 \
        summarise-v219b-family

build-v219b-family-memory-index:
	@test -d $(V219_BASELINE_INDEX) || (echo "ERROR: protected index not found at $(V219_BASELINE_INDEX)" && exit 1)
	@test -d $(V219_ENCODER)        || (echo "ERROR: baseline encoder not found at $(V219_ENCODER)" && exit 1)
	$(ENV) python scripts/build_v219b_family_records.py
	$(ENV) python scripts/build_structured_memory_records.py \
		--source-index $(V219_BASELINE_INDEX) \
		--extra-jsonl $(V219B_FAMILY_RECORDS) \
		--output $(V219B_STRUCT_RECORDS) \
		--schema-doc $(V219B_OUT_DIR)/structured_record_schema.md
	$(ENV) python scripts/build_dense_memory_index.py \
		--memory-jsonl $(V219B_STRUCT_RECORDS) \
		--output-dir $(V219B_STRUCT_INDEX) \
		--dense-model $(V219_ENCODER) \
		--batch-size 32 --device auto
	@echo "v2.19b combined structured index built: $(V219B_STRUCT_INDEX)"

# Template: generate structured-dense eval target for run $(1)
define V219B_EVAL_RULE
eval-v219b-structured-dense-32-run$(1):
	@test -d $(V219_MODEL)         || (echo "ERROR: model not found at $(V219_MODEL)" && exit 1)
	@test -f $(V219_TASKS_32)      || (echo "ERROR: task file not found at $(V219_TASKS_32)" && exit 1)
	@test -d $(V219B_STRUCT_INDEX) || (echo "ERROR: index not found — run make build-v219b-family-memory-index" && exit 1)
	@mkdir -p $(V219B_OUT_DIR)
	$$(ENV) python scripts/evaluate_code_agent.py \
		--hf-model $(V219_MODEL) \
		--tasks-file $(V219_TASKS_32) \
		--mode best_of_n --n 3 \
		--scoring-mode verified_agent \
		--agent-contract strict \
		--stop-after-pass \
		--memory-enabled \
		--retrieval-mode structured \
		--structured-index $(V219B_STRUCT_INDEX) \
		--dense-model $(V219_ENCODER) \
		--rerank-top-n $(V219_RERANK_N) \
		--memory-top-k 4 \
		--output outputs/eval_v219b_structured_dense_32_run$(1) \
		--verbose
	@echo "v2.19b structured-dense run $(1) complete."
endef

$(eval $(call V219B_EVAL_RULE,1))
$(eval $(call V219B_EVAL_RULE,2))
$(eval $(call V219B_EVAL_RULE,3))

summarise-v219b-family:
	@mkdir -p $(V219B_OUT_DIR)
	$(ENV) python scripts/summarise_v219b_family_memory.py
	@echo ""
	@echo "v2.19b summary : $(V219B_OUT_DIR)/summary.md"
	@echo "Claim boundary: $(V219B_OUT_DIR)/claim_boundary.md"

# ── v2.19c Confirmation + Targeted Coverage Expansion ─────────────────────
# Phase 1: confirm the v2.19b combined-pool lift over many seeds (mean > 18.3 stable?).
#          Reuses the v2.19b index directly (confirms that exact artifact).
# Phase 2: author targeted same-family-different-task records for the families that did
#          NOT convert in v2.19b (tree stable-fails, interval_union) — contamination-
#          guarded, execution-verified.
# Phase 3: build the EXPANDED pool (99 + v2.19b 16 + targeted) and re-run dense.
# Encoder held fixed throughout. Protected indexes untouched.
#
# Usage:
#   make eval-v219c-confirm-32-run1 ... run7   (Phase 1, or: make v219c-confirm-all)
#   make build-v219c-expanded-index            (Phase 2+3 build)
#   make eval-v219c-expanded-dense-32-run1 ... run3
#   make summarise-v219c

V219C_CONFIRM_INDEX   := $(V219B_STRUCT_INDEX)
V219C_TARGETED_RECORDS := data/v219c_targeted_repair_records.jsonl
V219C_COMBINED_EXTRA  := memory/structured_v219c/combined_extra.jsonl
V219C_STRUCT_RECORDS  := memory/structured_v219c/records.jsonl
V219C_EXPANDED_INDEX  := memory/dense_index_v219c_confirm
V219C_OUT_DIR         := results/v219c_confirmation_coverage

_v219c_eval = $(ENV) python scripts/evaluate_code_agent.py \
	--hf-model $(V219_MODEL) \
	--tasks-file $(V219_TASKS_32) \
	--mode best_of_n --n 3 \
	--scoring-mode verified_agent \
	--agent-contract strict \
	--stop-after-pass \
	--memory-enabled \
	--retrieval-mode structured \
	--dense-model $(V219_ENCODER) \
	--rerank-top-n $(V219_RERANK_N) \
	--memory-top-k 4 \
	--verbose

.PHONY: build-v219c-expanded-index summarise-v219c v219c-confirm-all

# Phase 1 confirmation template (confirms the v2.19b artifact directly)
define V219C_CONFIRM_RULE
.PHONY: eval-v219c-confirm-32-run$(1)
eval-v219c-confirm-32-run$(1):
	@test -d $(V219_MODEL)           || (echo "ERROR: model not found at $(V219_MODEL)" && exit 1)
	@test -d $(V219C_CONFIRM_INDEX)  || (echo "ERROR: v2.19b index not found — run make build-v219b-family-memory-index" && exit 1)
	@mkdir -p $(V219C_OUT_DIR)
	$$(_v219c_eval) --structured-index $(V219C_CONFIRM_INDEX) \
		--output outputs/eval_v219c_confirm_run$(1)
	@echo "v2.19c confirmation run $(1) complete."
endef
$(foreach n,1 2 3 4 5 6 7,$(eval $(call V219C_CONFIRM_RULE,$(n))))

# Phase 2+3: build expanded pool = 99 protected + v2.19b 16 + v2.19c targeted
build-v219c-expanded-index:
	@test -d $(V219_BASELINE_INDEX) || (echo "ERROR: protected index not found" && exit 1)
	@test -d $(V219_ENCODER)        || (echo "ERROR: baseline encoder not found" && exit 1)
	$(ENV) python scripts/build_v219c_targeted_records.py
	@mkdir -p memory/structured_v219c
	cat $(V219B_FAMILY_RECORDS) $(V219C_TARGETED_RECORDS) > $(V219C_COMBINED_EXTRA)
	$(ENV) python scripts/build_structured_memory_records.py \
		--source-index $(V219_BASELINE_INDEX) \
		--extra-jsonl $(V219C_COMBINED_EXTRA) \
		--output $(V219C_STRUCT_RECORDS) \
		--schema-doc $(V219C_OUT_DIR)/structured_record_schema.md
	$(ENV) python scripts/build_dense_memory_index.py \
		--memory-jsonl $(V219C_STRUCT_RECORDS) \
		--output-dir $(V219C_EXPANDED_INDEX) \
		--dense-model $(V219_ENCODER) \
		--batch-size 32 --device auto
	@echo "v2.19c expanded index built: $(V219C_EXPANDED_INDEX)"

# Phase 3 expanded-pool dense eval template
define V219C_EXPANDED_RULE
.PHONY: eval-v219c-expanded-dense-32-run$(1)
eval-v219c-expanded-dense-32-run$(1):
	@test -d $(V219_MODEL)          || (echo "ERROR: model not found at $(V219_MODEL)" && exit 1)
	@test -d $(V219C_EXPANDED_INDEX) || (echo "ERROR: expanded index not found — run make build-v219c-expanded-index" && exit 1)
	@mkdir -p $(V219C_OUT_DIR)
	$$(_v219c_eval) --structured-index $(V219C_EXPANDED_INDEX) \
		--output outputs/eval_v219c_expanded_dense_32_run$(1)
	@echo "v2.19c expanded-dense run $(1) complete."
endef
$(foreach n,1 2 3,$(eval $(call V219C_EXPANDED_RULE,$(n))))

summarise-v219c:
	@mkdir -p $(V219C_OUT_DIR)
	$(ENV) python scripts/summarise_v219c_confirmation.py
	@echo ""
	@echo "v2.19c summary : $(V219C_OUT_DIR)/summary.md"
	@echo "Claim boundary: $(V219C_OUT_DIR)/claim_boundary.md"

# ── v2.21 ForgeReasoningCore Tool-Action Curriculum ───────────────────────
# v2.19c proved tree-family failures are reasoning/control-bound (relevant memory is
# retrieved but the model still fails). v2.21 tests an EXECUTION-PLAN prompt (plan ->
# code -> test -> repair -> final) on top of the SAME v2.19c expanded retrieval, isolating
# the prompt/control variable. Control = v2.19c expanded dense WITHOUT the plan prompt.
# No new index, no model weights. Protected indexes untouched.
#
# Usage:
#   make eval-v221-reasoning-tree-32-run1   (run2, run3)
#   make eval-v221-tree-stablefails-run1    (run2, run3)
#   make summarise-v221-reasoning

V221_INDEX   := $(V219C_EXPANDED_INDEX)
V221_OUT_DIR := results/v221_reasoning_curriculum
V221_TREE_STABLEFAILS := v210_tree_from_list v210_tree_max_path_sum v210_tree_serialize v210_tree_width

_v221_eval = $(ENV) python scripts/evaluate_code_agent.py \
	--hf-model $(V219_MODEL) \
	--mode best_of_n --n 3 \
	--scoring-mode verified_agent \
	--agent-contract strict \
	--stop-after-pass \
	--execution-plan-mode \
	--memory-enabled \
	--retrieval-mode structured \
	--structured-index $(V221_INDEX) \
	--dense-model $(V219_ENCODER) \
	--rerank-top-n $(V219_RERANK_N) \
	--memory-top-k 4 \
	--verbose

.PHONY: summarise-v221-reasoning

# Full 32-task with execution-plan mode
define V221_FULL_RULE
.PHONY: eval-v221-reasoning-tree-32-run$(1)
eval-v221-reasoning-tree-32-run$(1):
	@test -d $(V219_MODEL)  || (echo "ERROR: model not found at $(V219_MODEL)" && exit 1)
	@test -f $(V219_TASKS_32) || (echo "ERROR: task file not found" && exit 1)
	@test -d $(V221_INDEX)  || (echo "ERROR: v2.19c index not found — run make build-v219c-expanded-index" && exit 1)
	@mkdir -p $(V221_OUT_DIR)
	$$(_v221_eval) --tasks-file $(V219_TASKS_32) \
		--output outputs/eval_v221_reasoning_tree_32_run$(1)
	@echo "v2.21 reasoning-tree-32 run $(1) complete."
endef
$(foreach n,1 2 3,$(eval $(call V221_FULL_RULE,$(n))))

# Isolated persistent tree stable-fails (best-of-3 each)
define V221_SUBSET_RULE
.PHONY: eval-v221-tree-stablefails-run$(1)
eval-v221-tree-stablefails-run$(1):
	@test -d $(V219_MODEL) || (echo "ERROR: model not found at $(V219_MODEL)" && exit 1)
	@test -d $(V221_INDEX) || (echo "ERROR: v2.19c index not found — run make build-v219c-expanded-index" && exit 1)
	@mkdir -p $(V221_OUT_DIR)
	$$(_v221_eval) --tasks-file $(V219_TASKS_32) \
		--task-ids $(V221_TREE_STABLEFAILS) \
		--output outputs/eval_v221_tree_stablefails_run$(1)
	@echo "v2.21 tree-stablefails run $(1) complete."
endef
$(foreach n,1 2 3,$(eval $(call V221_SUBSET_RULE,$(n))))

summarise-v221-reasoning:
	@mkdir -p $(V221_OUT_DIR)
	$(ENV) python scripts/summarise_v221_reasoning.py
	@echo ""
	@echo "v2.21 summary : $(V221_OUT_DIR)/summary.md"
	@echo "Claim boundary: $(V221_OUT_DIR)/claim_boundary.md"

# ── v2.21b Tree-Width Planning Ablation ───────────────────────────────────
# Controlled ablation: the EXECUTION_PLAN prompt with the worked example REMOVED (same
# contract, same v2.19c retrieval). Tests whether the v2.21 tree_width conversion (6/6) came
# from the plan structure or from the worked example. No weights, no new index.
#
# Usage:
#   make eval-v221b-tree-stablefails-run1 ... run6
#   make eval-v221b-reasoning-tree-32-run1 ... run3
#   make summarise-v221b-ablation

V221B_OUT_DIR := results/v221b_tree_width_ablation

_v221b_eval = $(ENV) python scripts/evaluate_code_agent.py \
	--hf-model $(V219_MODEL) \
	--mode best_of_n --n 3 \
	--scoring-mode verified_agent \
	--agent-contract strict \
	--stop-after-pass \
	--plan-prompt-without-example \
	--memory-enabled \
	--retrieval-mode structured \
	--structured-index $(V221_INDEX) \
	--dense-model $(V219_ENCODER) \
	--rerank-top-n $(V219_RERANK_N) \
	--memory-top-k 4 \
	--verbose

.PHONY: summarise-v221b-ablation

# Tree stable-fails under ablation (6 runs for a robust tree_width estimate)
define V221B_SUBSET_RULE
.PHONY: eval-v221b-tree-stablefails-run$(1)
eval-v221b-tree-stablefails-run$(1):
	@test -d $(V219_MODEL) || (echo "ERROR: model not found at $(V219_MODEL)" && exit 1)
	@test -d $(V221_INDEX) || (echo "ERROR: v2.19c index not found — run make build-v219c-expanded-index" && exit 1)
	@mkdir -p $(V221B_OUT_DIR)
	$$(_v221b_eval) --tasks-file $(V219_TASKS_32) \
		--task-ids $(V221_TREE_STABLEFAILS) \
		--output outputs/eval_v221b_tree_stablefails_run$(1)
	@echo "v2.21b ablation tree-stablefails run $(1) complete."
endef
$(foreach n,1 2 3 4 5 6,$(eval $(call V221B_SUBSET_RULE,$(n))))

# Full 32-task under ablation (aggregate + regressions)
define V221B_FULL_RULE
.PHONY: eval-v221b-reasoning-tree-32-run$(1)
eval-v221b-reasoning-tree-32-run$(1):
	@test -d $(V219_MODEL)  || (echo "ERROR: model not found at $(V219_MODEL)" && exit 1)
	@test -f $(V219_TASKS_32) || (echo "ERROR: task file not found" && exit 1)
	@test -d $(V221_INDEX)  || (echo "ERROR: v2.19c index not found" && exit 1)
	@mkdir -p $(V221B_OUT_DIR)
	$$(_v221b_eval) --tasks-file $(V219_TASKS_32) \
		--output outputs/eval_v221b_reasoning_tree_32_run$(1)
	@echo "v2.21b ablation reasoning-tree-32 run $(1) complete."
endef
$(foreach n,1 2 3,$(eval $(call V221B_FULL_RULE,$(n))))

summarise-v221b-ablation:
	@mkdir -p $(V221B_OUT_DIR)
	$(ENV) python scripts/summarise_v221b_ablation.py
	@echo ""
	@echo "v2.21b summary : $(V221B_OUT_DIR)/summary.md"
	@echo "Claim boundary: $(V221B_OUT_DIR)/claim_boundary.md"

# ── v2.22 Verifier-Guided Multi-Step Repair ───────────────────────────────
# Targets the three CAPABILITY-BOUND tree tasks (fail despite 100% plan adherence in
# v2.21/v2.21b). Adds a precise VERIFIER signal + bounded repair budget on top of the
# execution-plan prompt. Same v2.19c retrieval; no weights; no new index.
#
# Usage:
#   make eval-v222-repair-capbound-run1   (run2, run3)
#   make eval-v222-repair-32-run1         (run2, run3)
#   make summarise-v222-repair

V222_CAPBOUND     := v210_tree_serialize v210_tree_from_list v210_tree_max_path_sum
V222_REPAIR_ITERS ?= 3
V222_OUT_DIR      := results/v222_verifier_guided_repair

_v222_eval = $(ENV) python scripts/evaluate_code_agent.py \
	--hf-model $(V219_MODEL) \
	--mode best_of_n --n 3 \
	--scoring-mode verified_agent \
	--agent-contract strict \
	--stop-after-pass \
	--verifier-repair \
	--max-repair-iters $(V222_REPAIR_ITERS) \
	--memory-enabled \
	--retrieval-mode structured \
	--structured-index $(V221_INDEX) \
	--dense-model $(V219_ENCODER) \
	--rerank-top-n $(V219_RERANK_N) \
	--memory-top-k 4 \
	--verbose

.PHONY: summarise-v222-repair

# Capability-bound subset under verifier-guided repair
define V222_CAP_RULE
.PHONY: eval-v222-repair-capbound-run$(1)
eval-v222-repair-capbound-run$(1):
	@test -d $(V219_MODEL) || (echo "ERROR: model not found at $(V219_MODEL)" && exit 1)
	@test -d $(V221_INDEX) || (echo "ERROR: v2.19c index not found — run make build-v219c-expanded-index" && exit 1)
	@mkdir -p $(V222_OUT_DIR)
	$$(_v222_eval) --tasks-file $(V219_TASKS_32) \
		--task-ids $(V222_CAPBOUND) \
		--output outputs/eval_v222_repair_capbound_run$(1)
	@echo "v2.22 repair capbound run $(1) complete."
endef
$(foreach n,1 2 3,$(eval $(call V222_CAP_RULE,$(n))))

# Full-32 control (regression guard)
define V222_FULL_RULE
.PHONY: eval-v222-repair-32-run$(1)
eval-v222-repair-32-run$(1):
	@test -d $(V219_MODEL)  || (echo "ERROR: model not found at $(V219_MODEL)" && exit 1)
	@test -f $(V219_TASKS_32) || (echo "ERROR: task file not found" && exit 1)
	@test -d $(V221_INDEX)  || (echo "ERROR: v2.19c index not found" && exit 1)
	@mkdir -p $(V222_OUT_DIR)
	$$(_v222_eval) --tasks-file $(V219_TASKS_32) \
		--output outputs/eval_v222_repair_32_run$(1)
	@echo "v2.22 repair full-32 run $(1) complete."
endef
$(foreach n,1 2 3,$(eval $(call V222_FULL_RULE,$(n))))

summarise-v222-repair:
	@mkdir -p $(V222_OUT_DIR)
	$(ENV) python scripts/summarise_v222_repair.py
	@echo ""
	@echo "v2.22 summary : $(V222_OUT_DIR)/summary.md"
	@echo "Claim boundary: $(V222_OUT_DIR)/claim_boundary.md"

# ── v2.22b Repair-Signal Ablation (raw stderr vs structured VERIFIER) ──────
# Single-variable ablation of v2.22: SAME repair budget + no-repeat + diagnostic-assert
# contract, but the OBSERVATION is RAW stderr instead of the distilled VERIFIER block.
# Attributes the v2.22 aggregate lift (22.0) to signal format vs repair discipline.
#
# Usage:
#   make eval-v222b-repair-capbound-run1 (run2,3) ; eval-v222b-repair-32-run1 (run2,3)
#   make summarise-v222b-ablation

V222B_OUT_DIR := results/v222b_repair_signal_ablation

_v222b_eval = $(ENV) python scripts/evaluate_code_agent.py \
	--hf-model $(V219_MODEL) \
	--mode best_of_n --n 3 \
	--scoring-mode verified_agent \
	--agent-contract strict \
	--stop-after-pass \
	--verifier-repair --repair-raw-stderr \
	--max-repair-iters $(V222_REPAIR_ITERS) \
	--memory-enabled \
	--retrieval-mode structured \
	--structured-index $(V221_INDEX) \
	--dense-model $(V219_ENCODER) \
	--rerank-top-n $(V219_RERANK_N) \
	--memory-top-k 4 \
	--verbose

.PHONY: summarise-v222b-ablation

define V222B_CAP_RULE
.PHONY: eval-v222b-repair-capbound-run$(1)
eval-v222b-repair-capbound-run$(1):
	@test -d $(V219_MODEL) || (echo "ERROR: model not found at $(V219_MODEL)" && exit 1)
	@test -d $(V221_INDEX) || (echo "ERROR: v2.19c index not found — run make build-v219c-expanded-index" && exit 1)
	@mkdir -p $(V222B_OUT_DIR)
	$$(_v222b_eval) --tasks-file $(V219_TASKS_32) \
		--task-ids $(V222_CAPBOUND) \
		--output outputs/eval_v222b_repair_capbound_run$(1)
	@echo "v2.22b raw-repair capbound run $(1) complete."
endef
$(foreach n,1 2 3,$(eval $(call V222B_CAP_RULE,$(n))))

define V222B_FULL_RULE
.PHONY: eval-v222b-repair-32-run$(1)
eval-v222b-repair-32-run$(1):
	@test -d $(V219_MODEL)  || (echo "ERROR: model not found at $(V219_MODEL)" && exit 1)
	@test -f $(V219_TASKS_32) || (echo "ERROR: task file not found" && exit 1)
	@test -d $(V221_INDEX)  || (echo "ERROR: v2.19c index not found" && exit 1)
	@mkdir -p $(V222B_OUT_DIR)
	$$(_v222b_eval) --tasks-file $(V219_TASKS_32) \
		--output outputs/eval_v222b_repair_32_run$(1)
	@echo "v2.22b raw-repair full-32 run $(1) complete."
endef
$(foreach n,1 2 3,$(eval $(call V222B_FULL_RULE,$(n))))

summarise-v222b-ablation:
	@mkdir -p $(V222B_OUT_DIR)
	$(ENV) python scripts/summarise_v222b_ablation.py
	@echo ""
	@echo "v2.22b summary : $(V222B_OUT_DIR)/summary.md"
	@echo "Claim boundary: $(V222B_OUT_DIR)/claim_boundary.md"

# ── v2.23 Targeted Tree Capability Adapter ────────────────────────────────
# A small fresh LoRA trained on contamination-guarded same-family-different-task tree
# repair traces, applied ON TOP of the MERGED champion (no LoRA-on-LoRA; champion frozen).
# Evaluated under the v2.22 structured-verifier mode + v2.19c retrieval. Includes the
# required adapter-WITHOUT-verifier ablation. No champion overwrite; protected indexes intact.
#
# Build the data + adapter first:
#   make build-v223-tree-capability-data
#   make train-v223-tree-adapter
# Then:
#   make eval-v223-hardtree-run1 (run2,3) ; eval-v223-full32-run1 (run2,3)
#   make eval-v223-noverifier-run1 (run2,3) ; make summarise-v223

V223_CHAMPION := $(V219_MODEL)
V223_ADAPTER  := outputs/qwen15b_v223_tree_capability_adapter/final
V223_DATA     := data/v223_tree_capability_records.jsonl
V223_HARDTREE := v210_tree_serialize v210_tree_from_list v210_tree_max_path_sum
V223_OUT_DIR  := results/v223_tree_capability_adapter

.PHONY: build-v223-tree-capability-data train-v223-tree-adapter summarise-v223

build-v223-tree-capability-data:
	$(ENV) python scripts/build_v223_tree_capability_data.py
	$(ENV) python scripts/check_v223_contamination.py

train-v223-tree-adapter:
	@test -s $(V223_DATA) || (echo "ERROR: $(V223_DATA) missing — run make build-v223-tree-capability-data" && exit 1)
	@test -d $(V223_CHAMPION) || (echo "ERROR: merged champion not found at $(V223_CHAMPION)" && exit 1)
	$(ENV) python scripts/finetune_qwen_code_agent.py \
		--hf-model $(V223_CHAMPION) \
		--training-file $(V223_DATA) \
		--agent-contract strict \
		--steps 50 --lr 1e-5 --batch-size 1 --grad-accum 8 --max-length 1024 \
		--output-dir outputs/qwen15b_v223_tree_capability_adapter
	@echo "v2.23 adapter trained: $(V223_ADAPTER) (champion NOT modified)"

# adapter + structured verifier (the promoted inference config)
_v223_eval = $(ENV) python scripts/evaluate_code_agent.py \
	--hf-model $(V223_CHAMPION) --hf-lora $(V223_ADAPTER) \
	--tasks-file $(V219_TASKS_32) \
	--mode best_of_n --n 3 --scoring-mode verified_agent --agent-contract strict --stop-after-pass \
	--verifier-repair --max-repair-iters 3 \
	--memory-enabled --retrieval-mode structured \
	--structured-index $(V221_INDEX) --dense-model $(V219_ENCODER) \
	--rerank-top-n $(V219_RERANK_N) --memory-top-k 4 --verbose

# ablation: adapter WITHOUT the structured verifier (plain execution-plan mode)
_v223_eval_noverif = $(ENV) python scripts/evaluate_code_agent.py \
	--hf-model $(V223_CHAMPION) --hf-lora $(V223_ADAPTER) \
	--tasks-file $(V219_TASKS_32) \
	--mode best_of_n --n 3 --scoring-mode verified_agent --agent-contract strict --stop-after-pass \
	--execution-plan-mode \
	--memory-enabled --retrieval-mode structured \
	--structured-index $(V221_INDEX) --dense-model $(V219_ENCODER) \
	--rerank-top-n $(V219_RERANK_N) --memory-top-k 4 --verbose

define V223_RULE
.PHONY: eval-v223-hardtree-run$(1) eval-v223-full32-run$(1) eval-v223-noverifier-run$(1)
eval-v223-hardtree-run$(1):
	@test -d $(V223_ADAPTER) || (echo "ERROR: adapter not found — run make train-v223-tree-adapter" && exit 1)
	@mkdir -p $(V223_OUT_DIR)
	$$(_v223_eval) --task-ids $(V223_HARDTREE) --output outputs/eval_v223_hardtree_run$(1)
	@echo "v2.23 hardtree run $(1) complete."
eval-v223-full32-run$(1):
	@test -d $(V223_ADAPTER) || (echo "ERROR: adapter not found — run make train-v223-tree-adapter" && exit 1)
	@mkdir -p $(V223_OUT_DIR)
	$$(_v223_eval) --output outputs/eval_v223_full32_run$(1)
	@echo "v2.23 full32 run $(1) complete."
eval-v223-noverifier-run$(1):
	@test -d $(V223_ADAPTER) || (echo "ERROR: adapter not found — run make train-v223-tree-adapter" && exit 1)
	@mkdir -p $(V223_OUT_DIR)
	$$(_v223_eval_noverif) --task-ids $(V223_HARDTREE) --output outputs/eval_v223_noverifier_run$(1)
	@echo "v2.23 no-verifier ablation run $(1) complete."
endef
$(foreach n,1 2 3,$(eval $(call V223_RULE,$(n))))

summarise-v223:
	@mkdir -p $(V223_OUT_DIR)
	$(ENV) python scripts/summarise_v223_adapter.py
	@echo ""
	@echo "v2.23 summary : $(V223_OUT_DIR)/summary.md"
	@echo "Claim boundary: $(V223_OUT_DIR)/claim_boundary.md"

# ── v2.23b Scaled Tree Capability Adapter ─────────────────────────────────
# Scales the v2.23 pilot: ~2x data (94 records, 27 tasks) + 3x steps (150). Same
# contamination guard, regression gate, separate adapter, champion untouched. Tests
# whether the residual hard tasks are data-limited or at a capability ceiling.
#
#   make build-v223b-scaled-data ; make train-v223b-scaled-adapter
#   make eval-v223b-hardtree-run1 (2,3) ; eval-v223b-full32-run1 (2,3) ; eval-v223b-noverifier-run1 (2,3)
#   make summarise-v223b

V223B_DATA    := data/v223b_tree_capability_scaled.jsonl
V223B_ADAPTER := outputs/qwen15b_v223b_scaled_adapter/final
V223B_OUT_DIR := results/v223b_scaled_tree_capability

.PHONY: build-v223b-scaled-data train-v223b-scaled-adapter summarise-v223b

build-v223b-scaled-data:
	$(ENV) python scripts/build_v223b_scaled_data.py
	$(ENV) python scripts/check_v223_contamination.py

train-v223b-scaled-adapter:
	@test -s $(V223B_DATA)    || (echo "ERROR: $(V223B_DATA) missing — run make build-v223b-scaled-data" && exit 1)
	@test -d $(V223_CHAMPION) || (echo "ERROR: merged champion not found" && exit 1)
	$(ENV) python scripts/finetune_qwen_code_agent.py \
		--hf-model $(V223_CHAMPION) --training-file $(V223B_DATA) \
		--agent-contract strict \
		--steps 150 --lr 1e-5 --batch-size 1 --grad-accum 8 --max-length 1024 \
		--output-dir outputs/qwen15b_v223b_scaled_adapter
	@echo "v2.23b scaled adapter trained: $(V223B_ADAPTER) (champion NOT modified)"

_v223b_eval = $(ENV) python scripts/evaluate_code_agent.py \
	--hf-model $(V223_CHAMPION) --hf-lora $(V223B_ADAPTER) \
	--tasks-file $(V219_TASKS_32) \
	--mode best_of_n --n 3 --scoring-mode verified_agent --agent-contract strict --stop-after-pass \
	--verifier-repair --max-repair-iters 3 \
	--memory-enabled --retrieval-mode structured \
	--structured-index $(V221_INDEX) --dense-model $(V219_ENCODER) \
	--rerank-top-n $(V219_RERANK_N) --memory-top-k 4 --verbose

_v223b_eval_noverif = $(ENV) python scripts/evaluate_code_agent.py \
	--hf-model $(V223_CHAMPION) --hf-lora $(V223B_ADAPTER) \
	--tasks-file $(V219_TASKS_32) \
	--mode best_of_n --n 3 --scoring-mode verified_agent --agent-contract strict --stop-after-pass \
	--execution-plan-mode \
	--memory-enabled --retrieval-mode structured \
	--structured-index $(V221_INDEX) --dense-model $(V219_ENCODER) \
	--rerank-top-n $(V219_RERANK_N) --memory-top-k 4 --verbose

define V223B_RULE
.PHONY: eval-v223b-hardtree-run$(1) eval-v223b-full32-run$(1) eval-v223b-noverifier-run$(1)
eval-v223b-hardtree-run$(1):
	@test -d $(V223B_ADAPTER) || (echo "ERROR: adapter not found — run make train-v223b-scaled-adapter" && exit 1)
	@mkdir -p $(V223B_OUT_DIR)
	$$(_v223b_eval) --task-ids $(V223_HARDTREE) --output outputs/eval_v223b_hardtree_run$(1)
	@echo "v2.23b hardtree run $(1) complete."
eval-v223b-full32-run$(1):
	@test -d $(V223B_ADAPTER) || (echo "ERROR: adapter not found — run make train-v223b-scaled-adapter" && exit 1)
	@mkdir -p $(V223B_OUT_DIR)
	$$(_v223b_eval) --output outputs/eval_v223b_full32_run$(1)
	@echo "v2.23b full32 run $(1) complete."
eval-v223b-noverifier-run$(1):
	@test -d $(V223B_ADAPTER) || (echo "ERROR: adapter not found — run make train-v223b-scaled-adapter" && exit 1)
	@mkdir -p $(V223B_OUT_DIR)
	$$(_v223b_eval_noverif) --task-ids $(V223_HARDTREE) --output outputs/eval_v223b_noverifier_run$(1)
	@echo "v2.23b no-verifier ablation run $(1) complete."
endef
$(foreach n,1 2 3,$(eval $(call V223B_RULE,$(n))))

summarise-v223b:
	@mkdir -p $(V223B_OUT_DIR)
	$(ENV) python scripts/summarise_v223b_scaled.py
	@echo ""
	@echo "v2.23b summary : $(V223B_OUT_DIR)/summary.md"
	@echo "Claim boundary: $(V223B_OUT_DIR)/claim_boundary.md"

# ── v2.24 3B-Scale Capability Ceiling Test ────────────────────────────────
# Tests whether ~2x model scale (1.5B -> 3B) moves the residual capability ceiling on the 3
# hard tree tasks. Same strict protocol: v2.22 structured verifier + v2.19c retrieval, the
# v2.23b contamination-guarded data, regression gate, with/without-verifier ablation. The
# frozen 1.5B champion is untouched; this is a separate 3B base + adapter.
#
#   make eval-v224-base-hardtree-run1 (2,3) ; eval-v224-base-full32-run1 (2,3)
#   make train-v224-3b-adapter
#   make eval-v224-adapter-hardtree-run1 (2,3) ; eval-v224-adapter-full32-run1 (2,3)
#   make eval-v224-adapter-noverifier-run1 (2,3) ; make summarise-v224

V224_BASE     := Qwen/Qwen2.5-Coder-3B-Instruct
V224_ADAPTER  := outputs/qwen3b_v224_tree_adapter/final
V224_DATA     := $(V223B_DATA)
V224_OUT_DIR  := results/v224_3b_scale_test

.PHONY: train-v224-3b-adapter summarise-v224

train-v224-3b-adapter:
	@test -s $(V224_DATA) || (echo "ERROR: $(V224_DATA) missing — run make build-v223b-scaled-data" && exit 1)
	PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $(ENV) python scripts/finetune_qwen_code_agent.py \
		--hf-model $(V224_BASE) --training-file $(V224_DATA) \
		--agent-contract strict \
		--steps 150 --lr 1e-5 --batch-size 1 --grad-accum 8 --max-length 512 \
		--output-dir outputs/qwen3b_v224_tree_adapter
	@echo "v2.24 3B adapter trained: $(V224_ADAPTER) (1.5B champion untouched; max-length 512 for 16GB VRAM)"

# common eval flags (verifier mode + v2.19c retrieval); $(1)=extra args
_v224_common = --tasks-file $(V219_TASKS_32) \
	--mode best_of_n --n 3 --scoring-mode verified_agent --agent-contract strict --stop-after-pass \
	--memory-enabled --retrieval-mode structured \
	--structured-index $(V221_INDEX) --dense-model $(V219_ENCODER) \
	--rerank-top-n $(V219_RERANK_N) --memory-top-k 4 --verbose

# 3B BASE (no adapter) under verifier mode
define V224_BASE_RULE
.PHONY: eval-v224-base-hardtree-run$(1) eval-v224-base-full32-run$(1)
eval-v224-base-hardtree-run$(1):
	@mkdir -p $(V224_OUT_DIR)
	$$(ENV) python scripts/evaluate_code_agent.py --hf-model $(V224_BASE) \
		$(_v224_common) --verifier-repair --max-repair-iters 3 \
		--task-ids $(V223_HARDTREE) --output outputs/eval_v224_base_hardtree_run$(1)
	@echo "v2.24 base hardtree run $(1) complete."
eval-v224-base-full32-run$(1):
	@mkdir -p $(V224_OUT_DIR)
	$$(ENV) python scripts/evaluate_code_agent.py --hf-model $(V224_BASE) \
		$(_v224_common) --verifier-repair --max-repair-iters 3 \
		--output outputs/eval_v224_base_full32_run$(1)
	@echo "v2.24 base full32 run $(1) complete."
endef
$(foreach n,1 2 3,$(eval $(call V224_BASE_RULE,$(n))))

# 3B BASE + v2.24 adapter
define V224_ADAPTER_RULE
.PHONY: eval-v224-adapter-hardtree-run$(1) eval-v224-adapter-full32-run$(1) eval-v224-adapter-noverifier-run$(1)
eval-v224-adapter-hardtree-run$(1):
	@test -d $(V224_ADAPTER) || (echo "ERROR: adapter not found — run make train-v224-3b-adapter" && exit 1)
	@mkdir -p $(V224_OUT_DIR)
	$$(ENV) python scripts/evaluate_code_agent.py --hf-model $(V224_BASE) --hf-lora $(V224_ADAPTER) \
		$(_v224_common) --verifier-repair --max-repair-iters 3 \
		--task-ids $(V223_HARDTREE) --output outputs/eval_v224_adapter_hardtree_run$(1)
	@echo "v2.24 adapter hardtree run $(1) complete."
eval-v224-adapter-full32-run$(1):
	@test -d $(V224_ADAPTER) || (echo "ERROR: adapter not found — run make train-v224-3b-adapter" && exit 1)
	@mkdir -p $(V224_OUT_DIR)
	$$(ENV) python scripts/evaluate_code_agent.py --hf-model $(V224_BASE) --hf-lora $(V224_ADAPTER) \
		$(_v224_common) --verifier-repair --max-repair-iters 3 \
		--output outputs/eval_v224_adapter_full32_run$(1)
	@echo "v2.24 adapter full32 run $(1) complete."
eval-v224-adapter-noverifier-run$(1):
	@test -d $(V224_ADAPTER) || (echo "ERROR: adapter not found — run make train-v224-3b-adapter" && exit 1)
	@mkdir -p $(V224_OUT_DIR)
	$$(ENV) python scripts/evaluate_code_agent.py --hf-model $(V224_BASE) --hf-lora $(V224_ADAPTER) \
		$(_v224_common) --execution-plan-mode \
		--task-ids $(V223_HARDTREE) --output outputs/eval_v224_adapter_noverifier_run$(1)
	@echo "v2.24 adapter no-verifier run $(1) complete."
endef
$(foreach n,1 2 3,$(eval $(call V224_ADAPTER_RULE,$(n))))

summarise-v224:
	@mkdir -p $(V224_OUT_DIR)
	$(ENV) python scripts/summarise_v224_scale.py
	@echo ""
	@echo "v2.24 summary : $(V224_OUT_DIR)/summary.md"
	@echo "Claim boundary: $(V224_OUT_DIR)/claim_boundary.md"

# ── v2.25 7B-Scale Capability Test (QLoRA / 4-bit) ─────────────────────────
# Extends the v2.24 scale curve to 7B (Qwen2.5-Coder-7B-Instruct) via 4-bit NF4 (fits 16GB).
# Same strict protocol; tests whether the lone holdout (tree_serialize) cracks at 7B and how
# the scale trend continues. 1.5B champion + 3B artifacts untouched.
#
#   make eval-v225-base-hardtree-run1 (2,3) ; eval-v225-base-full32-run1 (2,3)
#   make train-v225-7b-adapter
#   make eval-v225-adapter-hardtree-run1 (2,3) ; eval-v225-adapter-full32-run1 (2,3)
#   make eval-v225-adapter-noverifier-run1 (2,3) ; make summarise-v225

V225_BASE     := Qwen/Qwen2.5-Coder-7B-Instruct
V225_ADAPTER  := outputs/qwen7b_v225_tree_adapter/final
V225_OUT_DIR  := results/v225_7b_scale_test

.PHONY: train-v225-7b-adapter summarise-v225

train-v225-7b-adapter:
	@test -s $(V224_DATA) || (echo "ERROR: $(V224_DATA) missing — run make build-v223b-scaled-data" && exit 1)
	PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $(ENV) python scripts/finetune_qwen_code_agent.py \
		--hf-model $(V225_BASE) --training-file $(V224_DATA) \
		--agent-contract strict --load-in-4bit \
		--steps 150 --lr 1e-5 --batch-size 1 --grad-accum 8 --max-length 512 \
		--output-dir outputs/qwen7b_v225_tree_adapter
	@echo "v2.25 7B QLoRA adapter trained: $(V225_ADAPTER) (1.5B champion + 3B untouched)"

_v225_common = --tasks-file $(V219_TASKS_32) --load-in-4bit \
	--mode best_of_n --n 3 --scoring-mode verified_agent --agent-contract strict --stop-after-pass \
	--memory-enabled --retrieval-mode structured \
	--structured-index $(V221_INDEX) --dense-model $(V219_ENCODER) \
	--rerank-top-n $(V219_RERANK_N) --memory-top-k 4 --verbose

define V225_BASE_RULE
.PHONY: eval-v225-base-hardtree-run$(1) eval-v225-base-full32-run$(1)
eval-v225-base-hardtree-run$(1):
	@mkdir -p $(V225_OUT_DIR)
	$$(ENV) python scripts/evaluate_code_agent.py --hf-model $(V225_BASE) \
		$(_v225_common) --verifier-repair --max-repair-iters 3 \
		--task-ids $(V223_HARDTREE) --output outputs/eval_v225_base_hardtree_run$(1)
	@echo "v2.25 base hardtree run $(1) complete."
eval-v225-base-full32-run$(1):
	@mkdir -p $(V225_OUT_DIR)
	$$(ENV) python scripts/evaluate_code_agent.py --hf-model $(V225_BASE) \
		$(_v225_common) --verifier-repair --max-repair-iters 3 \
		--output outputs/eval_v225_base_full32_run$(1)
	@echo "v2.25 base full32 run $(1) complete."
endef
$(foreach n,1 2 3,$(eval $(call V225_BASE_RULE,$(n))))

define V225_ADAPTER_RULE
.PHONY: eval-v225-adapter-hardtree-run$(1) eval-v225-adapter-full32-run$(1) eval-v225-adapter-noverifier-run$(1)
eval-v225-adapter-hardtree-run$(1):
	@test -d $(V225_ADAPTER) || (echo "ERROR: adapter not found — run make train-v225-7b-adapter" && exit 1)
	@mkdir -p $(V225_OUT_DIR)
	$$(ENV) python scripts/evaluate_code_agent.py --hf-model $(V225_BASE) --hf-lora $(V225_ADAPTER) \
		$(_v225_common) --verifier-repair --max-repair-iters 3 \
		--task-ids $(V223_HARDTREE) --output outputs/eval_v225_adapter_hardtree_run$(1)
	@echo "v2.25 adapter hardtree run $(1) complete."
eval-v225-adapter-full32-run$(1):
	@test -d $(V225_ADAPTER) || (echo "ERROR: adapter not found — run make train-v225-7b-adapter" && exit 1)
	@mkdir -p $(V225_OUT_DIR)
	$$(ENV) python scripts/evaluate_code_agent.py --hf-model $(V225_BASE) --hf-lora $(V225_ADAPTER) \
		$(_v225_common) --verifier-repair --max-repair-iters 3 \
		--output outputs/eval_v225_adapter_full32_run$(1)
	@echo "v2.25 adapter full32 run $(1) complete."
eval-v225-adapter-noverifier-run$(1):
	@test -d $(V225_ADAPTER) || (echo "ERROR: adapter not found — run make train-v225-7b-adapter" && exit 1)
	@mkdir -p $(V225_OUT_DIR)
	$$(ENV) python scripts/evaluate_code_agent.py --hf-model $(V225_BASE) --hf-lora $(V225_ADAPTER) \
		$(_v225_common) --execution-plan-mode \
		--task-ids $(V223_HARDTREE) --output outputs/eval_v225_adapter_noverifier_run$(1)
	@echo "v2.25 adapter no-verifier run $(1) complete."
endef
$(foreach n,1 2 3,$(eval $(call V225_ADAPTER_RULE,$(n))))

summarise-v225:
	@mkdir -p $(V225_OUT_DIR)
	$(ENV) python scripts/summarise_v225_scale.py
	@echo ""
	@echo "v2.25 summary : $(V225_OUT_DIR)/summary.md"
	@echo "Claim boundary: $(V225_OUT_DIR)/claim_boundary.md"

# ── v2.26 Self-Improving Trace Factory + tree_serialize Representation Attack ──
# Part A: does tree_serialize fail because of brittle EXACT-STRING format or true
# capability? The same 3 logical tree-serializations in 4 output representations
# (exact_string/token_list/nested_list/json) are eval'd at 3B-bf16 (clean config) + verifier.
# Part B: a source-only trace factory records full agentic repair trajectories (local-only).
# Part C: trace-quality + format-vs-capability summariser. No held-out task is trained on.
#
#   make build-v226-representation-tasks
#   make eval-v226-3b-run1 (2,3) ; [optional] eval-v226-7b-run1 (2,3)
#   make build-v226-traces ; make summarise-v226

V226_TASKS   := data/v226_representation_tasks.jsonl
V226_3B      := Qwen/Qwen2.5-Coder-3B-Instruct
V226_7B      := Qwen/Qwen2.5-Coder-7B-Instruct
V226_OUT_DIR := results/v226_self_improving_traces

.PHONY: build-v226-representation-tasks build-v226-traces summarise-v226

build-v226-representation-tasks:
	$(ENV) python scripts/build_v226_representation_tasks.py

_v226_common = --tasks-file $(V226_TASKS) \
	--mode best_of_n --n 3 --scoring-mode verified_agent --agent-contract strict --stop-after-pass \
	--verifier-repair --max-repair-iters 3 \
	--memory-enabled --retrieval-mode structured \
	--structured-index $(V221_INDEX) --dense-model $(V219_ENCODER) \
	--rerank-top-n $(V219_RERANK_N) --memory-top-k 4 --verbose

define V226_RULE
.PHONY: eval-v226-3b-run$(1) eval-v226-7b-run$(1)
eval-v226-3b-run$(1):
	@test -s $(V226_TASKS) || (echo "ERROR: run make build-v226-representation-tasks" && exit 1)
	@mkdir -p $(V226_OUT_DIR)
	$$(ENV) python scripts/evaluate_code_agent.py --hf-model $(V226_3B) \
		$(_v226_common) --output outputs/eval_v226_3b_run$(1)
	@echo "v2.26 3B-bf16 representation run $(1) complete."
eval-v226-7b-run$(1):
	@test -s $(V226_TASKS) || (echo "ERROR: run make build-v226-representation-tasks" && exit 1)
	@mkdir -p $(V226_OUT_DIR)
	$$(ENV) python scripts/evaluate_code_agent.py --hf-model $(V226_7B) --load-in-4bit \
		$(_v226_common) --output outputs/eval_v226_7b_run$(1)
	@echo "v2.26 7B-4bit (quantization-confounded) representation run $(1) complete."
endef
$(foreach n,1 2 3,$(eval $(call V226_RULE,$(n))))

build-v226-traces:
	$(ENV) python scripts/build_v226_trace_factory.py

summarise-v226:
	@mkdir -p $(V226_OUT_DIR)
	$(ENV) python scripts/summarise_v226_tree_serialize.py
	@echo ""
	@echo "v2.26 summary : $(V226_OUT_DIR)/summary.md"
	@echo "Claim boundary: $(V226_OUT_DIR)/claim_boundary.md"

# ── v2.27 Format-Robust Output Control ───────────────────────────────────
# Evaluation + infrastructure milestone (NOT training): hardens the trace factory to record
# genuine repair transitions, adds a deterministic tree-serialization format verifier, and tests
# whether a canonical intermediate representation + structured format verification yields stable
# tree_serialize conversion. All deterministic / model-free; the model baseline is mined from the
# existing v2.26 transcripts. Generated traces are local-only (data/generated/v227, gitignored).
V227_OUT_DIR := results/v227_format_robust
.PHONY: v227-format-verifier build-v227-traces eval-v227-format-control summarise-v227 v227 test-v227

v227-format-verifier:
	$(ENV) python scripts/v227_format_verifier.py

build-v227-traces:
	$(ENV) python scripts/build_v227_trace_factory.py

eval-v227-format-control:
	@mkdir -p $(V227_OUT_DIR)
	$(ENV) python scripts/v227_format_control_eval.py

summarise-v227: eval-v227-format-control
	@echo ""
	@echo "v2.27 summary : $(V227_OUT_DIR)/summary.md"
	@echo "Claim boundary: $(V227_OUT_DIR)/claim_boundary.md"

# Full v2.27 pipeline: hardened traces (local-only) -> format-control eval -> curated summary.
v227: build-v227-traces summarise-v227

test-v227:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $(ENV) python -m pytest tests/test_v227_format_verifier.py -v

# ── v2.28 Self-Improving Trace Dataset ───────────────────────────────────
# Dataset infrastructure (NOT training): normalises the local v2.26/v2.27 traces into one canonical,
# contamination-guarded, quality-scored self-improvement dataset schema for LATER SFT / preference /
# scaffold training. The generated dataset is local-only (data/generated/v228, gitignored); only the
# curated summary under results/v228_self_improving_dataset is committed.
V228_OUT_DIR := results/v228_self_improving_dataset
.PHONY: build-v228-dataset summarise-v228 v228 test-v228

build-v228-dataset:
	$(ENV) python scripts/build_v228_self_improving_dataset.py

summarise-v228: build-v228-dataset
	@mkdir -p $(V228_OUT_DIR)
	$(ENV) python scripts/summarise_v228_self_improving_dataset.py
	@echo ""
	@echo "v2.28 summary : $(V228_OUT_DIR)/summary.md"
	@echo "Claim boundary: $(V228_OUT_DIR)/claim_boundary.md"

# Full v2.28 pipeline: local-only dataset -> curated summary.
v228: summarise-v228

test-v228:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $(ENV) python -m pytest tests/test_v228_trace_schema.py -v

# ── v2.29 Genuine Repair Trace Harvest ───────────────────────────────────
# Trace harvest (NOT training): manufactures genuine execution-verified, verifier-labelled
# FORMAT-repair transitions (candidate fail -> verifier signal -> canonical repair -> verified pass)
# by perturbing ONLY the output format of known-correct serializers; held-out evaluation untouched.
# Output is local-only (data/generated/v229, gitignored). Feeding the harvest through the v2.28
# builder yields non-zero format-repair and verifier-format candidates.
V229_OUT_DIR := results/v229_repair_harvest
.PHONY: build-v229-harvest summarise-v229 v229 test-v229

build-v229-harvest:
	$(ENV) python scripts/build_v229_repair_harvest.py

# Rebuild the v2.28 dataset (now picking up the v2.29 traces) then summarise the harvest evidence.
summarise-v229: build-v229-harvest build-v228-dataset
	@mkdir -p $(V229_OUT_DIR)
	$(ENV) python scripts/summarise_v229_repair_harvest.py
	@echo ""
	@echo "v2.29 summary : $(V229_OUT_DIR)/summary.md"
	@echo "Claim boundary: $(V229_OUT_DIR)/claim_boundary.md"

v229: summarise-v229

test-v229:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $(ENV) python -m pytest tests/test_v229_repair_harvest.py -v

# ── v2.30 Broadened Repair Trace Harvest ─────────────────────────────────
# Trace harvest (NOT training): broadens the genuine repair corpus across more families and adds a
# controlled algorithmic repair slice on NEW non-held-out tasks. Output is local-only
# (data/generated/v230, gitignored). Feeding through the v2.28 builder yields non-zero format-repair
# AND algorithmic-repair candidates.
V230_OUT_DIR := results/v230_broadened_repair_harvest
.PHONY: build-v230-harvest summarise-v230 v230 test-v230

build-v230-harvest:
	$(ENV) python scripts/build_v230_broadened_repair_harvest.py

summarise-v230: build-v230-harvest build-v228-dataset
	@mkdir -p $(V230_OUT_DIR)
	$(ENV) python scripts/summarise_v230_broadened_repair_harvest.py
	@echo ""
	@echo "v2.30 summary : $(V230_OUT_DIR)/summary.md"
	@echo "Claim boundary: $(V230_OUT_DIR)/claim_boundary.md"

v230: summarise-v230

test-v230:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $(ENV) python -m pytest tests/test_v230_repair_harvest.py -v

# ── v2.31 Tiny Repair-Trace SFT Pilot (first training milestone) ──────────
# Phase 1 (dataset export) runs anywhere; Phase 2 (train) and Phase 3 (eval) are GPU-gated and skip
# cleanly without a CUDA device (no fabricated metrics). All training artifacts are local-only
# (data/generated/v231, outputs/v231_tiny_repair_trace_sft, gitignored). The frozen champion and
# prior adapters are never overwritten (separate output path).
V231_OUT_DIR := results/v231_tiny_repair_trace_sft
.PHONY: build-v231-sft-dataset train-v231 eval-v231 summarise-v231 v231 test-v231

build-v231-sft-dataset:
	$(ENV) python scripts/build_v231_repair_sft_dataset.py

train-v231: build-v231-sft-dataset
	$(ENV) python scripts/train_v231_repair_sft.py

eval-v231:
	$(ENV) python scripts/eval_v231_repair_sft.py

summarise-v231: build-v231-sft-dataset
	@mkdir -p $(V231_OUT_DIR)
	$(ENV) python scripts/summarise_v231_repair_sft.py
	@echo ""
	@echo "v2.31 summary : $(V231_OUT_DIR)/summary.md"
	@echo "Claim boundary: $(V231_OUT_DIR)/claim_boundary.md"

v231: summarise-v231

# End-to-end GPU runbook: export -> train 1.5B LoRA smoke -> eval -> summarise. Refuses without a
# CUDA GPU; protects the champion + memory indexes (read-only tamper check); outputs stay local-only.
#   make v231-gpu-runbook                       # defaults: 1.5B base, 60 steps
#   bash scripts/run_v231_gpu_runbook.sh <base> <max_steps>
.PHONY: v231-gpu-runbook
v231-gpu-runbook:
	bash scripts/run_v231_gpu_runbook.sh

test-v231:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $(ENV) python -m pytest tests/test_v231_repair_sft.py -v

# ── v2.32 Tool-Use Preservation During Repair-Trace Adaptation ───────────
# Mixed repair + tool-use-preservation split-loss SFT pilot. Phase 1 (mixed dataset) runs anywhere;
# train/eval/benchmark are GPU-gated and skip cleanly without CUDA (no fabricated metrics). Artifacts
# local-only (data/generated/v232, outputs/v232_tool_use_preservation_sft, gitignored). Champion +
# prior adapters never overwritten (separate output path).
V232_OUT_DIR := results/v232_tool_use_preservation
.PHONY: build-v232-mixed-dataset train-v232 eval-v232 summarise-v232 v232 v232-gpu-runbook test-v232

build-v232-mixed-dataset:
	$(ENV) python scripts/build_v232_mixed_dataset.py

train-v232: build-v232-mixed-dataset
	$(ENV) python scripts/train_v232_mixed_sft.py

eval-v232:
	$(ENV) python scripts/eval_v232_mixed_sft.py

summarise-v232: build-v232-mixed-dataset
	@mkdir -p $(V232_OUT_DIR)
	$(ENV) python scripts/summarise_v232_mixed_sft.py
	@echo ""
	@echo "v2.32 summary : $(V232_OUT_DIR)/summary.md"
	@echo "Claim boundary: $(V232_OUT_DIR)/claim_boundary.md"

v232: summarise-v232

# End-to-end GPU runbook: build -> train -> eval -> benchmark gate -> summarise (refuses w/o CUDA).
v232-gpu-runbook:
	bash scripts/run_v232_gpu_runbook.sh

test-v232:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $(ENV) python -m pytest tests/test_v232_mixed_sft.py -v

# ── v2.33 Scaffold-First Tool-Call Preservation ──────────────────────────
# Scaffold-only (no-repair) tool-call preservation SFT. Isolates the v2.31/v2.32 failure mode
# (no_tool_call collapse): preserve execute_code/tool-use + the frozen 32-task benchmark BEFORE repair
# traces return. Phase 1 (dataset) runs anywhere; train/eval/benchmark are GPU-gated and skip cleanly
# without CUDA (no fabricated metrics). Artifacts local-only (data/generated/v233,
# outputs/v233_scaffold_first_sft, gitignored). Champion + prior adapters never overwritten.
V233_OUT_DIR := results/v233_scaffold_first
.PHONY: build-v233-scaffold-dataset train-v233 eval-v233 summarise-v233 v233 v233-gpu-runbook test-v233

build-v233-scaffold-dataset:
	$(ENV) python scripts/build_v233_scaffold_dataset.py

train-v233: build-v233-scaffold-dataset
	$(ENV) python scripts/train_v233_scaffold_sft.py

eval-v233:
	$(ENV) python scripts/eval_v233_scaffold_sft.py

summarise-v233: build-v233-scaffold-dataset
	@mkdir -p $(V233_OUT_DIR)
	$(ENV) python scripts/summarise_v233_scaffold_sft.py
	@echo ""
	@echo "v2.33 summary : $(V233_OUT_DIR)/summary.md"
	@echo "Claim boundary: $(V233_OUT_DIR)/claim_boundary.md"

v233: summarise-v233

v233-gpu-runbook:
	bash scripts/run_v233_gpu_runbook.sh

test-v233:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $(ENV) python -m pytest tests/test_v233_scaffold_sft.py -v

# ── Unit test suite ──────────────────────────────────────────────────────
# Deterministic, dependency-light tests (no GPU / model / network). Plugin autoload is disabled
# to avoid system pytest plugins (e.g. ROS launch_testing) polluting collection.
.PHONY: test
test:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $(ENV) python -m pytest tests/test_v227_format_verifier.py tests/test_v228_trace_schema.py tests/test_v229_repair_harvest.py tests/test_v230_repair_harvest.py tests/test_v231_repair_sft.py tests/test_v232_mixed_sft.py tests/test_v233_scaffold_sft.py tests/test_heldout_tasks.py -v

# ── v2.14 Documentation, Attribution Audit, Notebook ─────────────────────
.PHONY: audit-attribution render-readme-check notebook-smoke v214-docs-check

# Attribution audit — verifies only Frank Asante Van Laarhoven appears as author.
# Logic is in scripts/audit_attribution.sh to keep patterns out of this file.
audit-attribution:
	@bash scripts/audit_attribution.sh

# Render README check — verifies README contains key evidence markers.
render-readme-check:
	@echo "=== README content check ==="
	@grep -q "23/28 = 82.1%" README.md || (echo "FAIL: champion score missing from README" && exit 1)
	@grep -q "27/28 = 96.4%" README.md || (echo "FAIL: diagnostic score missing from README" && exit 1)
	@grep -q "Diagnostic only" README.md || (echo "FAIL: diagnostic label missing from README" && exit 1)
	@grep -q "evidence_closure_certificate" README.md || (echo "FAIL: evidence closure link missing" && exit 1)
	@grep -q "architecture_overview" README.md || (echo "FAIL: architecture link missing" && exit 1)
	@echo "README check PASSED"

# Notebook smoke-test — verify notebook parses and all cells are valid JSON.
notebook-smoke:
	@echo "=== Notebook smoke-test ==="
	@python -c "import json; nb=json.load(open('notebooks/AetherForge_Forensic_Memory_Audit.ipynb')); [setattr(nb,'_',None) or __import__('sys').exit(1) if c.get('cell_type') not in ('markdown','code') else None for c in nb['cells']]; print(f'Notebook OK: {len(nb[\"cells\"])} cells, all valid')"
	@echo "Notebook smoke-test PASSED"

# v2.14 docs check — verify all new docs exist and are non-empty.
v214-docs-check:
	@echo "=== v2.14 docs check ==="
	@for f in \
	  docs/architecture_overview.md \
	  docs/architecture_diagram_ascii.md \
	  docs/benchmark_registry.md \
	  docs/model_and_memory_registry.md \
	  docs/dataset_provenance.md \
	  docs/evidence_closure_certificate.md \
	  notebooks/AetherForge_Forensic_Memory_Audit.ipynb \
	  paper/aetherforge_memory_augmented_code_agent_draft.md \
	  paper/tables/main_results_table.md \
	  paper/figures/experiment_timeline_spec.md \
	  paper/figures/retrieval_failure_taxonomy_spec.md; do \
	    test -s "$$f" || (echo "FAIL: missing or empty: $$f" && exit 1); \
	    echo "  OK: $$f"; \
	done
	@echo "v2.14 docs check PASSED"

# ── Help ──────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "AetherForge Makefile"
	@echo "========================================================"
	@echo "  make test-all                         Run all smoke-tests"
	@echo "  make data                             Generate synthetic training data"
	@echo "  make data-code                        Generate code-agent training data (~13k)"
	@echo ""
	@echo "  make train-aetherforge                Pretrain 128M AetherForge (JSONL)"
	@echo "  make train-aetherforge-gc             Pretrain 1B with gradient checkpointing"
	@echo "  make train-fineweb                    Stream FineWeb -> AetherForge 128M"
	@echo "  make train-fineweb-ddp                DDP FineWeb (NGPU=4 make train-fineweb-ddp)"
	@echo ""
	@echo "  make finetune                         Fine-tune Qwen2.5-VL-7B (text)"
	@echo "  make finetune-multimodal              Fine-tune Qwen2.5-VL-7B (vision)"
	@echo ""
	@echo "  make finetune-qwen-code-agent-test    Smoke-test (Qwen2.5-0.5B LoRA, 25 steps)"
	@echo "  make finetune-qwen-code-agent         Full LoRA fine-tune Qwen2.5-0.5B"
	@echo "  make finetune-qwen-code-agent-wandb   Same + W&B logging"
	@echo "  make agent-loop-qwen                  Interactive agent (fine-tuned)"
	@echo "  make agent-loop-qwen-benchmark        5-task agent benchmark"
	@echo ""
	@echo "  make eval-code-agent                  Evaluate code agent (single-pass)"
	@echo "  make eval-code-agent-best-of-n        Evaluate with Best-of-3"
	@echo "  make eval-code-agent-compare          Base vs fine-tuned vs best-of-3"
	@echo ""
	@echo "  make distill-test                     Distillation smoke-test (25 steps)"
	@echo "  make distill                          Full distillation run (5000 steps)"
	@echo "  make diagram                          Regenerate docs/architecture.png"
	@echo "  make merge-lora                       Merge LoRA weights into base model"
	@echo ""
	@echo "  make serve                            FastAPI server -- AetherForge 128M :8000"
	@echo "  make serve-1b                         FastAPI server -- AetherForge 1B-8k :8000"
	@echo "  make serve-qwen                       FastAPI server -- Qwen2.5-VL + LoRA :8000"
	@echo "  make chat                             Interactive chat -- AetherForge (GPU)"
	@echo "  make chat-qwen                        Interactive chat -- Qwen2.5-VL (GPU)"
	@echo "  make infer                            Interactive Qwen2.5-VL inference"
	@echo "  make infer-4bit                       Interactive Llama-3.1-8B 4-bit inference"
	@echo "  make env-create                       Create conda env from environment.yml"
	@echo ""
	@echo "  make audit-attribution                Audit git history and files for attribution"
	@echo "  make render-readme-check              Verify README contains key evidence markers"
	@echo "  make notebook-smoke                   Verify notebook parses correctly"
	@echo "  make v214-docs-check                  Verify all v2.14 documentation files exist"
	@echo ""
