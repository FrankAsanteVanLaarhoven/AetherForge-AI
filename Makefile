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
