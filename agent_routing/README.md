# Agent Routing — Learning When to Commit

This repository trains a Qwen3-8B manager to decide whether to commit to its
current answer or consult one of three frozen advisors:

```text
delegate(extractor) | delegate(reasoner) | delegate(verifier) | COMMIT
```

The main experiment is strictly **in-domain**. MedQA, LegalBench large5,
MMLU-Pro, and GPQA each train their own advisor LoRAs, cold-start manager, and
GRPO manager. No benchmark reuses a manager trained on another benchmark.

The canonical, copy-paste execution guide is
**[IN_DOMAIN_EXPERIMENTS.md](IN_DOMAIN_EXPERIMENTS.md)**. It specifies frozen
splits, exact question budgets, every command for all four benchmarks,
synthetic-data audits, GPU layout, baselines, ablations, and reporting rules.
`EXPERIMENTS.md` is an archived transfer-oriented plan and must not define new
main-paper runs.

## Current evaluation design

| Benchmark | In-domain training pool | Dev | Held-out pool | Final evaluation |
|---|---:|---:|---:|---:|
| MedQA | 3,000 | 300 | 500 | 300 |
| LegalBench large5 | about 2,134 | 200 balanced | 500 balanced | 300 balanced, 60/task |
| MMLU-Pro custom partition | 4,000 | 300 | 500 | 300 |
| GPQA Extended minus all Diamond | 318 + 30 dev | 30 | all Diamond 198 | all 198 |

MMLU-Pro results must be labeled as a custom in-domain split, not standard
leaderboard performance. GPQA-Diamond is never used for advisor synthesis,
cold-start, GRPO, prompt selection, or reward tuning.

## Architecture and invariants

```text
                         Manager (Qwen3-8B)
                   draft -> delegate or commit
                      /          |          \
             Extractor       Reasoner       Verifier
            factual signal   neutral plan   audits current draft
```

- Advisors emit structured signals, never final answers.
- Synthetic data is schema-checked, choice-leakage audited, deduplicated, and
  selected on one shared question set across all three advisor roles.
- Verifier candidates come from actual GT-blind base-manager predictions.
- Cold-start defaults to `base_stepwise`: the base manager is queried before
  tool use and after every advisor result, including the final result.
- Ground truth may supervise the terminal answer and select a cost-aware
  oracle action, but is never copied into a normal `DRAFT_ANSWER`.
- Advisor SFT, cold-start, GRPO, dev, and held-out question hashes are disjoint.
- GPQA training is Extended minus every Diamond question, regardless of how
  many Diamond questions are reported.

## Reward

The main manager uses the bounded anytime ADC objective:

```text
R = final_bonus * 1[final answer correct]
  + draft_bonus * mean(correctness of answer statements)
  - missing_draft_penalty * missing draft count
  - cost_per_tool * number of delegations
```

`transition` and `sum` remain ablation-only reward variants because they can
encourage sandbagging or unnecessary calls.

## Installation and GPU layout

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt accelerate deepspeed requests

conda create -n vllm_env python=3.11 -y
conda activate vllm_env
pip install vllm
```

For Qwen3-8B:

```text
GPU 0    vLLM base model + extractor/reasoner/verifier LoRAs
GPU 1-3  manager GRPO with Accelerate/DeepSpeed
GPU 1    final manager evaluation, calling advisors remotely on GPU 0
```

Start the shared base/advisor server after training a benchmark's three
adapters:

```bash
conda activate vllm_env
bash scripts/start_subagent_server.sh Qwen/Qwen3-8B <RUN_ID> outputs 8000 8192
curl -f http://localhost:8000/health
```

## Lightweight validation

These checks do not download models or datasets:

```bash
python -m py_compile \
  src/manager/evolve.py src/pipeline/cli.py src/pipeline/stages.py \
  src/subagents/synthesize.py scripts/*.py tests/*.py
python -m unittest discover -s tests -v
bash -n scripts/start_subagent_server.sh scripts/train_manager_grpo_multigpu.sh
```

Full GPU integration and benchmark runs require the model weights, accepted
dataset licenses, and the cluster environments described in
`IN_DOMAIN_EXPERIMENTS.md`.
