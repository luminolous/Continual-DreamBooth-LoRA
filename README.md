# Continual DreamBooth-CLoRA

> Sequential Anime Character Personalization with Continual Learning

A research-grade repository for continual learning in text-to-image diffusion personalization. Trains and evaluates a continual-learning pipeline for anime character personalization across sequential tasks.

## Methods

| Method | Description | Forgetting Defense |
|--------|-------------|-------------------|
| `naive_sequential` | Sequential fine-tuning baseline. No regularization. | None (measures catastrophic forgetting) |
| `c_lora_scaffold` | Legacy scaffold with importance-weighted L2 regularization toward previous checkpoint. | L2 penalty (magnitude or Fisher-weighted) |
| **`faithful_c_lora`** | **Shared continual model** with per-task LoRA adapters, randomly initialized tokens, occupancy regularization. | Occupancy constraint on past LoRA factor matrices |

### Faithful C-LoRA — How It Works

**Shared continual model**: After training N tasks, the model is:
```
Base UNet + adapter_0 + adapter_1 + ... + adapter_N (all active, deltas summed)
Text encoder with tokens <lumy01>, <lumy02>, ..., <lumyN> (each with learned embeddings)
```

**Task identity** comes from personalized token embeddings, not adapter selection. Prompting with `"a photo of <lumy01>"` activates concept 1 through the composed model.

**Forgetting** is real — new adapter deltas can interfere with old adapter deltas in the composed model (e.g., cancelling their contributions). The **occupancy regularizer** prevents this by penalizing new deltas in weight positions already used by past adapters:

```
L_occ = λ * Σ_m ||O^m ⊙ (B_new @ A_new)||²

where O^m = normalize(Σ_k |B_k @ A_k|)  (accumulated past adapter usage)
```

## Quick Start

```bash
# 1. Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

# 2. Install dependencies
pip install -r requirements.txt

# 3. Validate setup
python scripts/validate_setup.py --config configs/tasks_5char_faithful.yaml

# 4. Run experiment
python -m src.main --config configs/tasks_5char_faithful.yaml        # faithful C-LoRA
python -m src.main --config configs/tasks_5char.yaml                 # naive baseline
python -m src.main --config configs/tasks_5char.yaml --dry-run       # validate only
python -m src.main --config configs/tasks_5char.yaml --eval-only <CHECKPOINT_DIR>
```

## Dataset Structure

```
dataset/
├── char_01/
│   ├── train/     # 5-10 training images (.png, .jpg, .webp)
│   └── ref/       # 3-5 reference images for CCIP evaluation
├── char_02/
│   ├── train/
│   └── ref/
└── ...
```

## Configuration

### Faithful C-LoRA Config (`configs/tasks_5char_faithful.yaml`)

```yaml
experiment:
  method: "faithful_c_lora"

c_lora:
  token_init: "random"              # randomly initialized tokens
  train_token_embeddings: true      # train token embedding rows
  prompt_mode: "clora"              # "a photo of <lumy01>" — no class word
  adapter_strategy: "per_task"      # per-task adapters composed at inference
  regularizer_type: "occupancy"     # occupancy-based interference constraint
  regularization_weight: 0.1
  prior_preservation: true          # prior preservation (default ON)
  prior_loss_weight: 1.0
  num_class_images: 200
  class_prompt: "a photo of anime character"
  run_multi_concept_probe: true     # multi-concept generation probe
```

### Key Config Options

| Option | Values | Description |
|--------|--------|-------------|
| `method` | `naive_sequential`, `c_lora_scaffold`, `faithful_c_lora` | Continual learning method |
| `token_init` | `random`, `fixed` | Token embedding initialization |
| `prompt_mode` | `clora`, `dreambooth` | `clora`=no class word, `dreambooth`=full prompt |
| `regularizer_type` | `occupancy`, `l2`, `none` | Regularization type |
| `prior_preservation` | `true`, `false` | Generate class-prior images for regularization |
| `inference_adapter_mode` | `compose_all`, `per_task` | Primary eval uses composed model |
| `run_diagnostic_eval` | `true`, `false` | Optional isolated-adapter diagnostic |

## Output Structure

```
outputs/<experiment_name>/
├── checkpoints/
│   ├── task_00_char_01/
│   │   ├── lora_weights/                # PEFT adapter
│   │   ├── token_embeddings.pt          # learned token vectors
│   │   └── task_info.json               # per-task metadata
│   └── ...
├── eval_images/
│   ├── stage_00/char_01/                # PRIMARY eval images
│   ├── stage_01/char_01/
│   ├── stage_01/char_02/
│   ├── stage_01/multi_concept/          # multi-concept probe
│   └── diagnostic/ (optional)           # isolated-adapter eval
├── class_prior/                         # generated class images
├── task_registry.json                   # global experiment metadata
├── score_matrix.csv                     # PRIMARY score matrix
├── heatmap.png
├── score_progression.png
├── forgetting_bar.png
├── metrics.json
└── summary.txt
```

## Evaluation

### Primary Evaluation (Composed Model)

After each task, the **composed model** (all adapters active) is evaluated on all seen tasks. The score matrix `A[t][j]` measures whether concept `j` is retained when prompted through the shared continual model.

### Diagnostic Evaluation (Optional)

Loads each task's adapter in isolation. Used for debugging and ablation — not the primary continual-learning metric. Enable with `run_diagnostic_eval: true`.

### Multi-Concept Probe

Generates images with multiple task tokens (e.g., `"a photo of <lumy01> and <lumy02>"`). Visual output only — no automated metric in MVP. Enable with `run_multi_concept_probe: true`.

### Metrics

| Metric | Description |
|--------|-------------|
| **Score matrix** `A[t][j]` | CCIP score for task `j` after training task `t` |
| **Average accuracy** | Mean of diagonal `A[t][t]` |
| **Average forgetting** | `A[j][j] - A[T-1][j]` for old tasks |
| **Backward transfer** | `A[T-1][j] - A[j][j]` |
| **Confusion gap** | `target_CCIP - max(non_target_CCIP)` (optional) |

CCIP scoring uses `sdeval==0.2.4` (pinned for reproducibility).

## Faithfulness Assessment

### Exact (faithful to C-LoRA core)

- Per-task LoRA adapters for cross-attention K/V (`attn2.to_k`, `attn2.to_v`)
- Randomly initialized personalized token embeddings
- Training personalized token embeddings only
- Prompts without explicit class word
- Shared continual model (composed adapter deltas)
- Past LoRA factors inform regularization
- No replay of user training data

### Approximate

- **Occupancy mask**: uses `|B@A|` magnitude, not gradient or subspace analysis
- **Prior preservation**: standard DreamBooth technique, not C-LoRA-specific
- **Single token per concept**: paper may support multi-token

### Known Limitations

- Multi-concept generation evaluation is visual only (no automated metric)
- No Fisher-diagonal importance for occupancy masks (deferred)
- No subspace projection regularization (deferred)
- No training resume from mid-task checkpoint

## Repository Structure

```
├── configs/
│   ├── tasks_5char.yaml                # naive/scaffold config
│   └── tasks_5char_faithful.yaml       # faithful C-LoRA config
├── src/
│   ├── config/schema.py                # dataclass config + YAML loader
│   ├── data/dataset.py                 # datasets + prompt building
│   ├── training/trainer.py             # training loop + adapter + token management
│   ├── methods/
│   │   ├── base.py                     # abstract method interface
│   │   ├── naive_sequential.py         # no-op baseline
│   │   ├── c_lora_scaffold.py          # legacy L2 scaffold
│   │   └── faithful_c_lora.py          # occupancy regularization (core)
│   ├── orchestrator/pipeline.py        # sequential task loop
│   ├── eval/
│   │   ├── generator.py                # image generation
│   │   ├── metrics.py                  # CCIP scoring
│   │   └── report.py                   # plots + summary
│   ├── utils/
│   │   ├── io.py                       # checkpoints + token embeddings + registry
│   │   ├── seed.py                     # reproducibility
│   │   └── logging.py                  # structured logging
│   └── main.py                         # CLI entry point
├── scripts/validate_setup.py
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Dependencies

Key pinned versions:
- `sdeval==0.2.4` — CCIP evaluation
- `diffusers>=0.25.0,<0.31.0`
- `peft>=0.7.0`
- `torch>=2.0.0`
- `transformers>=4.30.0`

## License

Research use. See individual library licenses for dependencies.
