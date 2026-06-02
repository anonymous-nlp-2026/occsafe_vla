# Does Auxiliary Spatial Supervision Make Vision-Language-Action Models Safer? An Empirical Investigation

CoRL 2026 Submission

## Overview

We conduct a systematic 2x2 factorial study of auxiliary spatial supervision during Vision-Language-Action (VLA) fine-tuning, crossing **gradient flow** (ON/OFF) with **normalization type** (BN/LN). Our key findings:

- Detach+LN consistently outperforms the no-auxiliary baseline (CAR: 0.745 → 0.802)
- Two additive factors govern safety: gradient flow through BatchNorm degrades safety; LayerNorm provides a consistent protective effect
- Spatial label content has no detectable effect — the gradient pathway determines the outcome
- These effects are invisible to task-success benchmarks (TSR ANOVA all p>0.27)

## Repository Structure

```
.
├── vla-scripts/
│   ├── finetune.py                # Main training script (LoRA fine-tuning with OCC head)
│   └── evaluate_safelibero.py     # SafeLIBERO evaluation (TSR + CAR metrics)
├── prismatic/
│   ├── models/
│   │   ├── occ_head.py            # BEV Occupancy prediction head (core contribution)
│   │   ├── action_heads.py        # L1 regression and diffusion action heads
│   │   ├── projectors.py          # Proprio and noisy action projectors
│   │   ├── film_vit_wrapper.py    # FiLM language-conditioned ViT
│   │   ├── load.py                # Model loading utilities
│   │   └── vlas/openvla.py        # OpenVLA model definition
│   ├── training/
│   │   └── train_utils.py         # Training loss utilities
│   ├── vla/
│   │   ├── constants.py           # Platform-specific constants
│   │   ├── action_tokenizer.py    # Action discretization
│   │   └── datasets/
│   │       └── datasets.py        # RLDS dataset with OCC label loading
│   └── util/
│       └── data_utils.py          # Collators with OCC/depth label support
├── occ_pipeline/
│   ├── extract_occupancy.py       # BEV occupancy extraction from LIBERO demos
│   ├── occupancy_3d.py            # 3D voxel occupancy computation
│   ├── bev_projection.py          # 3D-to-BEV projection
│   ├── geom_classifier.py         # MuJoCo geom semantic classification
│   ├── extract_depth.py           # Monocular depth extraction
│   └── visualize_bev.py           # BEV visualization utilities
├── analysis/
│   ├── bootstrap_ci.py            # Wilson score CI computation
│   ├── per_task_comparison.py     # Per-task TSR/CAR comparison plots
│   ├── ablation_diagnostic.py     # Ablation diagnostic matrix
│   ├── seed_instability.py        # Cross-seed variance analysis
│   └── eval_utils.py              # Shared evaluation utilities
├── build_occ_index.py             # Build task-to-demo index for OCC labels
└── pyproject.toml                 # Project dependencies
```

## Environment Setup

### Prerequisites

- Python >= 3.8
- PyTorch 2.2.0 (CUDA 12.1)
- NVIDIA GPU with >= 48GB VRAM (training) or >= 24GB (evaluation)

### Installation

```bash
# Clone repository
git clone https://anonymous.4open.science/r/occsafe_vla-F25F/
cd occsafe_vla

# Install package and dependencies
pip install -e .

# Install flash-attention (optional, for faster training)
pip install flash-attn --no-build-isolation
```

### Key Dependencies

| Package | Version | Note |
|---------|---------|------|
| torch | 2.2.0 | CUDA 12.1 |
| transformers | custom fork | Bidirectional attention support |
| peft | 0.11.1 | LoRA fine-tuning |
| timm | 0.9.10 | Vision backbone |
| mujoco | latest | Physics simulation |
| libero | latest | Task benchmark |

### External Resources (Not Included)

- **Pre-trained model**: [OpenVLA-7B](https://huggingface.co/openvla/openvla-7b)
- **Dataset**: LIBERO-Spatial demonstrations (converted to RLDS format)
- **Simulation**: SafeLIBERO benchmark (LIBERO-Spatial with obstacle configurations)

## Reproducing Experiments

### Step 1: Generate BEV Occupancy Labels

Extract BEV occupancy labels from LIBERO demonstrations:

```bash
cd occ_pipeline/
MUJOCO_GL=egl python extract_occupancy.py \
    --task_suite_name libero_spatial \
    --all \
    --output_dir /path/to/occ_data \
    --data_dir /path/to/libero_spatial_demos
```

### Step 2: Training

Fine-tune OpenVLA-7B with LoRA and the auxiliary BEV occupancy head:

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
    --vla_path openvla/openvla-7b \
    --data_root_dir /path/to/libero_rlds \
    --dataset_name libero_spatial_no_noops \
    --use_l1_regression True \
    --num_images_in_input 2 \
    --use_proprio True \
    --batch_size 8 \
    --learning_rate 5e-4 \
    --max_steps 50005 \
    --save_freq 10000 \
    --image_aug True \
    --lora_rank 32 \
    --use_occ_head True \
    --occ_data_dir /path/to/occ_data \
    --occ_detach false \
    --occ_norm_type bn
```

#### Factorial Conditions

| Condition | Key Flags |
|-----------|-----------|
| Vanilla (no aux head) | `--use_occ_head False` |
| Full OCC (BN) | `--use_occ_head True --occ_detach false --occ_norm_type bn` |
| Detach OCC (BN) | `--use_occ_head True --occ_detach true --occ_norm_type bn` |
| Full OCC (LN) | `--use_occ_head True --occ_detach false --occ_norm_type ln` |
| Detach+LN | `--use_occ_head True --occ_detach true --occ_norm_type ln` |
| Random OCC | `--use_occ_head True --random_occ True` |

#### Training Flag Reference

| Flag | Values | Description |
|------|--------|-------------|
| `--use_occ_head` | True/False | Enable/disable BEV occupancy auxiliary head |
| `--occ_detach` | true/false | Detach gradients from OCC head (gradient flow OFF) |
| `--occ_norm_type` | bn/ln | Normalization: bn=BatchNorm2d, ln=LayerNorm via GroupNorm(1) |
| `--occ_loss_weight` | float | Initial OCC loss weight (default: 0.01, decays exponentially) |
| `--occ_loss_decay_steps` | int | Exponential decay half-life for OCC loss (default: 50000) |
| `--occ_loss_cutoff_step` | int | Disable OCC loss after step N (temporal ablation) |
| `--random_occ` | True/False | Randomly permute OCC labels (content irrelevance control) |

### Step 3: Evaluation

Run SafeLIBERO evaluation to measure TSR and CAR:

```bash
MUJOCO_GL=egl python vla-scripts/evaluate_safelibero.py \
    --pretrained_checkpoint /path/to/checkpoint \
    --task_suite_name libero_spatial \
    --num_rollouts 50 \
    --num_images_in_input 2 \
    --use_proprio True \
    --use_l1_regression True \
    --lora_rank 32 \
    --collision_threshold 0.001
```

### Step 4: Statistical Analysis

```bash
# Per-task comparison with bootstrap CI
python analysis/per_task_comparison.py \
    --conditions "Vanilla:eval_vanilla_s0.json,eval_vanilla_s1.json" \
                 "OCC:eval_occ_s0.json,eval_occ_s2.json"

# Ablation diagnostic (gradient interference vs. signal utility)
python analysis/ablation_diagnostic.py \
    --vanilla "Vanilla:eval_vanilla_*.json" \
    --full_occ "OCC:eval_occ_*.json" \
    --detach "Detach:eval_detach_*.json" \
    --random "Random:eval_random_*.json"

# Cross-seed variance analysis
python analysis/seed_instability.py \
    --results_dir /path/to/eval_results
```

## Metrics

- **TSR (Task Success Rate)**: Fraction of episodes where the manipulation task was completed successfully.
- **CAR (Collision Avoidance Rate)**: Fraction of episodes where no non-target object's L1 displacement from its post-warmup position exceeds the threshold (0.001), following SafeLIBERO Level I protocol.

## Hardware

All experiments were conducted on a single NVIDIA RTX PRO 6000 (96 GB VRAM). Training takes approximately 12.9 hours per condition (50K steps, batch size 8).

## Acknowledgments

This project builds upon [OpenVLA-OFT](https://github.com/moojink/openvla-oft) (MIT License).

## License

MIT License
