# Social Token Fine-tuning for LLMs

[![NeurIPS 2025 Workshop](https://img.shields.io/badge/NeurIPS-2025%20Workshop-blue)](https://neurips.cc/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-green.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

Official implementation of **"Look, Then Speak: Social Tokens for Grounding LLMs in Visual Interactions"** (NeurIPS 2025 Workshop).

## Overview

This codebase introduces **social tokens** - a lightweight mechanism that injects socially grounded visual information into frozen large language models (LLMs) without updating their parameters. By fine-tuning visual encoders (DINOv2) to capture socially relevant visual cues and projecting them into the LLM's embedding space, we enable language models to condition generation on social visual context.

### Key Innovation

- **Frozen LLM**: The language model parameters remain frozen throughout training
- **Visual Social Grounding**: Fine-tuned DINOv2 encoders capture social cues from video frames
- **Token-level Injection**: Social information is injected via:
  - `[SOC-G]` Global social tokens (aggregated visual representation)
  - `[SOC-L]` Local social tokens (frame-level representations inserted after nouns/verbs)
- **Substantial Improvements**:
  - Next-word prediction (Gemma-2-2b): 164.28 PPL (frozen) → 36.94 PPL (with social tokens)
  - **4.4x improvement** over frozen baseline on social dialogue tasks

## Recent Improvements (2025-01)

**Fair Perplexity Evaluation:** Removed `[CAP]:` pattern tokens that artificially reduced perplexity by allowing the model to memorize trivial formatting patterns (67-80% of predictions). The current results reflect actual content prediction ability.

**Generation Quality Fixes:**
1. **EOS Token Masking**: Prevents model from immediately predicting end-of-sequence
2. **Repetition Penalty**: Eliminates generation loops (e.g., "1 1 1 1...")
3. **Appropriate Length**: Reduced from 50 to 10 tokens for next-word prediction

**Ablation Study Framework:** Comprehensive evaluation across 4 modes (both tokens, global only, local only, frozen baseline) to measure individual token contributions.

See [ABLATION_STUDY_GUIDE.md](ABLATION_STUDY_GUIDE.md) for complete details.

## Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Pipeline Overview](#pipeline-overview)
- [Usage](#usage)
- [Data Format](#data-format)
- [Evaluation](#evaluation)
- [Citation](#citation)
- [License](#license)

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (training requires significant GPU memory)
- Conda or virtualenv

### Setup

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/social-token-finetuning.git
cd social-token-finetuning
```

2. **Create conda environment**

```bash
conda create -n soc_env python=3.8
conda activate soc_env
```

3. **Install dependencies**

```bash
# Core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers timm accelerate
pip install pandas numpy scipy scikit-learn
pip install nvidia-dali-cuda110  # Adjust CUDA version as needed

# Optional: for visualization
pip install matplotlib seaborn
```

4. **Set environment variables**

```bash
export HF_TOKEN="your_huggingface_token"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## Project Structure

```
social-token-finetuning/
├── scripts/
│   ├── preprocessing/          # Video preprocessing and feature extraction
│   │   ├── build_scene_packs.py
│   │   ├── build_scene_packs_gpu.py
│   │   └── generate_frame_manifest.py
│   ├── video_encoder/          # DINOv2 visual encoder training
│   │   └── train_dino_dali_ddp_improved.py
│   ├── llm_finetuning/        # Social token injection and LLM fine-tuning
│   │   ├── next_utt_social_ooo_fix.py      # Main training script
│   │   ├── ooo_rsa_gemma_new_.py           # RSA variant
│   │   └── train_text_only_baseline.py     # Text-only baseline
│   └── analysis/              # Evaluation and analysis tools
│       ├── plot_perplexity_comparison.py   # Ablation study plots
│       ├── logit_lens_from_jsonl_weights_ooo.py
│       ├── sim_judg_rsa_gemma.py
│       └── compare_soc_embeddings.py
├── slurm/                     # SLURM job scripts for HPC clusters
│   ├── run_dino_ddp.slurm
│   ├── run_gemma2_full_ablation.slurm     # Full ablation study
│   ├── run_next_utt_social_ooo.slurm
│   └── run_ooo_rsa_gemma_unl.slurm
├── ABLATION_STUDY_GUIDE.md    # Detailed ablation study instructions
├── RSA_ANALYSIS_GUIDE.md      # RSA evaluation guide
├── data/                      # Data directory (not included in repo)
    ├── videos/               # Input videos
    ├── frames/               # Extracted frames
    ├── transcripts/          # Conversation transcripts
    └── results/              # Training outputs and metrics
        ├── perplexity/       # Ablation study plots
        └── rsa/              # RSA evaluation results
```

## Quick Start

### 1. Preprocess Video Data

Extract frames and compute visual features from video files:

```bash
python scripts/preprocessing/build_scene_packs.py \
    --input-dir /path/to/videos \
    --output-dir /path/to/output \
    --fps 1 \
    --extract-features
```

### 2. Train Visual Encoder (DINOv2)

Fine-tune DINOv2 on social interaction frames:

```bash
# Single GPU
python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=1 \
    scripts/video_encoder/train_dino_dali_ddp_improved.py \
    --manifest ~/data_lisik3/kgarci18/seamless/full/preprocess/naturalistic/train/manifest.csv \
    --out_dir ~/data_lisik3/kgarci18/seamless/outputs/dino_checkpoints \
    --epochs 100 \
    --patience 10 \
    --model_name vit_base_patch14_dinov2 \
    --global_size 518 \
    --local_size 518 \
    --global_area 0.40 1.00 \
    --local_area 0.05 0.25 \
    --global_crops 2 \
    --local_crops 8 \
    --accum_steps 2 \
    --num_workers 8 \
    --amp

# Multi-GPU with DDP (4 GPUs)
python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=4 \
    scripts/video_encoder/train_dino_dali_ddp_improved.py \
    --manifest ~/data_lisik3/kgarci18/seamless/full/preprocess/naturalistic/train/manifest.csv \
    --out_dir ~/data_lisik3/kgarci18/seamless/outputs/dino_checkpoints \
    --epochs 100 \
    --patience 10 \
    --model_name vit_base_patch14_dinov2 \
    --global_size 518 \
    --local_size 518 \
    --global_area 0.40 1.00 \
    --local_area 0.05 0.25 \
    --global_crops 2 \
    --local_crops 8 \
    --accum_steps 2 \
    --num_workers 8 \
    --teacher_t_warmup 0.04 \
    --teacher_t_end 0.07 \
    --teacher_warmup_epochs 30 \
    --teacher_momentum_start 0.996 \
    --teacher_momentum_end 0.9995 \
    --amp

# Multi-GPU with SLURM
sbatch slurm/run_dino_ddp.slurm
```

### 3. Fine-tune LLM with Social Tokens

Train the social token projection layer while keeping LLM frozen:

```bash
python scripts/llm_finetuning/next_utt_social_ooo_fix.py \
    --parent-dir /path/to/data \
    --output-dir outputs/gemma_social \
    --lm-name google/gemma-2-2b \
    --dino-name vit_base_patch14_dinov2 \
    --dino-checkpoint outputs/dino_checkpoints/checkpoint_best.pt \
    --dino-checkpoint-key student_backbone \
    --dino-tune-mode cls_adapter \
    --caption-nextword \
    --train-ablation-mode both \
    --eval-ablations both global_only local_only frozen_baseline \
    --batch-size 4 \
    --epochs 5 \
    --visual-mode vectors
```

Or submit to SLURM cluster:

```bash
sbatch slurm/run_gemma2_full_ablation.slurm
```

### 4. Evaluate with RSA

Run Representational Similarity Analysis to measure alignment with human judgments:

```bash
python scripts/analysis/sim_judg_rsa_gemma.py \
    --lm google/gemma-2-2b \
    --tokenizer-dir outputs/gemma_social \
    --projector-ckpt outputs/gemma_social/checkpoints/projector_only.pt \
    --index data/rsa/ooo_index.csv \
    --sim-rsm data/rsa/human_judgments.csv \
    --inject global \
    --pool all
```

See [RSA_ANALYSIS_GUIDE.md](RSA_ANALYSIS_GUIDE.md) for detailed evaluation instructions.

## Pipeline Overview

The training pipeline consists of three main stages:

### Stage 1: Video Preprocessing

- **Input**: Raw video files (MP4, AVI, etc.)
- **Process**: Extract frames at specified FPS, compute DINO features
- **Output**: Frame directories, feature vectors (.npy), manifests (.csv)

**Key scripts**: `build_scene_packs.py`, `generate_frame_manifest.py`

### Stage 2: Visual Encoder Training

- **Input**: Frame manifests with temporal metadata
- **Process**: Fine-tune DINOv2 using self-supervised DINO loss with teacher-student architecture
- **Output**: Fine-tuned DINO checkpoint (.pt)

**Key scripts**: `train_dino_dali_ddp_improved.py`

**Architecture**:
- Teacher-student ViT models (vit_base_patch14_dinov2)
- Momentum-based teacher updates
- Multi-crop data augmentation
- DistributedDataParallel (DDP) for multi-GPU training

### Stage 3: Language Model Fine-tuning

- **Input**: Conversation transcripts, frame features, DINO checkpoint
- **Process**:
  1. POS tag transcripts to identify nouns/verbs
  2. Align frames to words temporally
  3. Project visual features into LLM embedding space
  4. Train MLP projector via next-token prediction (LLM frozen)
- **Output**: Projector checkpoint, social token embeddings

**Key scripts**: `next_utt_social_ooo_fix.py`, `ooo_rsa_gemma_new_.py`

## Usage

### Training on HPC Clusters

This project is designed for SLURM-based HPC environments:

```bash
# Visual encoder training (multi-GPU)
sbatch slurm/run_dino_ddp.slurm

# Language model fine-tuning
sbatch slurm/run_next_utt_social_ooo.slurm

# Large-scale Gemma training
sbatch slurm/run_ooo_rsa_gemma_unl.slurm
```

### Local Development

For testing on a single GPU:

```bash
# Activate environment
conda activate seamless_env

# Run with reduced batch size
python scripts/llm_finetuning/next_utt_social_ooo_fix.py \
    --parent-dir ./data \
    --output-dir ./outputs \
    --batch-size 1 \
    --gradient-accumulation-steps 8
```

### Configuration Options

**Ablation Study Modes** (`--train-ablation-mode`):
- `both`: Train with both global `[SOC-G]` and local `[SOC-L]` tokens (default, best performance)
- `global_only`: Train with only global `[SOC-G]` token
- `local_only`: Train with only local `[SOC-L]` tokens
- `none`: No social tokens (text-only baseline)

**Evaluation Ablations** (`--eval-ablations`):
- `both`: Evaluate with both global and local tokens
- `global_only`: Mask out local tokens at test time
- `local_only`: Mask out global token at test time
- `frozen_baseline`: Evaluate with untrained model (no visual information)
- Can specify multiple: `--eval-ablations both global_only local_only frozen_baseline`

**Visual Modes** (`--visual-mode`):
- `vectors`: Pre-computed visual feature vectors (recommended, faster)
- `frames`: Load frames on-the-fly (slower, more memory intensive)

**Task Mode** (`--caption-nextword`):
- Enable for next-word prediction task on captions
- Omit for full next-utterance prediction on dialogues

## Data Format

### Video Data

```
data/videos/
├── clip_001.mp4
├── clip_002.mp4
└── ...
```

### Extracted Frames

```
data/frames/
├── clip_001/
│   ├── frame_0000.jpg
│   ├── frame_0001.jpg
│   └── ...
└── clip_002/
    └── ...
```

### Transcripts

JSON format with conversation structure:

```json
{
  "conversation_id": "conv_001",
  "turns": [
    {
      "speaker": "A",
      "utterance": "Hello, how are you?",
      "timestamp": 1.5,
      "frame_path": "data/frames/clip_001/frame_0001.jpg"
    }
  ]
}
```

### Manifests

CSV format with columns: `split,clip_id,frame_path,t_sec`

```csv
split,clip_id,frame_path,t_sec
train,clip_001,data/frames/clip_001/frame_0001.jpg,1.5
train,clip_001,data/frames/clip_001/frame_0002.jpg,2.5
val,clip_002,data/frames/clip_002/frame_0001.jpg,0.8
```

## Evaluation

### Ablation Study

Run complete ablation study to evaluate all token configurations:

```bash
# Submit full ablation training job
sbatch slurm/run_gemma2_full_ablation.slurm

# After training completes, generate comparison plot
python scripts/analysis/plot_perplexity_comparison.py \
    --log-file logs/gemma2_full_abl_JOBID.out \
    --output-dir data/results/perplexity
```

This evaluates 4 modes:
- **Both tokens**: Full model with global + local social tokens
- **Global only**: Mask out local tokens at test time
- **Local only**: Mask out global token at test time
- **Frozen baseline**: Untrained model (no visual information)

See [ABLATION_STUDY_GUIDE.md](ABLATION_STUDY_GUIDE.md) for detailed instructions.

### Perplexity Measurement

Evaluate model perplexity on dialogue datasets:

```bash
python scripts/llm_finetuning/next_utt_social_ooo_fix.py \
    --parent-dir /path/to/data \
    --output-dir outputs/eval \
    --caption-nextword \
    --train-ablation-mode both \
    --eval-ablations both frozen_baseline \
    --checkpoint outputs/gemma_social/checkpoint_final.pt
```

### Representational Similarity Analysis (RSA)

Measure alignment with human similarity judgments on social videos:

```bash
# Run full ablation study
bash scripts/analysis/run_rsa_ablations.sh

# Compare results
python scripts/analysis/compare_rsa_results.py
```

See [RSA_ANALYSIS_GUIDE.md](RSA_ANALYSIS_GUIDE.md) for comprehensive evaluation instructions.

### Attention Analysis

Visualize where the model attends to social tokens:

```bash
python scripts/analysis/visualize_token_attention.py \
    --checkpoint outputs/gemma_social/checkpoint_final.pt \
    --input-text "She smiled warmly." \
    --frame-path data/frames/example/frame_001.jpg
```

## Results

### Perplexity on Next-Word Prediction (Gemma-2-2b)

| Ablation Mode | Test Perplexity | Description |
|---------------|-----------------|-------------|
| Both tokens (global + local) | **36.94** | Best - uses all visual information |
| Global only token | ~50-80 | No local tokens, less detailed visual info |
| Local only tokens | ~50-80 | No global token, less contextual visual info |
| Frozen baseline (no training) | **164.28** | Worst - no training, no visual info |

**Performance gain:** 4.4x improvement over frozen baseline (164.28 → 36.94 PPL)

**Note:** Previous results reporting extremely low perplexity (e.g., 5.25 PPL) included `[CAP]:` pattern tokens that allowed the model to memorize trivial formatting patterns. These have been removed for fair evaluation. See [ABLATION_STUDY_GUIDE.md](ABLATION_STUDY_GUIDE.md) for details on fixes applied.

### RSA Alignment (Spearman r)

| Model Configuration | Spearman r | p-value |
|---------------------|-----------|---------|
| Gemma-2-2B (baseline) | 0.323 | < 0.001 |
| + Global Social Token | 0.335 | < 0.001 |
| + Global + Local Tokens | 0.280 | < 0.001 |

Global tokens provide the best performance (+3.7% over baseline).

## Citation

If you use this codebase in your research, please cite:

```bibtex
@inproceedings{garcia2025social,
  title={Look, Then Speak: Social Tokens for Grounding LLMs in Visual Interactions},
  author={Garcia, Kevin and others},
  booktitle={NeurIPS 2025 Workshop on Unified Representations},
  year={2025}
}
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Troubleshooting

### Out of Memory Errors

```bash
# Reduce batch size and use gradient accumulation
--batch-size 1 --gradient-accumulation-steps 16

# Use pre-computed visual vectors instead of loading frames
--visual-mode vectors
```

### CUDA Errors

```bash
# Check GPU availability
nvidia-smi

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Enable expandable memory segments
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### Empty or Repetitive Generations

**Fixed in current version** (lines 927-931, 1499-1509 in `next_utt_social_ooo_fix.py`):
- ✅ EOS token masked from training labels
- ✅ Repetition penalty (1.2) applied during generation
- ✅ Max tokens reduced from 50 to 10 for next-word prediction

If still experiencing issues, ensure you're using the latest version of the script.

### Artificially Low Perplexity

**Fixed in current version** (lines 674-679):
- ✅ Removed `[CAP]:` pattern tokens that allowed trivial predictions
- Results now reflect actual content prediction ability

See [ABLATION_STUDY_GUIDE.md](ABLATION_STUDY_GUIDE.md) for details.

### Missing Dependencies

```bash
# Core dependencies
pip install torch torchvision transformers timm pandas numpy scipy scikit-learn matplotlib

# HuggingFace token required
export HF_TOKEN="your_token_here"
```

## Acknowledgments

- **DINOv2**: [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2)
- **Gemma**: [google/gemma](https://huggingface.co/google/gemma-2-2b)
- **Transformers**: [HuggingFace Transformers](https://github.com/huggingface/transformers)
- **NVIDIA DALI**: [NVIDIA Data Loading Library](https://github.com/NVIDIA/DALI)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or issues, please:
- Open an issue on GitHub
- Contact: [kgarci18@jhu.edu]

---

**Note**: This codebase is designed for HPC clusters with SLURM scheduling. For local development, adjust batch sizes and data paths accordingly.
