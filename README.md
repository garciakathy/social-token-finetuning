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
  - Seamless dataset: 90.37 → 5.25 perplexity (94.2% reduction)
  - Moments in Time: 2662.02 → 3.25 perplexity (99.9% reduction)

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
│   │   ├── next_utt_social_ooo_fix.py
│   │   └── ooo_rsa_gemma_new_.py
│   └── analysis/              # Evaluation and analysis tools
│       ├── logit_lens_from_jsonl_weights_ooo.py
│       ├── sim_judg_rsa_gemma.py
│       └── compare_soc_embeddings.py
├── slurm/                     # SLURM job scripts for HPC clusters
│   ├── run_dino_ddp.slurm
│   ├── run_next_utt_social_ooo.slurm
│   └── run_ooo_rsa_gemma_unl.slurm
├── data/                      # Data directory (not included in repo)
    ├── videos/               # Input videos
    ├── frames/               # Extracted frames
    ├── transcripts/          # Conversation transcripts
    └── results/              # Training outputs and metrics
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
python scripts/video_encoder/train_dino_dali_ddp_improved.py \
    --manifest data/manifests/train.csv \
    --out_dir outputs/dino_checkpoints \
    --batch-size 32 \
    --epochs 100

# Multi-GPU with SLURM
sbatch slurm/run_dino_ddp.slurm
```

### 3. Fine-tune LLM with Social Tokens

Train the social token projection layer while keeping LLM frozen:

```bash
python scripts/llm_finetuning/next_utt_social_ooo_fix.py \
    --parent-dir /path/to/data \
    --output-dir outputs/gemma_social \
    --lm google/gemma-2-2b \
    --dino-ckpt outputs/dino_checkpoints/checkpoint_best.pt \
    --inject full \
    --batch-size 4 \
    --epochs 3
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

**Social Token Injection Modes** (`--inject`):
- `none`: No social tokens (baseline)
- `global`: Only global `[SOC-G]` tokens
- `local`: Only local `[SOC-L]` tokens (requires `--pool locals_only`)
- `full`: Both global and local tokens

**Pooling Modes** (`--pool`):
- `all`: Pool all tokens including social tokens
- `exclude_soc`: Pool only text tokens
- `only_soc`: Pool only social tokens
- `locals_only`: Pool only local social tokens
- `eos`: Pool only the EOS token

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

### Perplexity Measurement

Evaluate model perplexity on dialogue datasets:

```bash
python scripts/llm_finetuning/next_utt_social_ooo_fix.py \
    --parent-dir /path/to/data \
    --output-dir outputs/eval \
    --eval-only \
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

### Perplexity Reduction

| Dataset | Baseline | + Social Tokens | Improvement |
|---------|----------|-----------------|-------------|
| Seamless | 90.37 | 5.25 | 94.2% |
| Moments in Time | 2662.02 | 3.25 | 99.9% |

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
# Reduce batch size
--batch-size 1 --gradient-accumulation-steps 16

# Enable memory-efficient attention
--use-flash-attention
```

### CUDA Errors

```bash
# Check GPU availability
nvidia-smi

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0
```

### Missing Dependencies

```bash
# Reinstall with specific versions
pip install -r requirements.txt  # If available
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
