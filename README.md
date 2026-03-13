# PRR: Progressive Refinement Regulation for Accelerating Diffusion LM Decoding

This repository contains the inference implementation for **Progressive Refinement Regulation (PRR)**, a framework that accelerates diffusion language model decoding by regulating the refinement process via a lightweight, token-wise controller.

## 📄 Paper
**Progressive Refinement Regulation for Accelerating Diffusion Language Model Decoding**

## 🚀 Features

- **PRR Controller**: A lightweight MLP Head that predicts token convergence progress.
- **Dynamic Regulation**: Adjusts the refinement trajectory using temperature-based distribution shaping.
- **Inference Acceleration**: Achieves significant speedup by identifying and committing stable tokens early.

## 📂 Project Structure

```
.
├── prr_inference.py       # Core PRR logic: Controller definition & Inference loop
├── prr_evaluate.py        # Main evaluation script
├── benchmark_gsm8k.py     # Benchmark runner for GSM8K
├── benchmark_humaneval.py # Benchmark runner for HumanEval
├── model/                 # LLaDA model definitions
└── requirements.txt       # Dependencies
```

## 🛠️ Usage

### 1. Installation

```bash
git clone https://github.com/huaxueye/PRR.git
cd PRR
pip install -r requirements.txt
```

### 2. Prepare Checkpoints

- **Base Model**: `GSAI-ML/LLaDA-1.5` (downloaded automatically or specify local path).
- **PRR Head Checkpoint**: You need the trained PRR Head checkpoint (e.g., `head_checkpoint.pt`).

### 3. Run Inference

To run a quick test with the PRR strategy:

```bash
python prr_inference.py
```
*Note: Ensure `head_checkpoint.pt` exists in the directory, or update the `HEAD_PATH` variable in the script.*

### 4. Run Benchmarks

**GSM8K:**
```bash
python benchmark_gsm8k.py --head_path /path/to/head.pt --gpus 0
```

**HumanEval:**
```bash
python benchmark_humaneval.py --head_path /path/to/head.pt --gpus 0
```

**Arguments:**
- `--head_path`: Path to the trained PRR controller checkpoint.
- `--steps`: Diffusion steps (default: 256 for GSM8K).
- `--alphas`: PRR control strength (alpha).
- `--thresholds`: Confidence threshold for acceleration.

## 📜 License

Apache License 2.0
## Citation

If you find PRR helpful in your research, please cite:

```bibtex
@misc{wan2026progressive,
  title={Progressive Refinement Regulation for Accelerating Diffusion Language Model Decoding},
  author={Wan, Lipeng and Gu, Jianhui and Ma, Junjie and Huang, Jianguo and Sun, Shiguang and Li, Siyuan and Lan, Xuguang},
  year={2026},
  eprint={2603.04514},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2603.04514}
}
```
