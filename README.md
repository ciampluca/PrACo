# PrACo: Prompt-Aware Counting Benchmark

This repository contains the code, evaluation scripts, and dataset links used for the PrACo benchmark family for prompt-aware, class-agnostic object counting. It now supports both the original PrACo benchmark (introduced in the WACV 2025 paper) and the extended multi-class benchmark suite PrACo++.

Papers:
- Does it Really Count? Assessing Semantic Grounding in Text-Guided Class-Agnostic Counting - New 2026 [arXiv](https://arxiv.org/pdf/2605.02752)
- Mind the Prompt: A Novel Benchmark for Prompt-based Class-Agnostic Counting — WACV 2025. [ArXiv](https://arxiv.org/abs/2409.15953)


<img width="1101" height="393" alt="{5652AFE7-7E1F-4DEA-9756-289009F71330}" src="https://github.com/user-attachments/assets/016b4ef3-09ed-4d49-a5ce-5ccab14225a4" />

## Table of Contents

1. [Overview](#overview)
2. [Repository Structure](#repository-structure)
3. [Installation](#installation)
   - [1. Create a Conda Environment](#1-create-a-conda-environment)
   - [2. Download the FSC-147 Dataset](#2-download-the-fsc-147-dataset)
   - [3. Download MUCCA / PrACo++ (multi-class)](#3-download-mucca--praco)
   - [4. Download Pre-Trained Weights and Model Files](#4-download-pre-trained-weights-and-model-files)
4. [Running the Benchmark](#running-the-benchmark)
5. [Multiclass tests, metrics and outputs](#multiclass-tests-metrics-and-outputs)
6. [Running statistics](#running-statistics)
7. [Available Models](#available-models)
8. [References](#references)

## Overview

PrACo evaluates prompt-aware, class-agnostic counting models and reveals failures where methods ignore the prompt or bias toward common classes. PrACo++ expands the benchmark to multi-class prompts and new metrics for measuring prompt adherence and semantic correctness in multi-class scenarios. Use this repository to reproduce the experiments in both the original and extension papers.

![qualitative_mosaics-1](https://github.com/user-attachments/assets/4e0eca81-8038-432f-845f-b5f92cc06035)

## Repository Structure

- `benchmark/`: Scripts for evaluating models on PrACo and PrACo++
- `models/`: Model adapters and integration code (see subfolders for examples)
- `main.py`: Single-class PrACo runner
- `multiclass_main.py`: Multi-class PrACo++ / MUCCA runner
- `main_statistics.py` / `multiclass_main_statistics.py`: Aggregation scripts for single/multi-class benchmarks
- `requirements.txt` and `environment.yml`: dependency manifests
- `benchmark_results/`, `multiclass_benchmark_results/`: folders for per-model output and aggregated CSVs

## Installation

#### 1. Create a Conda Environment (recommended)

We provide an `environment.yml` that can be used as the base environment for running experiments (including the multi-class MUCCA suite). Create and activate it with:

```bash
conda env create -f environment.yml
conda activate mucca_models
pip install -r requirements.txt
```

If you prefer creating a minimal environment manually, the original instructions (Python 3.10 + `pip install -r requirements.txt`) remain supported.

### 2. Download the FSC-147 Dataset (original PrACo)

The original PrACo benchmark uses the FSC-147 dataset. Download and place the data under the `data/` folder as follows:

- FSC-147 Dataset: https://drive.google.com/file/d/1ymDYrGs9DSRicfZbSCDiOu0ikGDh5k6S/view?usp=sharing
- Image descriptions (FSC-147-D): https://github.com/niki-amini-naieni/CounTX/blob/main/FSC-147-D.json

Unzip into `data/images_384_VarV2` (or point the scripts to your image folder):

```bash
unzip FSC147_384_V2.zip -d data/images_384_VarV2
```

### 3. Download the Multi-Category Class-Agnostic Counting (MUCCA) benchmark

The multi-class dataset MUCCA is available on Zenodo. Download the package and place it under `data/multiclass-dataset/` (or point your run commands to the folder):

- MUCCA download: [Zenodo](https://zenodo.org/records/19231375)

The MUCCA dataset contains:
- multi-class image splits and annotations
- class-level prompts and prompt templates used for evaluation
- per-image ground-truth counts for each class

Place the unpacked folder at `data/multiclass-dataset/` so that the multiclass scripts can find it by default.

#### Preparing density maps (important)

The MUCCA dataset contains images and annotations (points + class IDs) but does not include precomputed density maps. If you want to run localized metrics or methods that require density maps, generate them from the annotations using the provided script.

The dataset already contains a helper script and documentation at `data/multiclass-dataset/generate_density_maps.py` and `data/multiclass-dataset/README_density_maps.md`.

Basic command (generate `.npy` density maps for all classes and images):

```bash
python data/multiclass-dataset/generate_density_maps.py --data_dir data/multiclass-dataset --output_dir data/multiclass-dataset/density_maps
```

### 4. Download Pre-Trained Weights and Model Files

Pre-trained weights for the models evaluated in the papers should be placed in the appropriate model folders under `models/` or in a `pretrained_models/` directory at the repository root. Example links (as used in our experiments):

- CounTX: https://drive.google.com/file/d/1Vg5Mavkeg4Def8En3NhceiXa-p2Vb9MG/view?usp=sharing
- CLIP-Count: https://drive.google.com/file/d/17Dj0tjd29lPGOGYEF5IrE8aPClXUjTrR/view?usp=drive_link
- VLCounter: https://drive.google.com/file/d/1-2lqtsOm9XW4MXhLzrB5Jf9RkXOpDlaQ/view?usp=sharing
- DAVE: https://drive.google.com/drive/folders/10O4SB3Y380hcKPIK8Dt8biniVbdQ4dH4?usp=drive_link
- TFPOC / SAM backbone: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
- ZSC regressor / weights: https://drive.google.com/drive/folders/1FjkaK2EzcOdiH_N9WkGnh5c3G9xj9PmE?usp=drive_link
- PseCo artifacts: https://huggingface.co/Hzzone/PseCo/tree/main/data/fsc147
- CountGD++ weights: https://drive.google.com/file/d/1j6N22TtKu2NVcKpgfrf-sJHGeLDqs9hs/view?usp=sharing

Follow each model's README (under `models/<ModelName>/`) for exact placement and additional model-specific setup steps.

## Running the Benchmark

This repository supports both single-class PrACo runs (original) and multi-class PrACo++ runs. Use the provided scripts in the repository root:

- `main.py` — run the original (single-class prompt-aware) evaluation on FSC-147 / PrACo
- `multiclass_main.py` — run the multi-class evaluation on MUCCA / PrACo++

Example: single-class (original PrACo)

```bash
python main.py --model CounTX --data_dir ./data --img_directory ./data/images_384_VarV2 --split test
```

Example: multi-class (PrACo++ on MUCCA benchmark)

```bash
python multiclass_main.py --model CounTX --data_dir ./data/multiclass-dataset --img_directory ./data/multiclass-dataset/images --split test
```

Notes:
- Replace `CounTX` with any supported model name listed in the next section.
- Set `--data_dir` to point to the MUCCA dataset folder (e.g., `data/multiclass-dataset`).
- Additional model-specific flags may be required (see `models/<ModelName>/README.md`).

## Multiclass tests, metrics and outputs

PrACo++ extends the original prompt-aware evaluation to multi-class settings. The main characteristics are:

- Multi-class prompts: images can contain multiple object classes and each evaluation prompt may ask to count a specific class.
- Per-class counts and per-prompt evaluation: predictions are evaluated per requested class and then aggregated across the dataset.
- Metrics: PrACo++ supports standard counting metrics such as Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) computed per class and aggregated across classes; additionally, the extension paper defines prompt-adherence and semantic-aware analyses to measure whether the model followed the textual prompt. For formal definitions see the extension paper: https://arxiv.org/pdf/2605.02752
- Localized metrics: density maps are partitioned into grids so the benchmark can compute localized MAE, precision, recall, and F-score over quadrants and finer sub-grids. This evaluation exists for the original FSC-147 benchmark as well when ground-truth density maps are available, and PrACo++ extends the same idea to the multiclass setting.

Outputs:
- Per-image, per-class prediction files (CSV/JSON) matching the format used by the evaluation scripts.
- Aggregated metrics CSVs.

You can reproduce the quantitative tables in the papers by running the multiclass evaluation pipeline and then the statistics scripts.

## Running statistics

Use the statistics scripts to aggregate per-image results produced by each model and create the final metrics tables used in the papers:

```bash
python multiclass_main_statistics.py --data_dir ./data/multiclass-dataset --split test --model "[CounTX,CLIP-Count]" --benchmark_inference_dir ./multiclass_benchmark_results
```

Or for the single-class benchmark:

```bash
python main_statistics.py --data_dir ./data --split test --model CounTX
```

The scripts output CSV files in the repository root and `benchmark_results/` folders.

## Available Models

The repository includes integration code or evaluation wrappers for the following models (and variants). See the corresponding `models/<ModelName>/` folders for per-model details and setup:

- CounTX
- CLIP-Count
- CountGD
- CountGD++
- GroundingREC
- VLCounter
- DAVE
- ZSC
- PseCo
- UPC
- TFPOC

## 🤝 Contributing & Benchmarking Your Own Model

We welcome contributions to the PrACo family! To evaluate your model on MUCCA:
1. Create an adapter class inheriting from `BaseModel` inside `models/your_model/`.
2. Implement the `infer` function ensuring it complies with the **strict fair-comparison protocols** (no privilege leakage from Ground Truth).
3. Open a Pull Request! We aim to maintain an updated leaderboard.




