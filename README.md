# PrACo: Presence Aware Counting Benchmark

## Table of Contents

1. [Overview](#overview)
2. [Repository Structure](#repository-structure)
3. [Installation](#installation)
   - [1. Create a Conda Environment](#1-create-a-conda-environment)
   - [2. Download the FSC-147 Dataset](#2-download-the-fsc-147-dataset)
   - [3. Clone and Install Model Repositories](#3-clone-and-install-model-repositories)
   - [4. Download Pre-Trained Weights](#4-download-pre-trained-weights)
4. [Running the Benchmark](#running-the-benchmark)
5. [Example](#example)
6. [Running Statistics](#running-statistics)
7. [Available Models](#available-models)

## Overview

**Mind the Prompt: A Novel Benchmark for Prompt-based Class-Agnostic Counting** introduces a new benchmark named **PrACo** (**Pr**esence **A**ware **Co**unting Benchmark) designed to evaluate the performance of different models in counting objects in images with an awareness of object presence. This repository includes the necessary scripts and instructions to run the benchmark and evaluate the models described in the paper.

## Repository Structure

The repository is organized as follows:

- **`benchmark/`**: Scripts for evaluating the models on the PrACo dataset.
- **`models/`**: Contains the model implementations for:
  - `CounTX`
  - `CLIP-Count`
  - `TFPOC`
  - `VLCounter`
  - `DAVE`
  
- **`main.py`**: Main script to run the benchmark for a selected model.
- **`main_statistics.py`**: Script for computing and compiling benchmark statistics across different models.
- **`statistics.ipynb`**: notebook to reproduce statistics of the paper.
- **`qualitative.ipynb`**: notebook to reproduce qualitative results of the paper.
- **`requirements.txt`**: Dependencies required for the benchmark scripts and model evaluation.

## Installation

#### 1. Create a Conda Environment

To set up the environment for running the benchmark, execute the following commands:

```bash
conda create --name praco python=3.10
conda activate praco
pip install -r requirements.txt
```

### 2. Download the FSC-147 Dataset

This project uses the FSC-147 dataset for object counting. Download it from the following links:

- FSC-147 Dataset [Download](https://drive.google.com/file/d/1ymDYrGs9DSRicfZbSCDiOu0ikGDh5k6S/view?usp=sharing)
- Image Descriptions [FSC-147-D](https://github.com/niki-amini-naieni/CounTX/blob/main/FSC-147-D.json)
- Put the dataset (zipped) and the FSC-147-D file into the `data` folder
- Extract the dataset:
```bash
unzip FSC147_384_V2.zip
```


### 3. Download Pre-Trained Weights

The model weights used in the paper can be downloaded from the respective authors' repositories and must be placed in `pretrained_models/` which should be created in the project root. 
Download links are provided below.:

- **CounTX Weights:** [Download Link](https://drive.google.com/file/d/1Vg5Mavkeg4Def8En3NhceiXa-p2Vb9MG/view?usp=sharing)
- **CLIP-Count Weights:** [Download Link](https://drive.google.com/file/d/17Dj0tjd29lPGOGYEF5IrE8aPClXUjTrR/view?usp=drive_link)
- **VLCounter Weights:** [Download Link](https://drive.google.com/file/d/1-2lqtsOm9XW4MXhLzrB5Jf9RkXOpDlaQ/view?usp=sharing)
- **DAVE Weights:** [Download Link](https://drive.google.com/drive/folders/10O4SB3Y380hcKPIK8Dt8biniVbdQ4dH4?usp=sharing) 
  - Download verification.pth
  - Download and extract DAVE_0_shot.pth from models.zip
- **TFPOC Weights:** [Download Link](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)


### 4. Download Model-specific Files
- **CLIP weights:** [Download Link](https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt) - Put it into the`models/VLCounter/pretrain` folder


## Running the Benchmark

To evaluate a model on the PrACo benchmark, use the following command:

```bash
python main.py --model <MODEL_NAME> --data_dir <DATA_DIR> --img_directory <IMG_DIR> --split <SPLIT_NAME>
```

## Example

To run a model, use the following command:

```bash
python main.py --model CounTX --data_dir ../CounTX/data/FSC/FSC_147 --img_directory ../CounTX/data/FSC/images_384_VarV2 --split test
```

## Running Statistics

To generate statistics and final metrics for the benchmark:

```bash
python main_statistics.py --data_dir <DATA_DIR> --split <SPLIT_NAME>
```

## Available Models
- **CounTX**
- **CLIP-Count**
- **ClipSAM**
- **VLCounter**
- **DAVE**


