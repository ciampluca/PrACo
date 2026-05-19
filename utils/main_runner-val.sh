#!/bin/bash

# Function to check if a screen session exists
screen_exists() {
    screen -list | grep -q "$1"
}

# Launch screen sessions only if they don't already exist

if ! screen_exists "countx-praco-singleclass-val"; then
    echo "Starting countx-praco-singleclass-val..."
    screen -S countx-praco-singleclass-val -dm bash -c '
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate countx-38 
    CUDA_VISIBLE_DEVICES=0 python main.py --model CounTX --device cuda:0 --split val
    '
else
    echo "Screen countx-praco-singleclass-val is already running"
fi

if ! screen_exists "clipcount-praco-singleclass-val"; then
    echo "Starting clipcount-praco-singleclass-val..."
    screen -S clipcount-praco-singleclass-val -dm bash -c '
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate clipcount
    CUDA_VISIBLE_DEVICES=1 python main.py --model CLIP-Count --device cuda:0 --split val
    '
else
    echo "Screen clipcount-praco-singleclass-val is already running"
fi

if ! screen_exists "tfpoc-praco-singleclass-val"; then
    echo "Starting tfpoc-praco-singleclass-val..."
    screen -S tfpoc-praco-singleclass-val -dm bash -c '
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate tfpoc
    CUDA_VISIBLE_DEVICES=2 python main.py --model TFPOC --device cuda:0 --split val
    '
else
    echo "Screen tfpoc-praco-singleclass-val is already running"
fi

if ! screen_exists "vlcounter-praco-singleclass-val"; then
    echo "Starting vlcounter-praco-singleclass-val..."
    screen -S vlcounter-praco-singleclass-val -dm bash -c '
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate vlcounter
    CUDA_VISIBLE_DEVICES=3 python main.py --model VLCounter --device cuda:0 --split val
    '
else
    echo "Screen vlcounter-praco-singleclass-val is already running"
fi

if ! screen_exists "dave-praco-singleclass-val"; then
    echo "Starting dave-praco-singleclass-val..."
    screen -S dave-praco-singleclass-val -dm bash -c '
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate dave
    CUDA_VISIBLE_DEVICES=7 python main.py --model DAVE --device cuda:0 --split val
    '
else
    echo "Screen dave-praco-singleclass-val is already running"
fi

if ! screen_exists "zsc-praco-singleclass-val"; then
    echo "Starting zsc-praco-singleclass-val..."
    screen -S zsc-praco-singleclass-val -dm bash -c '
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate zsc
    CUDA_VISIBLE_DEVICES=2 python main.py --model ZSC --device cuda:0 --split val
    '
else
    echo "Screen zsc-praco-singleclass-val is already running"
fi

# pseco hf checkpoint: https://huggingface.co/Hzzone/PseCo/tree/main/data/fsc147/checkpoints
if ! screen_exists "pseco-praco-singleclass-val"; then
    echo "Starting pseco-praco-singleclass-val..."
    screen -S pseco-praco-singleclass-val -dm bash -c '
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate pseco
    CUDA_VISIBLE_DEVICES=6 python main.py --model PseCo --device cuda:0 --split val
    '
else
    echo "Screen pseco-praco-singleclass-val is already running"
fi

# dubbio: non ci dovrebbero essere due checkpoint per groundingrec? uno su fsc e uno sul loro dataset?
# io ne ho trovato solo uno, preso da qui: https://github.com/sydai/referring-expression-counting
if ! screen_exists "groundingrec-praco-singleclass-val"; then
    echo "Starting groundingrec-praco-singleclass-val..."
    screen -S groundingrec-praco-singleclass-val -dm bash -c '
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate groundingREC
    CUDA_VISIBLE_DEVICES=7 python main.py --model GroundingREC --device cuda:0 --split val
    '
else
    echo "Screen groundingrec-praco-singleclass-val is already running"
fi

if ! screen_exists "groundingrecFSC-praco-singleclass-val"; then
    echo "Starting groundingrecFSC-praco-singleclass-val..."
    screen -S groundingrecFSC-praco-singleclass-val -dm bash -c '
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate groundingREC
    CUDA_VISIBLE_DEVICES=7 python main.py --model GroundingRECFSC --device cuda:0 --split val
    '
else
    echo "Screen groundingrecFSC-praco-singleclass-val is already running"
fi

if ! screen_exists "countgd-praco-singleclass-val"; then
    echo "Starting countgd-praco-singleclass-val..."
    screen -S countgd-praco-singleclass-val -dm bash -c '
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate countgd
    CUDA_VISIBLE_DEVICES=0 python main.py --model CountGD --device cuda:0 --split val
    '
else
    echo "Screen countgd-praco-singleclass-val is already running"
fi

if ! screen_exists "fixedpointpromptcounting-praco-singleclass-val"; then
    echo "Starting fixedpointpromptcounting-praco-singleclass-val..."
    screen -S fixedpointpromptcounting-praco-singleclass-val -dm bash -c '
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate fxp-counting-38
    CUDA_VISIBLE_DEVICES=2 python main.py --model FixedPointPromptCounting --device cuda:0 --split val
    '
else
    echo "Screen fixedpointpromptcounting-praco-singleclass-val is already running"
fi