# open a screen session named countx-praco-singleclass, then run the following commands inside the screen session
#
screen -S countx-praco-singleclass -dm bash -c '
conda activate countx-38 
CUDA_VISIBLE_DEVICES=0 python main.py --model CounTX --device cuda:0
'
screen -S clipcount-praco-singleclass -dm bash -c '
conda activate clipcount
CUDA_VISIBLE_DEVICES=1 python main.py --model CLIP-Count --device cuda:0
'

screen -S tfpoc-praco-singleclass -dm bash -c '
conda activate tfpoc
CUDA_VISIBLE_DEVICES=2 python main.py --model TFPOC --device cuda:0
'

screen -S vlcounter-praco-singleclass -dm bash -c '
conda activate vlcounter
CUDA_VISIBLE_DEVICES=3 python main.py --model VLCounter --device cuda:0
'

screen -S dave-praco-singleclass -dm bash -c '
conda activate dave
CUDA_VISIBLE_DEVICES=7 python main.py --model DAVE --device cuda:0
'

screen -S zsc-praco-singleclass -dm bash -c '
conda activate zsc
CUDA_VISIBLE_DEVICES=2 python main.py --model ZSC --device cuda:0
'
# pseco hf checkpoint: https://huggingface.co/Hzzone/PseCo/tree/main/data/fsc147/checkpoints
screen -S pseco-praco-singleclass -dm bash -c '
conda activate pseco
CUDA_VISIBLE_DEVICES=6 python main.py --model PseCo --device cuda:0
'

# dubbio: non ci dovrebbero essere due checkpoint per groundingrec? uno su fsc e uno sul loro dataset?
# io ne ho trovato solo uno, preso da qui: https://github.com/sydai/referring-expression-counting
screen -S groundingrec-praco-singleclass -dm bash -c '
conda activate groundingREC
CUDA_VISIBLE_DEVICES=7 python main.py --model GroundingREC --device cuda:0
'

screen -S groundingrecFSC-praco-singleclass -dm bash -c '
conda activate groundingREC
CUDA_VISIBLE_DEVICES=7 python main.py --model GroundingRECFSC --device cuda:0
'

screen -S countgd-praco-singleclass -dm bash -c '
conda activate countgd
CUDA_VISIBLE_DEVICES=0 python main.py --model CountGD --device cuda:0
'

screen -S fixedpointpromptcounting-praco-singleclass -dm bash -c '
conda activate fxp-counting-38
CUDA_VISIBLE_DEVICES=2 python main.py --model FixedPointPromptCounting --device cuda:0
'