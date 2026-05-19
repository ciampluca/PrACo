# open a screen session named countx-praco-multiclass, then run the following commands inside the screen session
#
screen -S countx-praco-multiclass -dm bash -c '
conda activate countx-38 
CUDA_VISIBLE_DEVICES=0 python multiclass_main.py --model CounTX --device cuda:0
'
screen -S clipcount-praco-multiclass -dm bash -c '
conda activate clipcount
CUDA_VISIBLE_DEVICES=1 python multiclass_main.py --model CLIP-Count --device cuda:0
'

screen -S tfpoc-praco-multiclass -dm bash -c '
conda activate tfpoc
CUDA_VISIBLE_DEVICES=2 python multiclass_main.py --model TFPOC --device cuda:0
'

screen -S vlcounter-praco-multiclass -dm bash -c '
conda activate vlcounter
CUDA_VISIBLE_DEVICES=3 python multiclass_main.py --model VLCounter --device cuda:0
'

screen -S dave-praco-multiclass -dm bash -c '
conda activate dave
CUDA_VISIBLE_DEVICES=7 python multiclass_main.py --model DAVE --device cuda:0
'

screen -S zsc-praco-multiclass -dm bash -c '
conda activate zsc
CUDA_VISIBLE_DEVICES=2 python multiclass_main.py --model ZSC --device cuda:0
'

screen -S pseco-praco-multiclass -dm bash -c '
conda activate pseco
CUDA_VISIBLE_DEVICES=6 python multiclass_main.py --model PseCo --device cuda:0
'

screen -S groundingrec-praco-multiclass -dm bash -c '
conda activate groundingREC
CUDA_VISIBLE_DEVICES=7 python multiclass_main.py --model GroundingREC --device cuda:0
'

screen -S countgd-praco-multiclass -dm bash -c '
conda activate countgd
CUDA_VISIBLE_DEVICES=0 python multiclass_main.py --model CountGD --device cuda:0
'

screen -S fixedpointpromptcounting-praco-multiclass -dm bash -c '
conda activate fxp-counting-38
CUDA_VISIBLE_DEVICES=1 python multiclass_main.py --model FixedPointPromptCounting --device cuda:0
'