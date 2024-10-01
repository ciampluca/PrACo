<img src="https://github.com/shizenglin/training-free-object-counter/blob/main/model.png" width="700" height="300">
<a href="http://arxiv.org/abs/2307.00038" target="_blank">Training-free Object Counting with Prompts</a> authored by Zenglin Shi, Ying Sun, Mengmi Zhang. <a href="http://arxiv.org/abs/2307.00038" target="_blank">[pdf]</a> <a href="https://github.com/shizenglin/training-free-object-counter/blob/main/poster.pdf" target="_blank">[poster]</a> <a href="https://youtu.be/KhN7-pf-6mE" target="_blank">[video]</a> 

<h2> Installation </h2>
1. The code requires python>=3.8, as well as pytorch>=1.7 and torchvision>=0.8. <br>
2. Please follow the instructions <a href="https://pytorch.org/get-started/locally/" target="_blank">here</a> to install both PyTorch and TorchVision dependencies. <br>
3. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.

<h2> Getting Started </h2>
1. Download the <a href="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth" target="_blank">'vit_b'</a> pre-trained model of SAM and save it to the folder 'pretrain'. <br>
2. Download the <a href="https://drive.google.com/file/d/1ymDYrGs9DSRicfZbSCDiOu0ikGDh5k6S/view?usp=sharing" target="_blank">FSC-147</a> and <a href="https://drive.google.com/file/d/0BwSzgS8Mm48Ud2h2dW40Wko3a1E/view?usp=sharing&resourcekey=0-34K_uP-vYM7EWq0Q2iIVaw" target="_blank">CARPK</a> datasets and save them to the folder 'dataset' <br>
3. Run 

```
python main-fsc147.py --test-split='test' --prompt-type='box' --device='cuda:0'
```
or

```
python main-carpk.py --test-split='test' --prompt-type='box' --device='cuda:0'
```

<h2> Success and failure results </h2>
<img src="https://github.com/shizenglin/training-free-object-counter/blob/main/results.png" width="700" height="450">

<h2> Acknowledgment </h2>
We express our sincere gratitude to the brilliant minds behind <a href="https://github.com/facebookresearch/segment-anything" target="_blank">SAM</a>, <a href="https://github.com/ZrrSkywalker/Personalize-SAM" target="_blank">Personalize-SAM</a> and <a href="https://github.com/xmed-lab/CLIP_Surgery" target="_blank">CLIP-Surgery</a>, as our code builds upon theirs. 

<h2> Citing </h2>
If you use our code in your research, please use the following BibTeX entry.

  ```
  @inproceedings{Shi2023promptcounting,
    title={Training-free Object Counting with Prompts},
    author={Zenglin Shi, Ying Sun, Mengmi Zhang},
    booktitle={WACV},
    year={2024}
  }
 ```


