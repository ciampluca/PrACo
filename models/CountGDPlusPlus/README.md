# [CVPR 2026] CountGD++: Generalized Prompting for Open-World Counting (Adapted for PrACo)
Niki Amini-Naieni & Andrew Zisserman

Please refer to the up-to-date README.md at the official CountGD++ GitHub repository [here](https://github.com/niki-amini-naieni/CountGDPlusPlus/) for further information.

More details can be found in the paper, [[Paper]](https://arxiv.org/abs/2512.23351) [[Project page]](https://github.com/niki-amini-naieni/CountGDPlusPlus/).

<img src=img/teaser.jpg width="100%"/>
<strong>New capabilities of CountGD++.</strong>
<em>(a) Counting with Positive & Negative Prompts:</em> The negative visual exemplar enables CountGD++ to differentiate between cells that have the same round shape as the object to count but are of a different appearance;  
<em>(b) Pseudo-Exemplars:</em> Pseudo-exemplars are automatically detected from text-only input and fed back to the model, improving the accuracy of the final count for objects, like unfamiliar fruits, that are challenging to identify given text alone.

## CountGD++ Architecture
<img src=img/inference-architecture.jpg width="100%"/>

## Contents
* [Dependencies](#dependencies)
* [Reproduce Results From Paper](#reproduce-results-from-paper)
* [Citation](#citation)
* [Acknowledgements](#acknowledgements)

## Dependencies

### 1. Install GCC 

Install GCC. In this project, GCC 11.3 and 11.4 were tested. The following command installs GCC and other development libraries and tools required for compiling software in Ubuntu.

```
sudo apt update
sudo apt install build-essential
sudo apt install gcc-11 g++-11
```

### 2. Install CUDA Toolkit:

NOTE: In order to install detectron2 in step 4, you need to install the CUDA Toolkit. Refer to: https://developer.nvidia.com/cuda-downloads. If multiple CUDA versions are installed, make sure you are using the right one. [This repository](https://github.com/phohenecker/switch-cuda) is quite useful for switching between CUDA versions.

### 3. Set Up Anaconda Environment:

The following commands will create a suitable Anaconda environment for running and training CountGD++. To produce the results in the paper, we used [Anaconda version 2024.02-1](https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh).

```
conda create -n countgdplusplus python=3.10
conda activate countgdplusplus
conda install -c conda-forge gxx_linux-64 compilers libstdcxx-ng # ensure to install required compilers
cd CountGDPlusPlus
pip install -r requirements.txt
export CC=/usr/bin/gcc-11 # this ensures that gcc 11 is being used for compilation
cd models/GroundingDINO/ops
python setup.py build install
python test.py # should result in 6 lines of * True
cd ../../../
pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'
```

### 4. Download Pre-Trained Weights

* Make the ```checkpoints``` directory inside the ```CountGDPlusPlus``` folder.

  ```
  mkdir checkpoints
  ```

* Execute the following command.

  ```
  python download_bert.py
  ```

* Download the pretrained CountGD++ model available [here](https://drive.google.com/file/d/1j6N22TtKu2NVcKpgfrf-sJHGeLDqs9hs/view?usp=sharing) (1.25 GB), and place it in the ```checkpoints``` directory or use ```gdown``` to download the weights.

  ```
  pip install gdown
  gdown --id 1j6N22TtKu2NVcKpgfrf-sJHGeLDqs9hs
  ```
  
## Reproduce Results From Paper
Here we show how to reproduce the CountGD++ results on the PrACo benchmark. To reproduce results on the other benchmarks in the CountGD++ paper, please refer to the official CountGD++ repository [here](https://github.com/niki-amini-naieni/CountGDPlusPlus/).

To test the setting given both positive and negative text, run the following commands from inside the PrACo repository. Note that this setting uses both synthetic and pseudo-exemplars, obtained using only text. Please download the synthetic exemplars for FSC-147 from [here](https://drive.google.com/file/d/1XyE77D0bxEKDsPYvpaeD6KyXnSLye1Qg/view?usp=sharing). Unzip the folder inside the ```CountGDPlusPlus``` folder.

```
nohup python -u main.py --model CountGDPlusPlus --data_dir ./data --img_directory ./data/images_384_VarV2 --split test >>./test_countgdplusplus.log 2>&1 &
```

```
python main_statistics.py --data_dir data --split test --model CountGDPlusPlus
```

The results from these commands should produce the following output in ```final_metrics_test.csv```:

```
,Model,AvgNP,AvgNMN,PCCN,MAE,RMSE,AvgCntRecall,AvgCntPrecision,AvgCntFscore
0,CountGDPlusPlus,1.085,0.071,97.98,12.179,99.736,0.963,0.92,0.938
```

Note: the FSC-147 MAE and RMSE differ from those in the main paper because, for the PrACo benchmark, no adaptive cropping or test-time normalization are applied.

## Citation
Please cite our related papers if you build off of our work.
```
@InProceedings{AminiNaieni26b,
  title={CountGD++: Generalized Prompting for Open-World Counting},
  author={Amini-Naieni, N. and Zisserman, A.},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}

@InProceedings{AminiNaieni26a,
  title={Open-World Object Counting in Videos},
  author={Amini-Naieni, N. and Zisserman, A.},
  booktitle = {Association for Advancement of Artificial Intelligence Conference (AAAI)},
  year={2026}
}

@InProceedings{AminiNaieni24,
  title = {CountGD: Multi-Modal Open-World Counting},
  author = {Amini-Naieni, N. and Han, T. and Zisserman, A.},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year = {2024},
}
```

## Acknowledgements
The authors would like to thank Dr Christian Schroeder de Witt (Oxford Witt Lab, OWL) for his helpful feedback and insights on the paper figures and Gia Khanh Nguyen, Yifeng Huang, and Professor Minh Hoai for their help with the PairTally Benchmark. This research is funded by an AWS Studentship, the Reuben Foundation, a Qualcomm Innovation Fellowship (mentors: Dr Farhad Zanjani and Dr Davide Abati), the AIMS CDT program at the University of Oxford, EPSRC Programme Grant VisualAI EP/T028572/1, and a Royal Society Research Professorship RSRP\R\241003.
