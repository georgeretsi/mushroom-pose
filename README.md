# SMIRK: 3D Facial Expressions through Analysis-by-Neural-Synthesis

This is the official PyTorch implementation of SMIRK:
This repository is the official implementation of the [CVPR 2024](https://cvpr.thecvf.com) paper [3D Facial Expressions through Analysis-by-Neural Synthesis](https://arxiv.org/abs/2404.04104).


<p align="center">
  <a href='https://arxiv.org/abs/2404.04104' style='padding-left: 0.5rem;'>
    <img src='https://img.shields.io/badge/arXiv-2404.04104-brightgreen' alt='arXiv'>
  </a>
  <!-- <a href=''>
    <img src='https://img.shields.io/badge/PDF-Paper-2D963D?style=flat&logo=Adobe-Acrobat-Reader&logoColor=red' alt='Paper PDF'>
  </a>  -->
  <!-- <a href=''>
    <img src='https://img.shields.io/badge/PDF-Sup.Mat.-2D963D?style=flat&logo=Adobe-Acrobat-Reader&logoColor=red' alt='Sup. Mat. PDF'>
  </a>      -->
  <a href='https://www.youtube.com/watch?v=8ZVgr41wxbk' style='padding-left: 0.5rem;'>
    <img src='https://img.shields.io/badge/Video-Youtube-red?style=flat&logo=youtube&logoColor=red' alt='Youtube Video'>
  </a>
  <a href='https://georgeretsi.github.io/smirk/' style='padding-left: 0.5rem;'>
    <img src='https://img.shields.io/badge/Website-Project Page-blue?style=flat&logo=Google%20chrome&logoColor=blue' alt='Project Page'>
  </a>
</p>

<p align="center"> 
<img src="samples/cover.png">
SMIRK reconstructs 3D faces from monocular images with facial geometry that faithfully recover extreme, asymmetric, and subtle expressions.
</p>


## Installation
You need to have a working version of PyTorch and Pytorch3D installed. We provide a `requirements.txt` file that can be used to install the necessary dependencies for a Python 3.9 setup with CUDA 11.7:

```bash
conda create -n mpose python=3.9
conda activate mpose

conda install openblas-devel -c anaconda
conda install pytorch=2.0.1 torchvision cudatoolkit=11.0 -c pytorch -c conda-forge
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" --install-option="--blas=openblas"

pip install open3d 
```

Then, in order to download the required models, run:

```bash
bash quick_install.sh
```
*The above installation includes downloading the [FLAME](https://flame.is.tue.mpg.de/) model. This requires registration. If you do not have an account you can register at [https://flame.is.tue.mpg.de/](https://flame.is.tue.mpg.de/)*



## Demo 
We provide a demo that can be used to test the model on a single image or a video file. 

```bash
python demo.py --input_path samples/test_image2.png --out_path results/ --checkpoint pretrained_models/SMIRK_em1.pt --crop
```


## Training