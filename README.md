# A2J-Transformer

## Introduction
This is the official implementation for the paper, **"A2J-Transformer: Anchor-to-Joint Transformer Network for 3D Interacting Hand Pose Estimation from a Single RGB Image"**, CVPR 2023. 

Paper link here: [A2J-Transformer: Anchor-to-Joint Transformer Network for 3D Interacting Hand Pose Estimation from a Single RGB Image](https://arxiv.org/abs/2304.03635)

# About our code 

## Updates 

* (2023-9-19) Training code released! All training codes for Interhand 2.6M, NYU and HANDS 2017 dataset.
* (2023-9-19) Updated `test.py` and `base.py`, added the argparse `test_epoch`. Updated the name of our pre-trained model, from `snapshot.pth.tar` to `snapshot_0.pth.tar`.
* (2023-4-23) Deleted some compiled files in the `dab_deformable_detr\ops` folder.


## Installation and Setup

### Requirements

* Our code is tested under Ubuntu 20.04 environment with NVIDIA 2080Ti GPU and NVIDIA 3090 GPU, both Pytorch1.7 and Pytorch1.11 work.
  
* Python>=3.7

    We recommend you to use Anaconda to create a conda environment:
    ```bash
    conda create --name a2j_trans python=3.7
    ```
    Then, activate the environment:
    ```bash
    conda activate a2j_trans
    ```
  
* PyTorch>=1.7.1, torchvision>=0.8.2 (following instructions [here](https://pytorch.org/))

    We recommend you to use the following pytorch and torchvision:
    ```bash
    conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
    ```
  
* Other requirements
    ```bash
    conda install tqdm numpy matplotlib scipy
    pip install opencv-python pycocotools
    ```

### Compiling CUDA operators(Following [Deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR))
```bash
cd ./dab_deformable_detr/ops
sh make.sh
```

## Usage

### Dataset preparation

* Please download [InterHand 2.6M Dataset](https://mks0601.github.io/InterHand2.6M/) and organize them as following:

    ```
    your_dataset_path/
    └── Interhand2.6M_5fps/
        ├── annotations/
        └── images/
    ```



### Testing on InterHand 2.6M Dataset

* Please download our [pre-trained model](https://drive.google.com/file/d/1QKqokPnSkWMRJjZkj04Nhf0eQCl66-6r/view?usp=share_link) and organize the code as following:

    ```
    a2j-transformer/
    ├── dab_deformable_detr/
    ├── nets/
    ├── utils/
    ├── ...py
    ├── datalist/
    |   └── ...pkl
    └── output/
        └── model_dump/
            └── snapshot_0.pth.tar
    ```
    The `datalist` folder and the pkl files denotes the dataset-list generated during running the code. 


* In `config.py`, set `interhand_anno_dir`, `interhand_images_path` to the dataset abs directory.
* In `config.py`, set `cur_dir` to the a2j-transformer code directory.
* Run the following script:
    ```python
    python test.py --gpu <your_gpu_ids>
    ```
    You can also choose to change the `gpu_ids` in `test.py`.


### NYU and HANDS 2017 dataset

* Please use our code following [A2J](https://github.com/zhangboshen/A2J) to prepare the datasets.
* If you have prepared, just run the `nyu.py` and `hands2017.py`.



# Cite
Our code is protected by patents and **cannot be used for commercial purposes**. If you have commercial needs, please contact Prof. Yang Xiao (Huazhong University of Science and Technology): `Yang_Xiao@hust.edu.cn`.

If you find our work useful in your research or publication, please cite our work:
```
@inproceedings{jiang2023a2j,
  title={A2J-Transformer: Anchor-to-Joint Transformer Network for 3D Interacting Hand Pose Estimation from a Single RGB Image},
  author={Jiang, Changlong and Xiao, Yang and Wu, Cunlin and Zhang, Mingyang and Zheng, Jinghong and Cao, Zhiguo and Zhou, Joey Tianyi},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={8846--8855},
  year={2023}
}
```