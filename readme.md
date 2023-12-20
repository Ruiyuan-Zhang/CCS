# Co-creation Space - Code Release

## I. Introduction

We are excited to announce that our paper, titled "[Scalable Geometric Fracture Assembly via Co-creation Space among Assemblers](https://arxiv.org/abs/2312.12340)", has been accepted by the AAAI 2024. In line with our commitment to open science and reproducible research, we are preparing to release the code associated with our findings. `Co-creation Space`(CSS) is based on [Multi-Part Shape Assembly](https://github.com/Wuziyi616/multi_part_assembly).

## II. Traing codes

### 2.1 Data Preparation

Please visit [link](https://github.com/Wuziyi616/multi_part_assembly/blob/master/docs/install.md#data-preparation) for instructions on how to download these datasets.

-   [PartNet](https://partnet.cs.stanford.edu/) (semantic assembly)
-   [Breaking-Bad](https://breaking-bad-dataset.github.io/) (geometric assembly)


### 2.2 Virtual Environment Installation
```shell
conda create -n ccs python=3.8.5
conda activate ccs
conda install pytorch=1.10 torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install pytorch-lightning=1.6.2  # pip install lightning
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install pytorch3d -c pytorch3d   # pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu113_pyt1110/download.html

pip install setproctitle
pip install trimesh
pip install wandb
pip install scipy
pip install pyntcloud
pip install matplotlib
pip install ninja
```


### 2.3 Module Installation

1. Run `pip install -e .` to install this module
2. Go to `multi_part_assembly/utils/chamfer` and run `pip install -e .`
3. Go to `multi_part_assembly/models/modules/encoder/pointnet2/pointnet2_ops_lib` and run `pip install -e .`


### 2.4 Train Scripts
```shell
python scripts/train.py --project_name debug --gpu 0 --cfg configs/wx_transformer/wx_transformer/topk/partnet/FF-top10-partnet_chair.py
```

## III. Test Codes

### 3.1 Test Scripts
```shell
# categories = ['BeerBottle', 'Bowl', 'Cup', 'DrinkingUtensil', 'Mug', 'Plate', 'Spoon', 'Teacup', 'ToyFigure', 'WineBottle', 'Bottle', 'Cookie', 'DrinkBottle', 'Mirror', 'PillBottle', 'Ring', 'Statue', 'Teapot', 'Vase', 'WineGlass']
python scripts/test.py --gpus 0 --category BeerBottle --cfg_file configs/wx_transformer/wx_transformer/topk/partnet/FF-top10-partnet_chair.py --weight checkpoint/FF-top10-partnet_chair/models/model-epoch=499.ckpt --min_num_part 1 --max_num_part 20

```


## IV. Code Release Plan

Our code release will be structured in the following phases to ensure quality and ease of use:

1. **Core Code Organization**: We will first release the core algorithms and models that form the backbone of our research. This will include the primary scripts and modules necessary to understand and replicate the key results presented in our paper.

2. **Checkpoint Release**: Following the core code, we will provide pre-trained models or checkpoints. These will enable users to quickly reproduce our results without the need for extensive training.

3. **Documentation and Usage Instructions**: To facilitate the use of our code and models, comprehensive documentation will be provided. This will include detailed instructions on how to set up the environment, run the code, and replicate our experiments.

## V. Estimated Timeline

We are currently in the process of preparing the code for release. While we cannot provide an exact release date at this moment, we are targeting to complete the release process in the following order:

- Core Code: [Estimated 19/12/2023]
- Checkpoints: [Estimated 25/12/2023]
- Documentation: [Estimated 30/12/2023]

## VI. Stay Updated

To stay updated on our code release and other project news, please:

- Follow our [GitHub repository](#)
- Contact us at zhangruiyuan.0122@gmail.com (or zhangruiyuan@zju.edu.cn)

## VII. License

This project is released under the [MIT license](LICENSE).

## VIII. Acknowledgements

We would like to thank the AAAI reviewers and our colleagues who have contributed to this research. We are looking forward to sharing our work with the community and hope it will be a valuable resource. We also thank the benchmark provided by [Ziyi Wu](https://github.com/Wuziyi616).