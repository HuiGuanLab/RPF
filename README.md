# From Region to Patch: Attribute-Aware Foreground-Background Contrastive Learning for Fine-Grained Fashion Retrieval
This is a repository contains the implementation of our SIGIR'23 full paper [From Region to Patch: Attribute-Aware Foreground-Background Contrastive Learning for Fine-Grained Fashion Retrieval](https://doi.org/10.48550/arXiv.2305.10260).
![network structure](imgs/myframework.jpg)

## Table of Contents

* [Environments](#environments)
* [Datasets](#datasets)
* [Configuration](#configuration)
* [Training](#training)
* [Evaluation](#evaluation)
* [Performance](#performance)

## Environments
- **Ubuntu** 20.04
- **CUDA** 11.7
- **Python** 3.7

Install other required packages by
```sh
pip install -r requirements.txt
```

## Datasets
Following the previous work, we conduct experiments on three fashion related datasets, i.e., FashionAI, DARN, and DeepFashion. Please download and put them in the corresponding folders.
### Download Data
#### FashionAI

As the full FashionAI has not been publicly released, we utilize its early version for the [FashionAI Global Challenge 2018](https://tianchi.aliyun.com/competition/entrance/231671/introduction?spm=5176.12281949.1003.9.493e3eafCXLQGm). You can first sign up and download two training subsets:

- **fashionAI_attributes_train1.zip(6G)**
- **fashionAI_attributes_train2.zip(7G)**. 

Once done, you should uncompress and link them into the `data/FashionAI` directory.

#### DARN

As some images’ URLs have been broken, only 214,619 images are obtained for our experiments. We provide with a series of [URLs](https://drive.google.com/file/d/10jpHsFI2njzEGl7kdACXbvstz6tXyE0R/view?usp=sharing) for the images. Please download them into a `pic` directory that should be created in `data/DARN` directory.

#### DeepFashion

[DeepFashion](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Liu_DeepFashion_Powering_Robust_CVPR_2016_paper.pdf) is a large dataset which consists of four benchmarks for various tasks in the field of clothing including [category and attribute prediction](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html) which we use for our experiments, in-shop clothes retrieval, fashion landmark  detection and consumer-to-shop clothes retrieval. Download the images into a `img` directory that should be created in `data/DeepFashion` directory.

### Configuration

The behavior of our codes is controlled by configuration files under the `config` directory. 

```sh
config
│── FashionAI
│   ├── FashionAI.yaml
│   ├── s1.yaml
│   └── s2.yaml
├── DARN
│   ├── DARN.yaml
│   ├── s1.yaml
│   └── s2.yaml
└── DeepFashion
    ├── DeepFashion.yaml
    ├── s1.yaml
    └── s2.yaml
```

Each dataset is configured by two types of configuration files. One is `<Dataset>.yaml` that specifies basic dataset information such as path to the training data and annotation files. The other two set some training options as needed.

If the above `data` directory is placed at the same level with `main.py`, no changes are needed to the configuration files. Otherwise, be sure to correctly configure relevant path to the data according to your working environment.

## Training

Download Google pre-trained ViT models for our Patch-aware Branch:
```bash
wget https://drive.google.com/file/d/1N2rdQcbhegIOB4fHpifi92w1Lp86umN1/view?usp=sharing
```

RPF is trained in a two-stage way. For the first stage, we need to train the region-aware branch. Run the following script that uses default settings:

```python
python main.py --cfg config/<Dataset>/<Dataset>.yaml config/<Dataset>/s1.yaml
```

Based on the trained region-aware branch, the second stages jointly train the whole RPF:

```python
python main.py --cfg config/<Dataset>/<Dataset>.yaml config/<Dataset>/s2.yaml --resume runs/<Dataset>_s1/model_best.pth.tar
```

## Evaluation

Run the following script to test on the trained models:

```python
python main.py --cfg config/<Dataset>/<Dataset>.yaml config/<Dataset>/s2.yaml --resume runs/<Dataset>_s2/model_best.pth.tar --test TEST
```
We release several pretrained models:
- RPF on FashionAI: [released_model](https://drive.google.com/file/d/1pIJ2REblm2eXNq81vyhAj9bs8y1EzNvR/view?usp=sharing)
- RPF on DARN: [released_model](https://drive.google.com/file/d/1icLsQG7g1LL41i-T75IZb2wsRTkeIemQ/view?usp=sharing)
- RPF on DeepFashion: [released_model](https://drive.google.com/file/d/1BKuXrQWuQaou_1AGzONoKeEQ2iA6J6v3/view?usp=sharing)  

### Performance 
Expected MAP on FashionAI Dataset
|             |skirt length| sleeve length| coat length |pant length |collar design| lapel design| neckline design| neck design| overall|
| :---------: | :--: | :--: | :--: | :---: | :---: |:--: | :--: | :---: | :---: |
|RPF|66.75 |67.86 |59.65| 73.23| 75.72| 73.18 |74.40 |75.01 |70.11|

Expected MAP on DARN Dataset
|             | clothes category |clothes button |clothes color |clothes length |clothes pattern| clothes shape |collar shape| sleeve length |sleeve shape |overall|
| :---------: | :--: | :--: | :--: | :---: | :---: |:--: | :--: | :--: | :---: | :---: |
|RPF|45.18 |54.92 |55.08 |63.51| 57.04| 63.54| 41.20 |86.95| 62.43 |58.80|

Expected MAP on DeepFashion Dataset
|             |texture |fabric |shape |part| style |overall|
| :---------: | :--: | :--: | :--: | :---: | :---: | :---: |
|RPF| 15.62| 8.30 |15.02| 7.38| 4.77|10.22|
## Citation
If you find this repository useful, please consider citing our paper:
```
@inproceedings{RPF2023,
  title={From Region to Patch: Attribute-Aware Foreground-Background Contrastive Learning for Fine-Grained Fashion Retrieval},
  author={Jianfeng Dong and Xiaoman Peng and Zhe Ma and Daizong Liu and Xiaoye Qu and Xun Yang and Jixiang Zhu and Baolong Liu},
  booktitle={Proceedings of the 46rd International ACM SIGIR Conference on Research and Development in Information Retrieval},
  year={2023}
}

```
