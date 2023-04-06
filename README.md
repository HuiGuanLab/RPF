# From Region to Patch: Attribute-Aware Foreground-Background Contrastive Learning for Fine-Grained Fashion Retrieval

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
Following the previous work, we conduct experiments on three fashion related datasets, i.e., FashionAI, DARN, and DeepFashion. Please download them from the [URL](https://github.com/maryeon/asenpp#datasets) and put them in the corresponding folders.

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
