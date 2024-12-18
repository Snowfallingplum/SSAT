# SSAT: A Symmetric Semantic-Aware Transformer Network for Makeup Transfer and Removal

This is the official pytorch code for **"SSAT: A Symmetric Semantic-Aware Transformer Network for Makeup Transfer and Removal"**, which has been accepted by AAAI2022.

**The training code, testing code, dataset, and pre-trained model have all been open sourced**
## Author
**Zhaoyang Sun; Yaxiong Chen; Shengwu Xiong**


## News

+ Our paper SHMT was accepted by NeurIPS2024. [Paper link](https://arxiv.org/abs/2412.11058) and [code link](https://github.com/Snowfallingplum/SHMT).

+ Our paper SSAT++ was accepted by TNNLS2023. [Paper link](https://ieeexplore.ieee.org/document/10328655) and [code link](https://github.com/Snowfallingplum/SSAT_plus).

+ Our paper SSAT was accepted by AAAI2022. [Paper link](https://arxiv.org/abs/2112.03631) and [code link](https://github.com/Snowfallingplum/SSAT).


## The framework of SSAT

![](asset/network.jpg)


## Quick Start

If you only want to get results quickly, please go to the *"quick_start"* folder and follow the readme.md inside to download the pre trained model to generate results quickly.


## Requirements

We recommend that you just use your own pytorch environment; the environment needed to run our model is very simple. If you do so, please ignore the following environment creation.

A suitable [conda](https://conda.io/) environment named `SSAT` can be created
and activated with:

```
conda env create -f environment.yaml
conda activate SSAT
```
## Download our dataset
Our dataset can be downloaded here [Baidu Drive](https://pan.baidu.com/s/1ozcLdlsykv3tb32X2bfP3w), password: cdrb.

Extract the downloaded file and place it on top of this folder.

## Training code
We have set the default hyperparameters in the options.py file, please modify them yourself if necessary.
To train the model, please run the following command directly
```
python train.py
```

## Inference code

```
python inference.py
```

## Our results

![](asset/transfer_results.jpg)

## Citation

If this work is helpful for your research, please consider citing the following BibTeX entry.

```text
@inproceedings{sun2022ssat,
  title={Ssat: A symmetric semantic-aware transformer network for makeup transfer and removal},
  author={Sun, Zhaoyang and Chen, Yaxiong and Xiong, Shengwu},
  booktitle={Proceedings of the AAAI Conference on artificial intelligence},
  pages={2325--2334},
  year={2022}
}
```


## Acknowledgement

Some of the codes are build upon [PSGAN](https://github.com/wtjiang98/PSGAN), [Face Parsing](https://github.com/zllrunning/face-parsing.PyTorch) and [aster.Pytorch](https://github.com/ayumiymk/aster.pytorch).

## License

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg

