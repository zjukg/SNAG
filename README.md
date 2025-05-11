# SNAG
![](https://img.shields.io/badge/version-1.0.1-blue)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/zjukg/SNAG/blob/main/LICENSE)
[![arxiv badge](https://img.shields.io/badge/arxiv-2403.06832-red)](https://arxiv.org/abs/2403.06832)
[![Pytorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?e&logo=PyTorch&logoColor=white)](https://pytorch.org/)


>In this work, we introduce a **Unified Multi-Modal Knowledge Graph (MMKG) Representation Framework** that incorporates tailored training objectives for Multi-modal Knowledge Graph Completion (MKGC) and Multi-modal Entity Alignment (MMEA). Our approach achieves SOTA performance across a comprehensive suite of ten datasets, including three for MKGC and seven for MMEA, demonstrating the framework's effectiveness and versatility in diverse multi-modal contexts.

<div align="center">
    <img src="https://github.com/zjukg/SNAG/blob/main/Image/model.jpg" width="90%" height="auto" />
</div>



<!--Code and Data for paper: `The Power of Noise: Toward a Unified Multi-modal Knowledge Graph Representation Framework`-->

## 🔔 News
- **`2024-11` SNAG is accepted by COLING 2024 !**
- **`2024-02` We preprint our Survey [Knowledge Graphs Meet Multi-Modal Learning: A Comprehensive Survey](http://arxiv.org/abs/2402.05391)  [[`Repo`](https://github.com/zjukg/KG-MM-Survey)].**


## 🔬 Dependencies
```bash
pip install -r requirement.txt
```
#### Details
- Python (>= 3.7)
- [PyTorch](http://pytorch.org/) (>= 1.6.0)
- numpy (>= 1.19.2)
- [Transformers](http://huggingface.co/transformers/) (== 4.21.3)
- easydict (>= 1.10)
- unidecode (>= 1.3.6)
- tensorboard (>= 2.11.0)


## 🚀 MKGC Train
- **Quick start**: Using  script file (`run.sh`)
```bash
>> cd SNAG_MKGC
>> bash run.sh
```
- **Optional**: Using the `bash command`:
```bash
# Command Details:
#  GPU | DATA | num_proj | use_intermediate | joint_way | noise ratio | mask ratio | noise-level | num_hidden_layers | num_attention_heads | Exp ID

# # DATA=DB15K / MKG-W / MKG-Y
# num_proj: 1 / 2
# use_intermediate: 0 / 1
# joint_way: "Mformer_hd_mean" / "Mformer_hd_graph" / "Mformer_weight" / "atten_weight" / "learnable_weight"
# noise ratio: 0 ~ 1
# mask ratio: 0 ~ 1
# noise-level: epoch / step

>> bash scripts/run_base.sh 0 DB15K 2 0 Mformer_hd_graph 0.2 0.7 epoch 1 2 K001
>> bash scripts/run_base.sh 0 MKG-Y 1 0 Mformer_hd_mean 0.2 0.7 epoch 1 2 Y001
>> bash scripts/run_base.sh 0 MKG-W 1 0 Mformer_hd_mean 0.2 0.7 epoch 1 2 W001
```

❗Tips: you can open the `run.sh` file for parameter or training target modification.

- **Optional**: Modifying the basic parameters:
  - you can open the `scripts/run_base.sh` file for parameter or training target modification
    - Make `NOISE = 0` to abandon the gauss modality noise masking.
    - `EPOCH` can be set to `8000` as early stopping is employed by default.
    - The `noise_level` or `noise_update` parameter determines whether the Noise mask is updated at every step or every epoch. Through experimentation, we have found that updating at the epoch level is sufficient.
    - The `use_pool` flag indicates that pooling operations are applied to all pre-extracted visual/text features for dimensionality reduction and uniformity in dimensions.

```bash
EMB_DIM=128
NUM_BATCH=1024
MARGIN=12
LR=1e-4
LRG=1e-4
NEG_NUM=32
EPOCH=8000
NOISE=1
POOL=1
```

## 🚀 MMEA Train
- **Quick start**: Using  script file (`run.sh`)
```bash
>> cd SNAG_MMEA
>> bash run.sh 0
```
- **Optional**: Using the `bash command`
```bash
# Command Details:
# bash file | GPU | Dataset | data split | R_{sa} | random seed | use_surface | R_{img} | noise ratio | mask ratio |
# Begin:
>> bash run_snag.sh 0 DBP15K ja_en 0.3 3408 0 1.0 0.2 0.7
>> bash run_snag.sh 0 DBP15K ja_en 0.3 3408 0 0.6 0.2 0.7
>> bash run_snag.sh 0 DBP15K ja_en 0.3 3408 0 0.4 0.2 0.7
```

❗Tips: you can open the `run_XXX.sh` file for parameter or training target modification.

## 📚 Dataset
❗MMEA: From [UMAEA Repo](https://github.com/zjukg/UMAEA) 

❗MKGC: Download from [Google Driven](https://drive.google.com/file/d/1jIYo7JWXNNi7LZqBIsl103FL8tHcoW-q/view?usp=sharing) or [Ufile](https://ufile.io/3te19dx0) 

## 🤝 Cite:
Please condiser citing this paper if you use the ```code``` or ```data``` from our work.
Thanks a lot :)

```bigquery
@article{chen2024power,
  title={The Power of Noise: Toward a Unified Multi-modal Knowledge Graph Representation Framework},
  author={Chen, Zhuo and Fang, Yin and Zhang, Yichi and Guo, Lingbing and Chen, Jiaoyan and Chen, Huajun and Zhang, Wen},
  journal={arXiv preprint arXiv:2403.06832},
  year={2024}
}
```

