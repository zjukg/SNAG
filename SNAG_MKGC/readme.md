
## 📚 Dataset and Code
❗Download from [Here](https://ufile.io/3te19dx0)

## 🚀 MKGC Train
- **Quick start**: Using  script file (`run.sh`)
```bash
>> cd SNAG_MKGC
>> bash run.sh
```
- **Optional**: Using the `bash command`:
```bash
# Command Details:
#  GPU | DATA | num_proj | use_intermediate | joint_way | noise ratio (0~1) | mask ratio (0~1) | noise-level (Epoch/Step) | num_hidden_layers | num_attention_heads | Exp ID

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
