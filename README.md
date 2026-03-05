# RepLK-SE Interactive Medical Image Segmentation

This repository contains the implementation of our interactive medical image segmentation framework with **RepLK and SE attention enhancements**.

Our method improves the robustness of **one-click interactive segmentation** by introducing:

- RepLK-style large-kernel gating
- Squeeze-and-Excitation (SE) recalibration
- Cross-block refinement in late fusion stages
- Prototype-conditioned segmentation

The framework is designed for **interactive medical image segmentation under extremely limited user input (one-click setting)**.

---
![img](https://github.com/Faker-six/replk-se-interactive-segmentation/blob/asset/img1.png)

---
RepLK modules are inserted into the last two CrossBlocks.
![img](https://github.com/Faker-six/replk-se-interactive-segmentation/blob/asset/img2.png)

# Important Dependency

> **Important:**  
> This repository is **based on the official MultiverSeg implementation**.  
> Therefore, you **must first download the original MultiverSeg repository**, as our code is built directly on top of its framework.

Official MultiverSeg repository:

https://github.com/halleewong/MultiverSeg

Please follow the installation instructions in the official repository before running our code.

---
Download MultiverSeg pre-trained weights v1 [here](https://www.dropbox.com/scl/fo/71j9vl3d4db0u229rq689/AI_5oDICnt0HnBcry-xJSNQ?rlkey=7y42638h12ilqds8270owzric&st=3py413ys&dl=0).

## How to run (BUSI)
Download BUSI Dataset [here](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset).
### Train (finetune from the official pretrained weights provided by MultiverSeg)(replk+se)

This training script **finetunes from the official MultiverSeg pretrained checkpoint**.  
Before training, download the official pretrained weights:



```bash
python scripts/busi/multiverseg_busi_finetune_v5.py \
  --root /path/to/Dataset_BUSI_with_GT \
  --outdir /path/to/runs_busi_finetune_v5 \
  --img 256 --epochs 80 --batch 2 --workers 4

### Where to find checkpoints (trained weights)

After training, checkpoints are saved under the folder you pass to `--outdir`, e.g.

* `/path/to/runs_busi_finetune_v5/best.pt`
* `/path/to/runs_busi_finetune_v5/epoch_XX.pt`

Quickly list them:

```bash
ls -lh /path/to/runs_busi_finetune_v5 | grep -E "best\.pt|epoch_.*\.pt"
# or
ls -1 /path/to/runs_busi_finetune_v5/*.pt
```

### Test / Evaluate (one-click)+(replk+se)

```bash
python scripts/busi/test_busi_oneclick_from_trainpack.py \
  --root /path/to/Dataset_BUSI_with_GT \
  --ckpt /path/to/runs_busi_finetune_v5/best.pt \
  --outdir /path/to/output_busi_oneclick_test \
  --img 256 --num 80 \
  --support_k 2 --bank 200 --proto_m 48 --alpha 0.5 --beta 0.5
```
![img](https://github.com/Faker-six/replk-se-interactive-segmentation/blob/asset/img3.png)
### How to switch to a different checkpoint

Just change `--ckpt` to the `.pt` you want, e.g.

```bash
# use epoch checkpoint
python scripts/busi/test_busi_oneclick_from_trainpack.py \
  --root /path/to/Dataset_BUSI_with_GT \
  --ckpt /path/to/runs_busi_finetune_v5/epoch_20.pt \
  --outdir /path/to/output_busi_oneclick_test_epoch20 \
  --img 256 --num 80 \
  --support_k 2 --bank 200 --proto_m 48 --alpha 0.5 --beta 0.5
```
