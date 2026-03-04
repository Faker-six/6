# RepLK-SE Interactive Medical Image Segmentation

This repository contains the implementation of our interactive medical image segmentation framework with **RepLK and SE attention enhancements**.

Our method improves the robustness of **one-click interactive segmentation** by introducing:

- RepLK-style large-kernel gating
- Squeeze-and-Excitation (SE) recalibration
- Cross-block refinement in late fusion stages
- Prototype-conditioned segmentation

The framework is designed for **interactive medical image segmentation under extremely limited user input (one-click setting)**.

---

# Important Dependency

> **Important:**  
> This repository is **based on the official MultiverSeg implementation**.  
> Therefore, you **must first download the original MultiverSeg repository**, as our code is built directly on top of its framework.

Official MultiverSeg repository:

https://github.com/halleewong/MultiverSeg

Please follow the installation instructions in the official repository before running our code.

---

# Installation

First clone the official MultiverSeg repository:

```bash
git clone https://github.com/halleewong/MultiverSeg.git
cd MultiverSeg


## How to run (BUSI)

### Train (finetune from the official pretrained weights provided by MultiverSeg)

This training script **finetunes from the official MultiverSeg pretrained checkpoint**.  
Before training, download the official pretrained weights:

# Download pretrain weights

Run
````
./download.sh
````

 Download MultiverSeg weights from [here](https://www.dropbox.com/scl/fo/71j9vl3d4db0u229rq689/AI_5oDICnt0HnBcry-xJSNQ?rlkey=7y42638h12ilqds8270owzric&st=3py413ys&dl=0)

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

### Test / Evaluate (one-click)

```bash
python scripts/busi/test_busi_oneclick_from_trainpack.py \
  --root /path/to/Dataset_BUSI_with_GT \
  --ckpt /path/to/runs_busi_finetune_v5/best.pt \
  --outdir /path/to/output_busi_oneclick_test \
  --img 256 --num 80 \
  --support_k 2 --bank 200 --proto_m 48 --alpha 0.5 --beta 0.5
```

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
