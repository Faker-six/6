> **Note:** This repository is **based on** the official **MultiverSeg** implementation and paper/codebase: https://github.com/halleewong/MultiverSeg


## How to run (BUSI)

### Train (finetune from official pretrained weights)

This training script **finetunes from the official MultiverSeg pretrained checkpoint**.  
Before training, download the official pretrained weights:

```bash
cd checkpoints
./download.sh

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
