> **Note:** This repository is **based on** the official **MultiverSeg** implementation and paper/codebase: https://github.com/halleewong/MultiverSeg

## Models

We provide pre-trained weights [here](https://www.dropbox.com/scl/fo/71j9vl3d4db0u229rq689/AI_5oDICnt0HnBcry-xJSNQ?rlkey=7y42638h12ilqds8270owzric&st=3py413ys&dl=0). 

* `v0`: initial release (shorter training schedule).

* `v1`: trained for longer, with improved performance.

## Installation

You can install `multiverseg` in two ways:

* **With pip**:

```
pip install git+https://github.com/halleewong/MultiverSeg.git
```

* **Manually**: cloning it and installing dependencies
```
git clone https://github.com/halleewong/MultiverSeg
python -m pip install -r ./MultiverSeg/requirements.txt
export PYTHONPATH="$PYTHONPATH:$(realpath ./MultiverSeg)"
```

## Getting Started

First download the model checkpoints 
```
cd checkpoints
./download.sh
```

Then see `./notebooks/inference.ipynb` for a tutorial. 

# Acknowledgements

This project builds extensively on code originally developed for [ScribblePrompt](https://github.com/halleewong/ScribblePrompt) and [UniverSeg](https://github.com/JJGO/UniverSeg)

# Citation

If you find our work or any of our materials useful, please cite our paper:
```
@article{wong2025multiverseg,
  title={MultiverSeg: Scalable Interactive Segmentation of Biomedical Imaging Datasets with In-Context Guidance},
  author={Hallee E. Wong and Jose Javier Gonzalez Ortiz and John Guttag and Adrian V. Dalca},
  journal={International Conference on Computer Vision},
  year={2025},
}
```
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
