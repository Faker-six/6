<a href=https://arxiv.org/abs/2412.15058><img src="https://img.shields.io/badge/arxiv-2312.07381-orange?logo=arxiv&logoColor=white"/></a>
<a href=https://github.com/dalcalab/SlicerMultiverSeg><img src="https://img.shields.io/badge/3D Slicer Extension-SlicerMultiverSeg-blue"/></a>


# MultiverSeg

### [Project Page](https://multiverseg.csail.mit.edu) | [Paper](https://arxiv.org/abs/2412.15058) 

Official implementation of [MultiverSeg: Scalable Interactive Segmentation of Biomedical Imaging Datasets with In-Context Guidance](https://arxiv.org/abs/2412.15058) accepted at ICCV 2025

[Hallee E. Wong](https://halleewong.github.io/), [Jose Javier Gonzalez Ortiz](https://josejg.com/), [John Guttag](https://people.csail.mit.edu/guttag/), [Adrian V. Dalca](http://www.mit.edu/~adalca/)

![img](https://github.com/halleewong/MultiverSeg/blob/website/assets/teaser.png)

## Updates

* (2025-09-24) released `v1` weights used in the ICCV paper
* (2025-08-31) ICCV camera-ready [posted on arxiv](https://arxiv.org/abs/2412.15058) with additional evaluations on TotalSegmentator and more baselines  
* (2025-07-01) 3D Slicer extension developed by Kitware Inc. released: https://github.com/dalcalab/SlicerMultiverSeg
* (2025-06-25) MultiverSeg was accepted to ICCV 2025!
* (2025-01-26) inference code and `v0` weights released
* (2024-12-19) arxiv preprint released!

## Models

We provide pre-trained weights [here](https://www.dropbox.com/scl/fo/71j9vl3d4db0u229rq689/AI_5oDICnt0HnBcry-xJSNQ?rlkey=7y42638h12ilqds8270owzric&st=3py413ys&dl=0). 

* `v0`: initial release (shorter training schedule).

* `v1`: trained for longer, with improved performance. These are the weights used in the ICCV 2025 paper.

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

### Train

```bash
python scripts/busi/multiverseg_busi_finetune_v5.py \
  --root /path/to/Dataset_BUSI_with_GT \
  --outdir /path/to/runs_busi_finetune_v5 \
  --img 256 --epochs 80 --batch 2 --workers 4
```

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
