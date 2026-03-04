#!/usr/bin/env python3
# ============================================================
# multiverseg_stageA_BUSI_BENMAL_POSONLY_1CLICK_ONLY__SUPPORTK2__FAIR_WEIGHTEDPROTO_AREAxDTMAX__REPLKSE__CROSSLAST2_REPLKSE__EMA__FULLPACK.py
#
# ✅ BUSI ultrasound
# ✅ ONLY use: benign + malignant (IGNORE normal)
# ✅ POS-only, exactly ONE click (K=1) fixed (train + val)
# ✅ Val: POS-only, exactly ONE click (K=1) only (no curve)
#
# ✅ SUPPORT K = 2  (你要的 K=2)
#    - Each step samples TWO independent weighted-average prototypes
#    - Output support is (B,2,1,H,W) for s_img/s_msk
#
# ✅ Fair support weighting (NO query): prototype weight = area^alpha * dtmax^beta
# ✅ Support bank: prefer positive masks to avoid empty-mask dilution
# ✅ Input stem: RepLK + SE on q5 (5-ch) + residual
# ✅ Cross blocks (last N): RepLK + SE refinement AFTER the cross block (residual)
# ✅ EMA eval
# ✅ BATCH>1 FIX (proto expand)
#
# ✅ BUSI structure (yours):
#   <root>/
#     benign/images/*    benign/mask/*
#     malignant/images/* malignant/mask/*
#     normal/... (ignored)
#
# ✅ Pairing FIX (BUSI masks often use *_mask):
#   image: xxx.png
#   mask : xxx_mask.png
#   This script supports:
#     - exact stem match: xxx <-> xxx
#     - suffix variants: xxx_mask / xxx-mask / xxx mask / xxxMask (normalized)
#
# Run:
#   python multiverseg_stageA_BUSI_BENMAL_POSONLY_1CLICK_ONLY__SUPPORTK2__FAIR_WEIGHTEDPROTO_AREAxDTMAX__REPLKSE__CROSSLAST2_REPLKSE__EMA__FULLPACK.py \
#     --root /home/anywhere4090/alioth_project/Dataset_BUSI_with_GT \
#     --outdir /home/anywhere4090/alioth_project/runs_busi_posonly_1click_k2_areaxdt_replkse_crosslast2_ema \
#     --img 256 --epochs 80 --batch 2 --workers 4 \
#     --bank 200 --proto_m 48 --alpha 0.5 --beta 0.5 --val_subset 200 \
#     --support_k 2 \
#     --cross_replkse 1 --cross_last_n 2 --cross_k 17
# ============================================================

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import gc
import cv2
import argparse
import random
import numpy as np
from pathlib import Path
from collections import OrderedDict
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split

# avoid unstable fused SDP kernels on some drivers
try:
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
except Exception:
    pass


# ----------------------------
# Config
# ----------------------------
class CFG:
    ROOT = r"/home/anywhere4090/alioth_project/Dataset_BUSI_with_GT"
    OUTDIR = r"/home/anywhere4090/alioth_project/runs_busi_posonly_1click_k2_areaxdt_replkse_crosslast2_ema"

    MVS_WEIGHT = r"/home/anywhere4090/MultiverSeg_v1_nf256_res128.pt"

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    IMG_SIZE = 256

    EPOCHS = 80
    LR = 2e-4
    WEIGHT_DECAY = 1e-5
    MIN_LR = 5e-6

    BATCH = 2
    VAL_BATCH = 1
    NUM_WORKERS = 4
    SEED = 42

    # AMP
    AMP = True
    AMP_DTYPE = "bf16"  # bf16/fp16

    # -------------------------
    # Support bank + FAIR weighted proto (support-only)
    # -------------------------
    SUPPORT_NUM = 200
    PROTO_M = 48
    ALPHA = 0.50
    BETA = 0.50
    STAT_EPS = 1e-6

    PROTO_CAND_MULT = 4
    PROTO_TOPM = True
    PROTO_NORM = True

    # ✅ SUPPORT K=2 (can override by --support_k)
    SUPPORT_K = 2

    # ✅ ONLY ONE CLICK
    TRAIN_K = 1
    VAL_ONLY_ONECLICK = True

    # Prompt stamping
    CLICK_STAMP_R = 4

    # Prompt scales
    PROMPT_SCALE_POS = 4.0
    PROMPT_SCALE_NEG = 4.0
    PROMPT_SCALE_PREV = 1.0
    Q5_CLAMP_MAX = 6.0

    # Ultrasound normalize (CLAHE often helps)
    USE_CLAHE = True
    CLAHE_CLIP = 2.0
    CLAHE_TILE = 8

    # Sampling weights (in case you have some empty/near-empty masks)
    POS_WEIGHT = 2.0
    NEG_WEIGHT = 1.0

    # Cache decoded images/masks
    CACHE_ITEMS = 256

    # Speed knobs
    FAST_SUBSET = 0
    SAVE_EVERY = 5

    # Val
    VAL_SUBSET = 200
    VAL_FULL_EVERY = 5

    DEBUG_SYNC = 0

    # Oracle threshold (only for constructing FN candidates)
    ORACLE_PRED_THR = 0.5
    PREV_USE_PROB = True

    # Loss mix
    LOSS_W_BCE = 0.15
    LOSS_W_DICE = 0.75
    LOSS_W_IOU  = 0.10

    # Split
    TEST_SIZE = 0.15
    VAL_SIZE_FROM_TRAIN = 0.15

    # -------------------------
    # POS-only human-sim clicking
    # -------------------------
    CLICK_MODE = "noisy"      # random/noisy
    POS_PREFER_FN = True
    POS_BOUNDARY_PROB = 0.35

    NOISE_STD = 10.0
    NOISE_CLAMP = 25
    NOISE_MAX_TRIES = 30
    MIX_RANDOM_PROB = 0.15
    BOUNDARY_BAND = 3

    # -------------------------
    # ✅ RepLK + SE input stem (q5)
    # -------------------------
    REPLKSE_ENABLE = True
    REPLK_KERNEL = 31
    REPLK_DILATION = 1
    SE_REDUCTION = 4

    # -------------------------
    # ✅ RepLK + SE on last cross blocks
    # -------------------------
    CROSS_REPLKSE_ENABLE = True
    CROSS_LAST_N = 2
    CROSS_KERNEL = 17
    CROSS_DILATION = 1
    CROSS_SE_REDUCTION = 8
    CROSS_NORM = "gn1"         # "gn1" or "ln2d" or "bn"

    # -------------------------
    # ✅ EMA
    # -------------------------
    USE_EMA = True
    EMA_DECAY = 0.999


# ----------------------------
# Utils
# ----------------------------
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def seed_worker(worker_id: int):
    base = CFG.SEED + worker_id
    random.seed(base); np.random.seed(base); torch.manual_seed(base)

def is_garbage_filename(name: str) -> bool:
    return name.startswith("._") or name.startswith(".") or name.endswith(":Zone.Identifier") or name in ["Thumbs.db", "desktop.ini"]

def _autocast():
    if (not bool(CFG.AMP)) or (CFG.DEVICE != "cuda"):
        return torch.amp.autocast(device_type="cuda", enabled=False)
    if str(CFG.AMP_DTYPE).lower() == "bf16":
        return torch.amp.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16)
    return torch.amp.autocast(device_type="cuda", enabled=True, dtype=torch.float16)

def normalize_ultrasound_gray(gray_u8: np.ndarray) -> np.ndarray:
    g = gray_u8
    if CFG.USE_CLAHE:
        clahe = cv2.createCLAHE(
            clipLimit=float(CFG.CLAHE_CLIP),
            tileGridSize=(int(CFG.CLAHE_TILE), int(CFG.CLAHE_TILE))
        )
        g = clahe.apply(g)
    x = g.astype(np.float32) / 255.0
    return np.clip(x, 0.0, 1.0).astype(np.float32)

def resize2d(img: np.ndarray, size: int, is_mask=False) -> np.ndarray:
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    return cv2.resize(img, (size, size), interpolation=interp)

def _stamp(map_t, y, x, r=3, v=1.0):
    H, W = map_t.shape[-2], map_t.shape[-1]
    y0, y1 = max(0, y - r), min(H, y + r + 1)
    x0, x1 = max(0, x - r), min(W, x + r + 1)
    map_t[..., y0:y1, x0:x1] = v

def _dt_argmax(binary_mask_uint8):
    if binary_mask_uint8.max() == 0:
        return None
    dist = cv2.distanceTransform(binary_mask_uint8, cv2.DIST_L2, 5)
    if dist.max() <= 1e-6:
        return None
    y, x = np.unravel_index(np.argmax(dist), dist.shape)
    return int(y), int(x)

def _seeded_rng_from_torch(gen: torch.Generator):
    s = int(torch.randint(0, 2**31 - 1, (1,), generator=gen).item())
    return np.random.default_rng(s)

def _mask_to_points(mask01_u8: np.ndarray):
    ys, xs = np.where(mask01_u8 > 0)
    if len(ys) == 0:
        return None
    return ys, xs

def _random_point_in_mask(mask01_u8: np.ndarray, rng: np.random.Generator):
    pts = _mask_to_points(mask01_u8)
    if pts is None:
        return None
    ys, xs = pts
    j = int(rng.integers(0, len(ys)))
    return int(ys[j]), int(xs[j])

def _boundary_band(mask_u8: np.ndarray, band: int):
    if mask_u8.max() == 0:
        return np.zeros_like(mask_u8, dtype=np.uint8), np.zeros_like(mask_u8, dtype=np.uint8)
    k = int(max(1, band))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*k+1, 2*k+1))
    dil = cv2.dilate(mask_u8, ker, iterations=1)
    ero = cv2.erode(mask_u8, ker, iterations=1)
    inside_band = (mask_u8 == 1) & (ero == 0)
    outside_band = (mask_u8 == 0) & (dil == 1)
    return inside_band.astype(np.uint8), outside_band.astype(np.uint8)

def _noisy_project_point(y, x, valid_mask_u8: np.ndarray, rng: np.random.Generator):
    H, W = valid_mask_u8.shape
    for _ in range(int(CFG.NOISE_MAX_TRIES)):
        dy = float(rng.normal(0.0, float(CFG.NOISE_STD)))
        dx = float(rng.normal(0.0, float(CFG.NOISE_STD)))
        dy = float(np.clip(dy, -float(CFG.NOISE_CLAMP), float(CFG.NOISE_CLAMP)))
        dx = float(np.clip(dx, -float(CFG.NOISE_CLAMP), float(CFG.NOISE_CLAMP)))
        yy = int(np.clip(int(round(y + dy)), 0, H-1))
        xx = int(np.clip(int(round(x + dx)), 0, W-1))
        if valid_mask_u8[yy, xx] > 0:
            return yy, xx
    return _random_point_in_mask(valid_mask_u8, rng)

def _dtmax_from_mask_u8(mask01_u8: np.ndarray) -> float:
    if mask01_u8 is None or mask01_u8.max() == 0:
        return 0.0
    dist = cv2.distanceTransform(mask01_u8.astype(np.uint8), cv2.DIST_L2, 5)
    return float(dist.max())

def _soft_iou_loss_prob(p, t, eps=1e-6):
    inter = (p * t).sum(dim=(1,2,3))
    union = (p + t - p*t).sum(dim=(1,2,3))
    iou = (inter + eps) / (union + eps)
    return (1.0 - iou).mean()


# ----------------------------
# Data listing (BUSI benign/malignant only) ✅ *_mask pairing FIX
# ----------------------------
def _find_first_existing_dir(base: Path, candidates):
    for c in candidates:
        p = base / c
        if p.exists() and p.is_dir():
            return p
    return None

def _list_files_flat(d: Path):
    out = []
    for p in sorted(d.iterdir()):
        if p.is_dir():
            continue
        if is_garbage_filename(p.name):
            continue
        out.append(p)
    return out

def _norm_key(stem: str) -> str:
    s = stem.lower().strip()
    s = s.replace(" ", "")
    s = s.replace("-", "")
    return s

def list_busi_pairs(root_dir: str):
    root = Path(root_dir)
    assert root.exists(), f"Missing root: {root}"

    groups = [("benign", 0), ("malignant", 1)]  # IGNORE normal
    pairs = []

    for gname, cid in groups:
        gdir = root / gname
        if not gdir.exists():
            raise RuntimeError(f"Missing class folder: {gdir}")

        img_dir = _find_first_existing_dir(gdir, ["images", "image", "imgs", "img"])
        msk_dir = _find_first_existing_dir(gdir, ["mask", "masks", "labels", "label"])

        if img_dir is None or msk_dir is None:
            raise RuntimeError(
                f"BUSI expects {gdir}/(images|image) and {gdir}/(mask|masks). "
                f"Got img_dir={img_dir} msk_dir={msk_dir}"
            )

        img_files = _list_files_flat(img_dir)
        msk_files = _list_files_flat(msk_dir)

        # mask map with multiple keys
        msk_map = {}
        for mp in msk_files:
            st = mp.stem
            k0 = _norm_key(st)
            msk_map[k0] = mp

            base = st
            low = st.lower().strip()
            for suf in ["_mask", "-mask", "mask"]:
                if low.endswith(suf):
                    base = st[:len(st) - len(suf)]
                    break
            k1 = _norm_key(base)
            if k1 not in msk_map:
                msk_map[k1] = mp

        miss = 0
        added = 0
        for ip in img_files:
            st = ip.stem
            k_img = _norm_key(st)
            cand_keys = [
                k_img,
                _norm_key(st + "_mask"),
                _norm_key(st + "-mask"),
                _norm_key(st + "mask"),
                _norm_key(st + " Mask"),
            ]
            mp = None
            for kk in cand_keys:
                if kk in msk_map:
                    mp = msk_map[kk]
                    break
            if mp is not None:
                pairs.append((str(ip), str(mp), int(cid)))
                added += 1
            else:
                miss += 1

        print(f"[ListBUSI] {gname}: images={len(img_files)} masks={len(msk_files)} pairs={added} miss={miss}")

    if len(pairs) == 0:
        dbg = []
        for gname, _cid in groups:
            img_dir = (root / gname / "images")
            msk_dir = (root / gname / "mask")
            if img_dir.exists():
                dbg.append(f"{gname}/images sample: " + ", ".join([p.name for p in _list_files_flat(img_dir)[:5]]))
            if msk_dir.exists():
                dbg.append(f"{gname}/mask sample: " + ", ".join([p.name for p in _list_files_flat(msk_dir)[:5]]))
        raise RuntimeError(f"No benign/malignant pairs found under: {root}\n" + "\n".join(dbg))

    return pairs

def split_pairs(pairs):
    idxs = list(range(len(pairs)))
    tr, te = train_test_split(idxs, test_size=float(CFG.TEST_SIZE), random_state=CFG.SEED, shuffle=True)
    tr2, va = train_test_split(tr, test_size=float(CFG.VAL_SIZE_FROM_TRAIN), random_state=CFG.SEED, shuffle=True)
    return tr2, va, te


# ----------------------------
# Dataset
# ----------------------------
class BUSIDatasetBenMal(Dataset):
    def __init__(self, pairs, indices, train=True, cache_items=256):
        self.pairs = list(pairs)
        self.indices = list(indices)
        self.train = bool(train)

        self.cache_items = int(max(0, cache_items))
        self._cache = OrderedDict()

        self._is_pos = {}
        for ii in self.indices:
            _imgp, mskp, _cid = self.pairs[ii]
            m = cv2.imread(mskp, cv2.IMREAD_GRAYSCALE)
            self._is_pos[ii] = 1 if (m is not None and int((m > 0).sum()) > 0) else 0

    def __len__(self):
        return len(self.indices)

    def _cache_get(self, key):
        if key not in self._cache:
            return None
        val = self._cache.pop(key)
        self._cache[key] = val
        return val

    def _cache_put(self, key, val):
        if self.cache_items <= 0:
            return
        if key in self._cache:
            self._cache.pop(key)
        self._cache[key] = val
        while len(self._cache) > self.cache_items:
            self._cache.popitem(last=False)

    def _load_pair(self, img_path, msk_path):
        key = (img_path, msk_path)
        hit = self._cache_get(key)
        if hit is not None:
            return hit

        img_any = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        msk_g = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
        if img_any is None or msk_g is None:
            raise RuntimeError(f"Bad read: img={img_path} msk={msk_path}")

        if img_any.ndim == 2:
            gray = img_any
        elif img_any.ndim == 3:
            if img_any.shape[2] == 4:
                img_any = img_any[:, :, :3]
            gray = cv2.cvtColor(img_any, cv2.COLOR_BGR2GRAY)
        else:
            raise RuntimeError(f"Unexpected image shape: {img_any.shape} for {img_path}")

        self._cache_put(key, (gray, msk_g))
        return gray, msk_g

    def __getitem__(self, i):
        idx = self.indices[i]
        img_path, msk_path, cid = self.pairs[idx]
        gray_u8, msk_u8 = self._load_pair(img_path, msk_path)

        msk = (msk_u8 > 0).astype(np.float32)

        img = normalize_ultrasound_gray(gray_u8)
        img = resize2d(img, CFG.IMG_SIZE, is_mask=False)
        msk = resize2d(msk, CFG.IMG_SIZE, is_mask=True)
        msk = (msk > 0.5).astype(np.float32)

        if self.train:
            if random.random() < 0.5:
                img = np.ascontiguousarray(np.flip(img, axis=1))
                msk = np.ascontiguousarray(np.flip(msk, axis=1))
            if random.random() < 0.2:
                img = np.ascontiguousarray(np.flip(img, axis=0))
                msk = np.ascontiguousarray(np.flip(msk, axis=0))
            if random.random() < 0.2:
                k = random.choice([3, 5])
                img = cv2.GaussianBlur(img, (k, k), 0)
            if random.random() < 0.2:
                noise = np.random.normal(0.0, 0.03, size=img.shape).astype(np.float32)
                img = np.clip(img + noise, 0.0, 1.0)

        is_pos = int(self._is_pos.get(idx, 0))
        return (
            torch.from_numpy(img).unsqueeze(0),
            torch.from_numpy(msk).unsqueeze(0),
            torch.tensor(int(cid), dtype=torch.long),
            img_path, msk_path, is_pos
        )


# ----------------------------
# Model + RepLKSE stem + Cross-last replkse injection
# ----------------------------
from multiverseg.models.sp_mvs import MultiverSegNet

class SEBlock(nn.Module):
    def __init__(self, c: int, reduction: int = 4):
        super().__init__()
        hidden = max(1, c // max(1, int(reduction)))
        self.fc1 = nn.Conv2d(c, hidden, kernel_size=1, bias=True)
        self.fc2 = nn.Conv2d(hidden, c, kernel_size=1, bias=True)

    def forward(self, x):
        s = F.adaptive_avg_pool2d(x, 1)
        s = F.silu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s

class RepLKSE2D(nn.Module):
    def __init__(self, c: int, k: int = 17, dilation: int = 1, se_reduction: int = 8, norm: str = "gn1"):
        super().__init__()
        k = int(k)
        assert k % 2 == 1, "kernel must be odd"
        pad = (k // 2) * int(dilation)

        self.dw = nn.Conv2d(c, c, kernel_size=k, padding=pad, dilation=int(dilation), groups=c, bias=False)
        self.pw = nn.Conv2d(c, c, kernel_size=1, bias=False)

        norm = str(norm).lower()
        if norm == "gn1":
            self.norm = nn.GroupNorm(num_groups=1, num_channels=c)
        elif norm == "bn":
            self.norm = nn.BatchNorm2d(c)
        elif norm == "ln2d":
            self.norm = nn.GroupNorm(num_groups=1, num_channels=c)
        else:
            self.norm = nn.GroupNorm(num_groups=1, num_channels=c)

        self.se = SEBlock(c, reduction=se_reduction)

    def forward(self, x):
        y = self.dw(x)
        y = self.pw(y)
        y = self.norm(y)
        y = F.gelu(y)
        y = self.se(y)
        return x + y

class RepLKSEStem(nn.Module):
    def __init__(self, c: int = 5, k: int = 31, dilation: int = 1, se_reduction: int = 4):
        super().__init__()
        self.block = RepLKSE2D(c=c, k=k, dilation=dilation, se_reduction=se_reduction, norm="gn1")
    def forward(self, x):
        return self.block(x)

class _CrossWrap(nn.Module):
    def __init__(self, inner: nn.Module, c_hint: int = 256):
        super().__init__()
        self.inner = inner
        self.c_hint = int(c_hint)
        self.refine = None

    def _ensure(self, x: torch.Tensor):
        if self.refine is not None:
            return
        c = int(x.shape[1]) if (x is not None and x.ndim == 4) else int(self.c_hint)
        self.refine = RepLKSE2D(
            c=c,
            k=int(CFG.CROSS_KERNEL),
            dilation=int(CFG.CROSS_DILATION),
            se_reduction=int(CFG.CROSS_SE_REDUCTION),
            norm=str(CFG.CROSS_NORM),
        ).to(x.device if x is not None else "cpu")

    def _apply_tensor(self, t):
        if isinstance(t, torch.Tensor) and t.ndim == 4:
            self._ensure(t)
            return self.refine(t)
        return t

    def forward(self, *args, **kwargs):
        out = self.inner(*args, **kwargs)
        if isinstance(out, torch.Tensor):
            return self._apply_tensor(out)
        if isinstance(out, (tuple, list)):
            new = []
            for v in out:
                new.append(self._apply_tensor(v))
            return type(out)(new)
        if isinstance(out, dict):
            new = {}
            for k, v in out.items():
                new[k] = self._apply_tensor(v)
            return new
        return out

def _inject_cross_lastN(net: nn.Module, last_n: int = 2) -> bool:
    if last_n <= 0:
        return False

    candidates = []
    for attr in ["cross_blocks", "cross_layers", "cross", "x_blocks", "cross_block"]:
        if hasattr(net, attr):
            m = getattr(net, attr)
            if isinstance(m, (nn.ModuleList, list, tuple)) and len(m) >= 1 and all(isinstance(x, nn.Module) for x in m):
                candidates.append((attr, m))

    if len(candidates) == 0:
        for name, m in net.named_modules():
            if isinstance(m, nn.ModuleList) and len(m) >= 1:
                if "cross" in name.lower():
                    candidates.append((name, m))

    if len(candidates) == 0:
        print("[Warn] CROSS injection: no cross ModuleList found (skip).")
        return False

    candidates.sort(key=lambda x: len(x[1]), reverse=True)
    name, ml = candidates[0]
    L = len(ml)
    n = min(int(last_n), L)

    print(f"[Info] CROSS injection: found '{name}' length={L} -> wrap last {n} blocks with RepLKSE(k={CFG.CROSS_KERNEL}).")
    for i in range(L - n, L):
        ml[i] = _CrossWrap(ml[i], c_hint=256)
    return True

class WrappedMVS(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.net = MultiverSegNet(
            in_channels=[5, 2],
            encoder_blocks=[256, 256, 256, 256],
            block_kws=dict(conv_kws=dict(norm="layer")),
            cross_relu=True
        ).to(device)

        self.stem = RepLKSEStem(
            c=5,
            k=int(CFG.REPLK_KERNEL),
            dilation=int(CFG.REPLK_DILATION),
            se_reduction=int(CFG.SE_REDUCTION),
        ) if bool(CFG.REPLKSE_ENABLE) else nn.Identity()

        if bool(CFG.CROSS_REPLKSE_ENABLE):
            _ = _inject_cross_lastN(self.net, last_n=int(CFG.CROSS_LAST_N))

    def forward(self, q5, s_img, s_msk):
        q5 = self.stem(q5)
        if q5.ndim == 4: q5 = q5.unsqueeze(1)
        if s_img.ndim == 4: s_img = s_img.unsqueeze(1)
        if s_msk.ndim == 4: s_msk = s_msk.unsqueeze(1)
        return self.net(q5, s_img, s_msk)


# ----------------------------
# EMA
# ----------------------------
class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {}
        self.backup = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name not in self.shadow:
                self.shadow[name] = p.detach().clone()
            self.shadow[name].mul_(d).add_(p.detach(), alpha=(1.0 - d))

    def apply_shadow(self, model: nn.Module):
        self.backup = {}
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            self.backup[name] = p.detach().clone()
            if name in self.shadow:
                p.data.copy_(self.shadow[name].data)

    def restore(self, model: nn.Module):
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name in self.backup:
                p.data.copy_(self.backup[name].data)
        self.backup = {}


# ----------------------------
# Support bank + FAIR weighted prototype (area x dtmax) ✅ SUPPORT_K=2
# ----------------------------
@torch.no_grad()
def build_support_bank(loader_for_bank):
    imgs_pos, msks_pos, areas_pos, dtmaxs_pos = [], [], [], []
    imgs_any, msks_any, areas_any, dtmaxs_any = [], [], [], []

    for img, msk, _cid, *_meta in loader_for_bank:
        img16 = img.to(dtype=torch.float16, device="cpu")
        msk16 = msk.to(dtype=torch.float16, device="cpu")
        m = (msk16[0,0].detach().cpu().numpy() > 0.5).astype(np.uint8)
        area = float(m.sum())
        dtm = _dtmax_from_mask_u8(m)

        imgs_any.append(img16); msks_any.append(msk16)
        areas_any.append(area); dtmaxs_any.append(dtm)

        if area > 0:
            imgs_pos.append(img16); msks_pos.append(msk16)
            areas_pos.append(area); dtmaxs_pos.append(dtm)

        if len(imgs_pos) >= int(CFG.SUPPORT_NUM):
            break
        if len(imgs_any) >= int(CFG.SUPPORT_NUM) * 10:
            break

    if len(imgs_pos) > 0:
        imgs = imgs_pos[:int(CFG.SUPPORT_NUM)]
        msks = msks_pos[:int(CFG.SUPPORT_NUM)]
        areas = areas_pos[:int(CFG.SUPPORT_NUM)]
        dtmaxs = dtmaxs_pos[:int(CFG.SUPPORT_NUM)]
    else:
        imgs = imgs_any[:max(1, int(CFG.SUPPORT_NUM))]
        msks = msks_any[:max(1, int(CFG.SUPPORT_NUM))]
        areas = areas_any[:max(1, int(CFG.SUPPORT_NUM))]
        dtmaxs = dtmaxs_any[:max(1, int(CFG.SUPPORT_NUM))]

    return {
        "imgs": torch.cat(imgs, dim=0).contiguous(),   # (N,1,H,W)
        "msks": torch.cat(msks, dim=0).contiguous(),   # (N,1,H,W)
        "areas": torch.tensor(areas, dtype=torch.float32, device="cpu").contiguous(),
        "dtmaxs": torch.tensor(dtmaxs, dtype=torch.float32, device="cpu").contiguous()
    }

@torch.no_grad()
def sample_weighted_prototype(bank, device, B: int, gen: torch.Generator):
    """
    Return:
      proto_img: (B, K, 1, H, W)
      proto_msk: (B, K, 1, H, W)
    where each k is an independent weighted-average prototype.
    """
    imgs = bank["imgs"]
    msks = bank["msks"]
    areas = bank["areas"]
    dtmaxs = bank["dtmaxs"]

    N = int(imgs.shape[0])
    M = int(min(max(1, CFG.PROTO_M), N))
    cand = int(min(N, max(M, M * int(CFG.PROTO_CAND_MULT))))
    K = int(max(1, getattr(CFG, "SUPPORT_K", 1)))

    proto_imgs = []
    proto_msks = []

    for _ in range(K):
        if N <= 1:
            idx_cand = torch.zeros((cand,), dtype=torch.long)
        else:
            idx_cand = torch.randint(0, N, (cand,), generator=gen)

        a = areas.index_select(0, idx_cand)
        d = dtmaxs.index_select(0, idx_cand)

        w_cand = (a + float(CFG.STAT_EPS)).pow(float(CFG.ALPHA)) * (d + float(CFG.STAT_EPS)).pow(float(CFG.BETA))

        if bool(CFG.PROTO_TOPM) and cand > M:
            topv, topi = torch.topk(w_cand, k=M, largest=True)
            idx = idx_cand.index_select(0, topi)
            w = topv
        else:
            idx = idx_cand[:M]
            w = w_cand[:M]

        if bool(CFG.PROTO_NORM):
            w = w / (w.sum() + 1e-12)

        w = w.to(dtype=torch.float32, device="cpu")
        sup_img = imgs.index_select(0, idx).to(dtype=torch.float32, device="cpu")  # (M,1,H,W)
        sup_msk = msks.index_select(0, idx).to(dtype=torch.float32, device="cpu")  # (M,1,H,W)

        w_view = w.view(M, 1, 1, 1)
        proto_img = (sup_img * w_view).sum(dim=0, keepdim=True)  # (1,1,H,W)
        proto_msk = (sup_msk * w_view).sum(dim=0, keepdim=True)  # (1,1,H,W)

        proto_imgs.append(proto_img)
        proto_msks.append(proto_msk)

    proto_img = torch.cat(proto_imgs, dim=0)  # (K,1,H,W)
    proto_msk = torch.cat(proto_msks, dim=0)  # (K,1,H,W)

    dtype = torch.float16 if (CFG.DEVICE == "cuda" and CFG.AMP) else torch.float32
    proto_img = proto_img.to(device=device, dtype=dtype, non_blocking=True)
    proto_msk = proto_msk.to(device=device, dtype=dtype, non_blocking=True)

    # -> (1,K,1,H,W) -> expand to (B,K,1,H,W)
    proto_img = proto_img.unsqueeze(0).expand(int(B), -1, -1, -1, -1).contiguous()
    proto_msk = proto_msk.unsqueeze(0).expand(int(B), -1, -1, -1, -1).contiguous()

    return proto_img, proto_msk


# ----------------------------
# Prompts / POS-only One-click human-sim
# ----------------------------
@torch.no_grad()
def build_q5(imgs, pos01, neg01, prev01, device):
    B, _, H, W = imgs.shape
    box01 = torch.zeros((B,1,H,W), device=device, dtype=torch.float32)
    pos  = pos01.clamp(0,1) * float(CFG.PROMPT_SCALE_POS)
    neg  = neg01.clamp(0,1) * float(CFG.PROMPT_SCALE_NEG)
    prev = prev01.clamp(0,1) * float(CFG.PROMPT_SCALE_PREV)
    q5 = torch.cat([imgs, pos, neg, box01, prev], dim=1)
    q5 = torch.nan_to_num(q5, nan=0.0, posinf=float(CFG.Q5_CLAMP_MAX), neginf=0.0).clamp(0.0, float(CFG.Q5_CLAMP_MAX))
    return q5

@torch.no_grad()
def _choose_pos_point_humansim(g_u8, fn_u8, rng: np.random.Generator):
    if int(g_u8.sum()) == 0:
        return None
    inside_band, _ = _boundary_band(g_u8, band=int(CFG.BOUNDARY_BAND))

    cand = fn_u8.copy() if (bool(CFG.POS_PREFER_FN) and int(fn_u8.sum()) > 0) else g_u8.copy()

    if float(CFG.POS_BOUNDARY_PROB) > 0 and rng.random() < float(CFG.POS_BOUNDARY_PROB) and int(inside_band.sum()) > 0:
        cand = inside_band.copy()

    if int(cand.sum()) == 0:
        cand = g_u8.copy()

    if str(CFG.CLICK_MODE).lower() == "noisy" and (rng.random() >= float(CFG.MIX_RANDOM_PROB)):
        base = _dt_argmax(cand.astype(np.uint8))
        if base is None:
            base = _random_point_in_mask(cand.astype(np.uint8), rng)
        if base is None:
            return None
        return _noisy_project_point(base[0], base[1], g_u8.astype(np.uint8), rng)
    else:
        return _random_point_in_mask(cand.astype(np.uint8), rng)

@torch.no_grad()
def posonly_oneclick_teacher(seg, imgs, msks, s_img, s_msk, device, gen: torch.Generator):
    B, _, H, W = imgs.shape
    pos01 = torch.zeros((B,1,H,W), device=device)
    neg01 = torch.zeros((B,1,H,W), device=device)
    prev  = torch.zeros((B,1,H,W), device=device)

    q5_0 = build_q5(imgs, pos01, neg01, prev, device)
    pred = torch.sigmoid(seg(q5_0, s_img, s_msk)).float()

    rng = _seeded_rng_from_torch(gen)
    pred_bin = (pred > float(CFG.ORACLE_PRED_THR)).float()
    gt_bin   = (msks > 0.5).float()

    for b in range(B):
        g = (gt_bin[b,0].detach().cpu().numpy() > 0.5).astype(np.uint8)
        p = (pred_bin[b,0].detach().cpu().numpy() > 0.5).astype(np.uint8)
        if int(g.sum()) > 0:
            fn = ((g==1) & (p==0)).astype(np.uint8)
            yx = _choose_pos_point_humansim(g, fn, rng)
            if yx is not None:
                _stamp(pos01[b,0], yx[0], yx[1], r=int(CFG.CLICK_STAMP_R), v=1.0)

    prev_in = pred if CFG.PREV_USE_PROB else pred_bin
    return pos01, neg01, prev_in.clone()

@torch.no_grad()
def posonly_oneclick_predict(seg, imgs, msks, s_img, s_msk, device, gen: torch.Generator):
    B, _, H, W = imgs.shape
    pos01 = torch.zeros((B,1,H,W), device=device)
    neg01 = torch.zeros((B,1,H,W), device=device)
    prev  = torch.zeros((B,1,H,W), device=device)

    q5_0 = build_q5(imgs, pos01, neg01, prev, device)
    pred0 = torch.sigmoid(seg(q5_0, s_img, s_msk)).float()

    rng = _seeded_rng_from_torch(gen)
    pred_bin = (pred0 > float(CFG.ORACLE_PRED_THR)).float()
    gt_bin   = (msks > 0.5).float()

    for b in range(B):
        g = (gt_bin[b,0].detach().cpu().numpy() > 0.5).astype(np.uint8)
        p = (pred_bin[b,0].detach().cpu().numpy() > 0.5).astype(np.uint8)
        if int(g.sum()) > 0:
            fn = ((g==1) & (p==0)).astype(np.uint8)
            yx = _choose_pos_point_humansim(g, fn, rng)
            if yx is not None:
                _stamp(pos01[b,0], yx[0], yx[1], r=int(CFG.CLICK_STAMP_R), v=1.0)

    prev_in = pred0 if CFG.PREV_USE_PROB else pred_bin
    q5_1 = build_q5(imgs, pos01, neg01, prev_in, device)
    pred1 = torch.sigmoid(seg(q5_1, s_img, s_msk)).float()
    return pred1


# ----------------------------
# Loss / metric
# ----------------------------
@torch.no_grad()
def dice_metric_tensor(p01, t01, eps=1e-8):
    p = (p01 > 0.5).float()
    t = (t01 > 0.5).float()
    inter = (p*t).sum(dim=(1,2,3))
    den = p.sum(dim=(1,2,3)) + t.sum(dim=(1,2,3))
    dice = (2.0*inter + eps) / (den + eps)
    both_empty = (den < 0.5)
    return torch.where(both_empty, torch.ones_like(dice), dice)

def dice_loss_prob(p, t, eps=1e-6):
    num = 2.0 * (p*t).sum(dim=(1,2,3))
    den = (p+t).sum(dim=(1,2,3)) + eps
    return (1.0 - (num/den)).mean()

@torch.no_grad()
def eval_posonly_oneclick_dice(seg, val_loader, bank, device, val_subset=0):
    seg.eval()
    vals = []
    gen_support = torch.Generator(device="cpu").manual_seed(12345)

    for i, (imgs, msks, _cids, *_meta) in enumerate(val_loader):
        if val_subset and i >= int(val_subset):
            break
        imgs = imgs.to(device, non_blocking=True)
        msks = msks.to(device, non_blocking=True)

        s_img, s_msk = sample_weighted_prototype(bank, device=device, B=int(imgs.shape[0]), gen=gen_support)

        gen_step = torch.Generator(device="cpu").manual_seed(777777 + i * 100 + 1)
        pred1 = posonly_oneclick_predict(seg, imgs, msks, s_img, s_msk, device=device, gen=gen_step)
        vals.append(float(dice_metric_tensor(pred1, msks).mean().item()))

    return float(np.mean(vals)) if len(vals) else 0.0


# ----------------------------
# Train
# ----------------------------
def run():
    set_seed(CFG.SEED)
    device = torch.device(CFG.DEVICE)
    os.makedirs(CFG.OUTDIR, exist_ok=True)

    if CFG.DEVICE == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    print(f"[Info] Device={CFG.DEVICE} AMP={CFG.AMP} dtype={CFG.AMP_DTYPE} IMG={CFG.IMG_SIZE}")
    print(f"[Info] BUSI BEN/MAL ONLY (ignore normal) | POS-ONLY ONE-CLICK: TrainK=1 ValK=1")
    print(f"[Info] Support FAIR proto: bank={CFG.SUPPORT_NUM} proto_m={CFG.PROTO_M} alpha={CFG.ALPHA} beta={CFG.BETA} cand_mult={CFG.PROTO_CAND_MULT} SUPPORT_K={CFG.SUPPORT_K}")
    print(f"[Info] InputStem: RepLKSE={CFG.REPLKSE_ENABLE} k={CFG.REPLK_KERNEL} se_red={CFG.SE_REDUCTION}")
    print(f"[Info] CrossLast: enable={CFG.CROSS_REPLKSE_ENABLE} last_n={CFG.CROSS_LAST_N} k={CFG.CROSS_KERNEL} se_red={CFG.CROSS_SE_REDUCTION}")
    print(f"[Info] CLICK_MODE={CFG.CLICK_MODE} noise_std={CFG.NOISE_STD} mix_random={CFG.MIX_RANDOM_PROB}")
    print(f"[Info] EMA={CFG.USE_EMA} decay={CFG.EMA_DECAY}")

    pairs = list_busi_pairs(CFG.ROOT)
    tr_idx, val_idx, te_idx = split_pairs(pairs)
    print(f"[Info] pairs={len(pairs)} | train={len(tr_idx)} val={len(val_idx)} test={len(te_idx)}")

    tr_ds = BUSIDatasetBenMal(pairs, tr_idx, train=True,  cache_items=int(CFG.CACHE_ITEMS))
    val_ds = BUSIDatasetBenMal(pairs, val_idx, train=False, cache_items=int(CFG.CACHE_ITEMS))

    weights = []
    pos_count = 0
    for ii in tr_idx:
        is_pos = int(tr_ds._is_pos.get(ii, 0))
        pos_count += int(is_pos)
        weights.append(float(CFG.POS_WEIGHT) if is_pos == 1 else float(CFG.NEG_WEIGHT))
    print(f"[Sanity] Train positive-mask imgs={pos_count}/{len(tr_idx)}")

    gen_sampler = torch.Generator(device="cpu").manual_seed(CFG.SEED)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True, generator=gen_sampler)

    tr_loader = DataLoader(
        tr_ds,
        batch_size=CFG.BATCH,
        sampler=sampler,
        num_workers=CFG.NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=4 if CFG.NUM_WORKERS > 0 else None,
        persistent_workers=True if CFG.NUM_WORKERS > 0 else False,
        worker_init_fn=seed_worker if CFG.NUM_WORKERS > 0 else None,
        drop_last=False
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=CFG.VAL_BATCH,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # support bank
    bank_ds = BUSIDatasetBenMal(pairs, tr_idx, train=False, cache_items=0)
    bank_loader = DataLoader(bank_ds, batch_size=1, shuffle=True, num_workers=0)
    bank = build_support_bank(bank_loader)
    print(f"[Info] Support bank built: N={bank['imgs'].shape[0]} | proto_m={min(CFG.PROTO_M, int(bank['imgs'].shape[0]))} | SUPPORT_K={CFG.SUPPORT_K}")
    print(f"[Info] Bank stats: area_mean={float(bank['areas'].mean().item()):.1f} dtmax_mean={float(bank['dtmaxs'].mean().item()):.2f}")

    seg = WrappedMVS(device).to(device)

    if os.path.exists(CFG.MVS_WEIGHT):
        state = torch.load(CFG.MVS_WEIGHT, map_location=device)
        if isinstance(state, dict) and ("model" in state):
            seg.net.load_state_dict(state["model"], strict=True)
        elif isinstance(state, dict) and ("state_dict" in state):
            seg.net.load_state_dict(state["state_dict"], strict=False)
        else:
            seg.net.load_state_dict(state, strict=False)
        print("[Info] Pretrained weights loaded.")
    else:
        print("[Warn] MVS_WEIGHT not found. Training from init.")

    ema = EMA(seg, decay=float(CFG.EMA_DECAY)) if bool(CFG.USE_EMA) else None

    opt = torch.optim.AdamW(seg.parameters(), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG.EPOCHS, eta_min=CFG.MIN_LR)

    use_scaler = (CFG.DEVICE == "cuda") and bool(CFG.AMP) and (str(CFG.AMP_DTYPE).lower() == "fp16")
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    best_fast = -1.0
    gen_support_train = torch.Generator(device="cpu").manual_seed(2026)

    for epoch in range(1, CFG.EPOCHS + 1):
        print(f"[Epoch {epoch}] Train BUSI(BEN/MAL) POS-only ONE click | SUPPORT_K={CFG.SUPPORT_K} FAIR proto | Input RepLKSE | CrossLast RepLKSE | EMA")

        seg.train()
        tloss = 0.0

        if CFG.DEVICE == "cuda":
            torch.cuda.synchronize()
            mem = torch.cuda.memory_allocated() / (1024**2)
            rsv = torch.cuda.memory_reserved() / (1024**2)
            print(f"[GPU] alloc={mem:.0f}MiB reserved={rsv:.0f}MiB")

        pbar = tqdm(tr_loader, desc=f"Epoch {epoch} [Train]")

        for step, batch in enumerate(pbar):
            if CFG.FAST_SUBSET and step >= int(CFG.FAST_SUBSET):
                break

            imgs, msks, _cids, img_path, msk_path, is_pos = batch
            imgs = imgs.to(device, non_blocking=True)
            msks = msks.to(device, non_blocking=True)

            s_img, s_msk = sample_weighted_prototype(bank, device=device, B=int(imgs.shape[0]), gen=gen_support_train)

            opt.zero_grad(set_to_none=True)

            with torch.no_grad():
                with _autocast():
                    gen_click = torch.Generator(device="cpu").manual_seed(333333 + epoch * 100000 + step)
                    pos01, neg01, prev01 = posonly_oneclick_teacher(seg, imgs, msks, s_img, s_msk, device=device, gen=gen_click)

            with _autocast():
                q5 = build_q5(imgs, pos01, neg01, prev01, device)
                logits = seg(q5, s_img, s_msk)

                logits_f = logits.float()
                msks_f = msks.float()
                prob_f = torch.sigmoid(logits_f)

                loss_bce = F.binary_cross_entropy_with_logits(logits_f, msks_f)
                loss_dice = dice_loss_prob(prob_f, msks_f)
                loss_iou  = _soft_iou_loss_prob(prob_f, msks_f)
                loss = float(CFG.LOSS_W_BCE) * loss_bce + float(CFG.LOSS_W_DICE) * loss_dice + float(CFG.LOSS_W_IOU) * loss_iou

            if not torch.isfinite(loss).all():
                print(f"\n[NaNGuard] skip | img={img_path[0]}")
                continue

            if use_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(seg.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(seg.parameters(), 1.0)
                opt.step()

            if ema is not None:
                ema.update(seg)

            if int(CFG.DEBUG_SYNC) == 1 and CFG.DEVICE == "cuda":
                torch.cuda.synchronize()

            tloss += float(loss.item())
            pbar.set_postfix(loss=f"{(tloss / max(1, step+1)):.4f}", clickK=1, supportK=CFG.SUPPORT_K)

        tloss /= max(1, (step+1))
        sched.step()

        torch.cuda.empty_cache()
        gc.collect()

        if ema is not None:
            ema.apply_shadow(seg)
        try:
            with torch.no_grad():
                with _autocast():
                    d1_fast = eval_posonly_oneclick_dice(seg, val_loader, bank, device, val_subset=int(CFG.VAL_SUBSET))
        finally:
            if ema is not None:
                ema.restore(seg)

        if d1_fast > best_fast:
            best_fast = d1_fast
            torch.save(
                {"model": seg.state_dict(), "epoch": epoch, "best_fast@1": float(best_fast),
                 "cfg": {k: getattr(CFG, k) for k in dir(CFG) if k.isupper()}},
                os.path.join(CFG.OUTDIR, "best.pt")
            )

        if epoch % int(CFG.SAVE_EVERY) == 0:
            torch.save({"model": seg.state_dict(), "epoch": epoch, "best_fast@1": float(best_fast)},
                       os.path.join(CFG.OUTDIR, f"epoch_{epoch}.pt"))

        if int(CFG.VAL_FULL_EVERY) > 0 and (epoch % int(CFG.VAL_FULL_EVERY) == 0):
            if ema is not None:
                ema.apply_shadow(seg)
            try:
                with torch.no_grad():
                    with _autocast():
                        d1_full = eval_posonly_oneclick_dice(seg, val_loader, bank, device, val_subset=0)
            finally:
                if ema is not None:
                    ema.restore(seg)

            print(f"[E{epoch:03d}] TrainLoss={tloss:.4f} | FAST ValDice@1={d1_fast:.4f} | FULL ValDice@1={d1_full:.4f} | BestFAST@1={best_fast:.4f}")
        else:
            print(f"[E{epoch:03d}] TrainLoss={tloss:.4f} | FAST ValDice@1={d1_fast:.4f} | BestFAST@1={best_fast:.4f}")

        torch.cuda.empty_cache()
        gc.collect()


# ----------------------------
# Args
# ----------------------------
def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--root", type=str, default=None)
    ap.add_argument("--outdir", type=str, default=None)
    ap.add_argument("--img", type=int, default=None)

    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--wd", type=float, default=None)
    ap.add_argument("--min_lr", type=float, default=None)

    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--val_batch", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)

    ap.add_argument("--amp", type=int, default=None)
    ap.add_argument("--amp_dtype", type=str, default=None)

    ap.add_argument("--bank", type=int, default=None)
    ap.add_argument("--proto_m", type=int, default=None)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--beta", type=float, default=None)
    ap.add_argument("--cand_mult", type=int, default=None)

    ap.add_argument("--support_k", type=int, default=None)

    ap.add_argument("--cache_items", type=int, default=None)
    ap.add_argument("--fast_subset", type=int, default=None)
    ap.add_argument("--val_subset", type=int, default=None)
    ap.add_argument("--val_full_every", type=int, default=None)
    ap.add_argument("--save_every", type=int, default=None)
    ap.add_argument("--debug_sync", type=int, default=None)

    ap.add_argument("--clahe", type=int, default=None)

    # pos-only human-sim knobs
    ap.add_argument("--click_mode", type=str, default=None)
    ap.add_argument("--noise_std", type=float, default=None)
    ap.add_argument("--mix_random", type=float, default=None)
    ap.add_argument("--pos_boundary_p", type=float, default=None)

    # input replkse
    ap.add_argument("--replkse", type=int, default=None)
    ap.add_argument("--replk_k", type=int, default=None)
    ap.add_argument("--se_red", type=int, default=None)

    # cross replkse
    ap.add_argument("--cross_replkse", type=int, default=None)
    ap.add_argument("--cross_last_n", type=int, default=None)
    ap.add_argument("--cross_k", type=int, default=None)
    ap.add_argument("--cross_se_red", type=int, default=None)

    # ema
    ap.add_argument("--ema", type=int, default=None)
    ap.add_argument("--ema_decay", type=float, default=None)

    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.root is not None: CFG.ROOT = str(args.root)
    if args.outdir is not None: CFG.OUTDIR = str(args.outdir)
    if args.img is not None: CFG.IMG_SIZE = int(args.img)

    if args.epochs is not None: CFG.EPOCHS = int(args.epochs)
    if args.lr is not None: CFG.LR = float(args.lr)
    if args.wd is not None: CFG.WEIGHT_DECAY = float(args.wd)
    if args.min_lr is not None: CFG.MIN_LR = float(args.min_lr)

    if args.batch is not None: CFG.BATCH = int(args.batch)
    if args.val_batch is not None: CFG.VAL_BATCH = int(args.val_batch)
    if args.workers is not None: CFG.NUM_WORKERS = int(args.workers)
    if args.seed is not None: CFG.SEED = int(args.seed)

    if args.amp is not None: CFG.AMP = bool(int(args.amp))
    if args.amp_dtype is not None: CFG.AMP_DTYPE = str(args.amp_dtype)

    if args.bank is not None: CFG.SUPPORT_NUM = int(args.bank)
    if args.proto_m is not None: CFG.PROTO_M = int(args.proto_m)
    if args.alpha is not None: CFG.ALPHA = float(args.alpha)
    if args.beta is not None: CFG.BETA = float(args.beta)
    if args.cand_mult is not None: CFG.PROTO_CAND_MULT = int(args.cand_mult)

    if args.support_k is not None: CFG.SUPPORT_K = int(args.support_k)

    if args.cache_items is not None: CFG.CACHE_ITEMS = int(args.cache_items)
    if args.fast_subset is not None: CFG.FAST_SUBSET = int(args.fast_subset)
    if args.val_subset is not None: CFG.VAL_SUBSET = int(args.val_subset)
    if args.val_full_every is not None: CFG.VAL_FULL_EVERY = int(args.val_full_every)
    if args.save_every is not None: CFG.SAVE_EVERY = int(args.save_every)
    if args.debug_sync is not None: CFG.DEBUG_SYNC = int(args.debug_sync)

    if args.clahe is not None: CFG.USE_CLAHE = bool(int(args.clahe))

    if args.click_mode is not None: CFG.CLICK_MODE = str(args.click_mode)
    if args.noise_std is not None: CFG.NOISE_STD = float(args.noise_std)
    if args.mix_random is not None: CFG.MIX_RANDOM_PROB = float(args.mix_random)
    if args.pos_boundary_p is not None: CFG.POS_BOUNDARY_PROB = float(args.pos_boundary_p)

    if args.replkse is not None: CFG.REPLKSE_ENABLE = bool(int(args.replkse))
    if args.replk_k is not None: CFG.REPLK_KERNEL = int(args.replk_k)
    if args.se_red is not None: CFG.SE_REDUCTION = int(args.se_red)

    if args.cross_replkse is not None: CFG.CROSS_REPLKSE_ENABLE = bool(int(args.cross_replkse))
    if args.cross_last_n is not None: CFG.CROSS_LAST_N = int(args.cross_last_n)
    if args.cross_k is not None: CFG.CROSS_KERNEL = int(args.cross_k)
    if args.cross_se_red is not None: CFG.CROSS_SE_REDUCTION = int(args.cross_se_red)

    if args.ema is not None: CFG.USE_EMA = bool(int(args.ema))
    if args.ema_decay is not None: CFG.EMA_DECAY = float(args.ema_decay)

    os.makedirs(CFG.OUTDIR, exist_ok=True)
    run()