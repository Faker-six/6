#!/usr/bin/env python3
# ============================================================
# test_busi_oneclick_from_trainpack.py
#
# Test script matching the training protocol of:
# multiverseg_stageA_BUSI_BENMAL_POSONLY_1CLICK_ONLY__SUPPORTK2__FAIR_WEIGHTEDPROTO_AREAxDTMAX__REPLKSE__CROSSLAST2_REPLKSE__EMA__FULLPACK.py
#
# ✅ BUSI ultrasound
# ✅ ONLY use: benign + malignant (IGNORE normal)
# ✅ POS-only, exactly ONE click (K=1) at test
# ✅ SUPPORT K = 2 (two independent weighted-average prototypes)
# ✅ Fair support weighting (NO query): area^alpha * dtmax^beta
# ✅ RepLK + SE stem on q5, and RepLKSE injection on last N cross blocks
# ✅ Save visualizations with click point
#
# Run example:
#   python test_busi_oneclick_from_trainpack.py \
#     --root /home/anywhere4090/alioth_project/Dataset_BUSI_with_GT \
#     --ckpt /home/anywhere4090/alioth_project/runs_xxx/best.pt \
#     --outdir /home/anywhere4090/alioth_project/output_busi_test \
#     --img 256 --num 80 --support_k 2 --bank 200 --proto_m 48 --alpha 0.5 --beta 0.5
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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# avoid unstable fused SDP kernels on some drivers
try:
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
except Exception:
    pass


# ----------------------------
# Default Config (override by args)
# ----------------------------
class CFG:
    ROOT = r"/home/anywhere4090/alioth_project/Dataset_BUSI_with_GT"
    OUTDIR = r"/home/anywhere4090/alioth_project/output_busi_test"
    CKPT = r"/home/anywhere4090/alioth_project/runs_busi_posonly_1click_k2_areaxdt_replkse_crosslast2_ema/best.pt"

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    IMG_SIZE = 256
    SEED = 42

    # AMP (bf16/fp16)
    AMP = True
    AMP_DTYPE = "bf16"

    # Support bank + FAIR weighted proto (support-only)
    SUPPORT_NUM = 200
    PROTO_M = 48
    ALPHA = 0.50
    BETA = 0.50
    STAT_EPS = 1e-6
    PROTO_CAND_MULT = 4
    PROTO_TOPM = True
    PROTO_NORM = True
    SUPPORT_K = 2

    # 1-click protocol
    ORACLE_PRED_THR = 0.5
    PREV_USE_PROB = True
    CLICK_STAMP_R = 4
    PROMPT_SCALE_POS = 4.0
    PROMPT_SCALE_NEG = 4.0
    PROMPT_SCALE_PREV = 1.0
    Q5_CLAMP_MAX = 6.0

    # Ultrasound normalize
    USE_CLAHE = True
    CLAHE_CLIP = 2.0
    CLAHE_TILE = 8

    # POS-only human-sim clicking
    CLICK_MODE = "noisy"   # noisy/random
    POS_PREFER_FN = True
    POS_BOUNDARY_PROB = 0.35
    NOISE_STD = 10.0
    NOISE_CLAMP = 25
    NOISE_MAX_TRIES = 30
    MIX_RANDOM_PROB = 0.15
    BOUNDARY_BAND = 3

    # RepLK + SE stem (q5)
    REPLKSE_ENABLE = True
    REPLK_KERNEL = 31
    REPLK_DILATION = 1
    SE_REDUCTION = 4

    # RepLK + SE on last cross blocks
    CROSS_REPLKSE_ENABLE = True
    CROSS_LAST_N = 2
    CROSS_KERNEL = 17
    CROSS_DILATION = 1
    CROSS_SE_REDUCTION = 8
    CROSS_NORM = "gn1"      # gn1/ln2d/bn

    # Split (same as train pack)
    TEST_SIZE = 0.15
    VAL_SIZE_FROM_TRAIN = 0.15

    # Test controls
    NUM_SAMPLES = 50          # how many images to run from test split
    SAVE_EVERY = 1            # save every N samples
    PICK_NONEMPTY = True      # prefer images with GT mask > 0


# ----------------------------
# Utils
# ----------------------------
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

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

@torch.no_grad()
def dice_metric_tensor(p01, t01, eps=1e-8):
    p = (p01 > 0.5).float()
    t = (t01 > 0.5).float()
    inter = (p*t).sum(dim=(1,2,3))
    den = p.sum(dim=(1,2,3)) + t.sum(dim=(1,2,3))
    dice = (2.0*inter + eps) / (den + eps)
    both_empty = (den < 0.5)
    return torch.where(both_empty, torch.ones_like(dice), dice)

@torch.no_grad()
def iou_metric_tensor(p01, t01, eps=1e-8):
    p = (p01 > 0.5).float()
    t = (t01 > 0.5).float()
    inter = (p*t).sum(dim=(1,2,3))
    union = (p + t - p*t).sum(dim=(1,2,3))
    iou = (inter + eps) / (union + eps)
    both_empty = (union < 0.5)
    return torch.where(both_empty, torch.ones_like(iou), iou)


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
        raise RuntimeError(f"No benign/malignant pairs found under: {root}")

    return pairs

def split_pairs(pairs):
    idxs = list(range(len(pairs)))
    tr, te = train_test_split(idxs, test_size=float(CFG.TEST_SIZE), random_state=CFG.SEED, shuffle=True)
    tr2, va = train_test_split(tr, test_size=float(CFG.VAL_SIZE_FROM_TRAIN), random_state=CFG.SEED, shuffle=True)
    return tr2, va, te


# ----------------------------
# Dataset (no aug in test)
# ----------------------------
class BUSIDatasetBenMal(Dataset):
    def __init__(self, pairs, indices, train=False, cache_items=256):
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

        is_pos = int(self._is_pos.get(idx, 0))
        return (
            torch.from_numpy(img).unsqueeze(0),
            torch.from_numpy(msk).unsqueeze(0),
            torch.tensor(int(cid), dtype=torch.long),
            img_path, msk_path, is_pos
        )


# ----------------------------
# Model + RepLKSE stem + Cross-last injection
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
            return type(out)([self._apply_tensor(v) for v in out])
        if isinstance(out, dict):
            return {k: self._apply_tensor(v) for k, v in out.items()}
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
            if isinstance(m, nn.ModuleList) and len(m) >= 1 and ("cross" in name.lower()):
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
            _inject_cross_lastN(self.net, last_n=int(CFG.CROSS_LAST_N))

    def forward(self, q5, s_img, s_msk):
        q5 = self.stem(q5)
        if q5.ndim == 4: q5 = q5.unsqueeze(1)
        if s_img.ndim == 4: s_img = s_img.unsqueeze(1)
        if s_msk.ndim == 4: s_msk = s_msk.unsqueeze(1)
        return self.net(q5, s_img, s_msk)


# ----------------------------
# Support bank + weighted prototypes (K=2)
# ----------------------------
@torch.no_grad()
def build_support_bank(loader_for_bank):
    imgs_pos, msks_pos, areas_pos, dtmaxs_pos = [], [], [], []
    imgs_any, msks_any, areas_any, dtmaxs_any = [], [], [], []

    for img, msk, _cid, *_meta in loader_for_bank:
        img16 = img.to(dtype=torch.float16, device="cpu")
        msk16 = msk.to(dtype=torch.float16, device="cpu")
        m = (msk16[0, 0].detach().cpu().numpy() > 0.5).astype(np.uint8)
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
    imgs = bank["imgs"]
    msks = bank["msks"]
    areas = bank["areas"]
    dtmaxs = bank["dtmaxs"]

    N = int(imgs.shape[0])
    M = int(min(max(1, CFG.PROTO_M), N))
    cand = int(min(N, max(M, M * int(CFG.PROTO_CAND_MULT))))
    K = int(max(1, int(CFG.SUPPORT_K)))

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

    proto_img = proto_img.unsqueeze(0).expand(int(B), -1, -1, -1, -1).contiguous()
    proto_msk = proto_msk.unsqueeze(0).expand(int(B), -1, -1, -1, -1).contiguous()
    return proto_img, proto_msk


# ----------------------------
# Prompts / POS-only 1-click
# ----------------------------
@torch.no_grad()
def build_q5(imgs, pos01, neg01, prev01, device):
    B, _, H, W = imgs.shape
    box01 = torch.zeros((B, 1, H, W), device=device, dtype=torch.float32)
    pos  = pos01.clamp(0, 1) * float(CFG.PROMPT_SCALE_POS)
    neg  = neg01.clamp(0, 1) * float(CFG.PROMPT_SCALE_NEG)
    prev = prev01.clamp(0, 1) * float(CFG.PROMPT_SCALE_PREV)
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
def posonly_oneclick_predict_with_click(seg, imgs, msks, s_img, s_msk, device, gen: torch.Generator):
    """
    Returns:
      pred0, pred1: (B,1,H,W)
      click_yx: list of (y,x) or None per sample
      pos01: (B,1,H,W) click map (stamped)
    """
    B, _, H, W = imgs.shape
    pos01 = torch.zeros((B, 1, H, W), device=device)
    neg01 = torch.zeros((B, 1, H, W), device=device)
    prev  = torch.zeros((B, 1, H, W), device=device)

    q5_0 = build_q5(imgs, pos01, neg01, prev, device)
    pred0 = torch.sigmoid(seg(q5_0, s_img, s_msk)).float()

    rng = _seeded_rng_from_torch(gen)
    pred_bin = (pred0 > float(CFG.ORACLE_PRED_THR)).float()
    gt_bin   = (msks > 0.5).float()

    click_yx = []
    for b in range(B):
        g = (gt_bin[b, 0].detach().cpu().numpy() > 0.5).astype(np.uint8)
        p = (pred_bin[b, 0].detach().cpu().numpy() > 0.5).astype(np.uint8)
        if int(g.sum()) > 0:
            fn = ((g == 1) & (p == 0)).astype(np.uint8)
            yx = _choose_pos_point_humansim(g, fn, rng)
            if yx is not None:
                _stamp(pos01[b, 0], yx[0], yx[1], r=int(CFG.CLICK_STAMP_R), v=1.0)
                click_yx.append((int(yx[0]), int(yx[1])))
            else:
                click_yx.append(None)
        else:
            click_yx.append(None)

    prev_in = pred0 if CFG.PREV_USE_PROB else pred_bin
    q5_1 = build_q5(imgs, pos01, neg01, prev_in, device)
    pred1 = torch.sigmoid(seg(q5_1, s_img, s_msk)).float()
    return pred0, pred1, click_yx, pos01


# ----------------------------
# Visualization helpers
# ----------------------------
def _to_u8(x01):
    x = np.clip(x01, 0.0, 1.0)
    return (x * 255.0 + 0.5).astype(np.uint8)

def _overlay_mask(gray_u8, mask01, color=(0, 255, 0), alpha=0.45):
    # gray_u8: (H,W) uint8
    # mask01: (H,W) float/bool
    img = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)
    m = (mask01 > 0.5).astype(np.uint8)
    if m.max() == 0:
        return img
    col = np.zeros_like(img, dtype=np.uint8)
    col[:, :] = np.array(color, dtype=np.uint8)
    img2 = img.copy()
    img2[m == 1] = (img2[m == 1].astype(np.float32) * (1 - alpha) + col[m == 1].astype(np.float32) * alpha).astype(np.uint8)
    return img2

def _draw_click(img_bgr, yx, color=(0, 0, 255), r=5, thickness=2):
    if yx is None:
        return img_bgr
    y, x = int(yx[0]), int(yx[1])
    cv2.circle(img_bgr, (x, y), int(r), color, int(thickness))
    cv2.circle(img_bgr, (x, y), 1, (255, 255, 255), -1)
    return img_bgr


# ----------------------------
# Main
# ----------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=None)
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--outdir", type=str, default=None)
    ap.add_argument("--img", type=int, default=None)
    ap.add_argument("--device", type=str, default=None)

    ap.add_argument("--amp", type=int, default=None)
    ap.add_argument("--amp_dtype", type=str, default=None)

    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--num", type=int, default=None)
    ap.add_argument("--save_every", type=int, default=None)
    ap.add_argument("--pick_nonempty", type=int, default=None)

    ap.add_argument("--bank", type=int, default=None)
    ap.add_argument("--proto_m", type=int, default=None)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--beta", type=float, default=None)
    ap.add_argument("--cand_mult", type=int, default=None)
    ap.add_argument("--support_k", type=int, default=None)

    ap.add_argument("--replkse", type=int, default=None)
    ap.add_argument("--replk_k", type=int, default=None)
    ap.add_argument("--se_red", type=int, default=None)

    ap.add_argument("--cross_replkse", type=int, default=None)
    ap.add_argument("--cross_last_n", type=int, default=None)
    ap.add_argument("--cross_k", type=int, default=None)
    ap.add_argument("--cross_se_red", type=int, default=None)

    ap.add_argument("--clahe", type=int, default=None)
    return ap.parse_args()

def main():
    args = parse_args()

    if args.root is not None: CFG.ROOT = str(args.root)
    if args.ckpt is not None: CFG.CKPT = str(args.ckpt)
    if args.outdir is not None: CFG.OUTDIR = str(args.outdir)
    if args.img is not None: CFG.IMG_SIZE = int(args.img)
    if args.device is not None: CFG.DEVICE = str(args.device)

    if args.amp is not None: CFG.AMP = bool(int(args.amp))
    if args.amp_dtype is not None: CFG.AMP_DTYPE = str(args.amp_dtype)

    if args.seed is not None: CFG.SEED = int(args.seed)
    if args.num is not None: CFG.NUM_SAMPLES = int(args.num)
    if args.save_every is not None: CFG.SAVE_EVERY = int(args.save_every)
    if args.pick_nonempty is not None: CFG.PICK_NONEMPTY = bool(int(args.pick_nonempty))

    if args.bank is not None: CFG.SUPPORT_NUM = int(args.bank)
    if args.proto_m is not None: CFG.PROTO_M = int(args.proto_m)
    if args.alpha is not None: CFG.ALPHA = float(args.alpha)
    if args.beta is not None: CFG.BETA = float(args.beta)
    if args.cand_mult is not None: CFG.PROTO_CAND_MULT = int(args.cand_mult)
    if args.support_k is not None: CFG.SUPPORT_K = int(args.support_k)

    if args.replkse is not None: CFG.REPLKSE_ENABLE = bool(int(args.replkse))
    if args.replk_k is not None: CFG.REPLK_KERNEL = int(args.replk_k)
    if args.se_red is not None: CFG.SE_REDUCTION = int(args.se_red)

    if args.cross_replkse is not None: CFG.CROSS_REPLKSE_ENABLE = bool(int(args.cross_replkse))
    if args.cross_last_n is not None: CFG.CROSS_LAST_N = int(args.cross_last_n)
    if args.cross_k is not None: CFG.CROSS_KERNEL = int(args.cross_k)
    if args.cross_se_red is not None: CFG.CROSS_SE_REDUCTION = int(args.cross_se_red)

    if args.clahe is not None: CFG.USE_CLAHE = bool(int(args.clahe))

    assert CFG.CKPT and os.path.exists(CFG.CKPT), f"--ckpt not found: {CFG.CKPT}"

    set_seed(CFG.SEED)
    os.makedirs(CFG.OUTDIR, exist_ok=True)
    vis_dir = os.path.join(CFG.OUTDIR, "vis")
    os.makedirs(vis_dir, exist_ok=True)

    device = torch.device(CFG.DEVICE)

    print(f"[Info] root={CFG.ROOT}")
    print(f"[Info] ckpt={CFG.CKPT}")
    print(f"[Info] outdir={CFG.OUTDIR}")
    print(f"[Info] device={CFG.DEVICE} amp={CFG.AMP} dtype={CFG.AMP_DTYPE} img={CFG.IMG_SIZE}")
    print(f"[Info] SUPPORT_K={CFG.SUPPORT_K} bank={CFG.SUPPORT_NUM} proto_m={CFG.PROTO_M} alpha={CFG.ALPHA} beta={CFG.BETA}")

    # list + split (same as train pack)
    pairs = list_busi_pairs(CFG.ROOT)
    tr_idx, _val_idx, te_idx = split_pairs(pairs)
    print(f"[Info] pairs={len(pairs)} | train={len(tr_idx)} test={len(te_idx)}")

    # Build support bank from train split
    bank_ds = BUSIDatasetBenMal(pairs, tr_idx, train=False, cache_items=0)
    bank_loader = DataLoader(bank_ds, batch_size=1, shuffle=True, num_workers=0)
    bank = build_support_bank(bank_loader)
    print(f"[Info] Support bank built: N={bank['imgs'].shape[0]}")

    # Test dataset/loader
    te_ds = BUSIDatasetBenMal(pairs, te_idx, train=False, cache_items=256)
    te_loader = DataLoader(te_ds, batch_size=1, shuffle=False, num_workers=0)

    # Model
    seg = WrappedMVS(device).to(device)

    # Load ckpt
    ck = torch.load(CFG.CKPT, map_location=device)
    if isinstance(ck, dict) and ("model" in ck):
        seg.load_state_dict(ck["model"], strict=False)
    elif isinstance(ck, dict) and ("state_dict" in ck):
        seg.load_state_dict(ck["state_dict"], strict=False)
    elif isinstance(ck, dict):
        # sometimes saved as full state_dict
        try:
            seg.load_state_dict(ck, strict=False)
        except Exception:
            # last resort: ck might contain nested keys
            raise RuntimeError(f"Unrecognized ckpt dict keys: {list(ck.keys())[:20]}")
    else:
        seg.load_state_dict(ck, strict=False)

    seg.eval()
    print("[Info] Loaded checkpoint.")

    # pick indices preference: non-empty GT
    chosen = []
    if CFG.PICK_NONEMPTY:
        for i in range(len(te_ds)):
            _img, _msk, _cid, _ip, _mp, is_pos = te_ds[i]
            if int(is_pos) == 1:
                chosen.append(i)
            if len(chosen) >= int(CFG.NUM_SAMPLES):
                break

    if len(chosen) < int(CFG.NUM_SAMPLES):
        # fill remaining with any samples
        for i in range(len(te_ds)):
            if i in chosen:
                continue
            chosen.append(i)
            if len(chosen) >= int(CFG.NUM_SAMPLES):
                break

    print(f"[Info] Will run {len(chosen)} samples (pick_nonempty={CFG.PICK_NONEMPTY}).")

    gen_support = torch.Generator(device="cpu").manual_seed(12345)

    dices0, dices1, ious0, ious1 = [], [], [], []

    for k, ds_i in enumerate(chosen):
        imgs, msks, _cid, img_path, msk_path, is_pos = te_ds[ds_i]
        imgs = imgs.unsqueeze(0).to(device, non_blocking=True)  # (1,1,H,W)
        msks = msks.unsqueeze(0).to(device, non_blocking=True)

        # sample prototypes (K=2)
        s_img, s_msk = sample_weighted_prototype(bank, device=device, B=1, gen=gen_support)

        # deterministic click seed per sample
        gen_click = torch.Generator(device="cpu").manual_seed(777777 + ds_i * 100 + 1)

        with torch.no_grad():
            with _autocast():
                pred0, pred1, click_yx, pos01 = posonly_oneclick_predict_with_click(
                    seg, imgs, msks, s_img, s_msk, device=device, gen=gen_click
                )

        d0 = float(dice_metric_tensor(pred0, msks).mean().item())
        d1 = float(dice_metric_tensor(pred1, msks).mean().item())
        j0 = float(iou_metric_tensor(pred0, msks).mean().item())
        j1 = float(iou_metric_tensor(pred1, msks).mean().item())
        dices0.append(d0); dices1.append(d1); ious0.append(j0); ious1.append(j1)

        if (CFG.SAVE_EVERY > 0) and (k % int(CFG.SAVE_EVERY) == 0):
            # make images
            img01 = imgs[0, 0].detach().float().cpu().numpy()
            gt01  = msks[0, 0].detach().float().cpu().numpy()
            p001  = pred0[0, 0].detach().float().cpu().numpy()
            p101  = pred1[0, 0].detach().float().cpu().numpy()

            img_u8 = _to_u8(img01)
            gt_u8  = _to_u8(gt01)
            p0_u8  = _to_u8(p001)
            p1_u8  = _to_u8(p101)

            ov_gt = _overlay_mask(img_u8, gt01, color=(0, 255, 0), alpha=0.45)
            ov_p0 = _overlay_mask(img_u8, p001, color=(255, 0, 0), alpha=0.45)
            ov_p1 = _overlay_mask(img_u8, p101, color=(0, 0, 255), alpha=0.45)

            yx = click_yx[0] if (len(click_yx) > 0) else None
            ov_p1 = _draw_click(ov_p1, yx, color=(0, 255, 255), r=6, thickness=2)
            ov_gt = _draw_click(ov_gt, yx, color=(0, 255, 255), r=6, thickness=2)

            tag = f"{k:04d}_pos{int(is_pos)}_d0{d0:.3f}_d1{d1:.3f}"
            cv2.imwrite(os.path.join(vis_dir, f"{tag}_img.png"), img_u8)
            cv2.imwrite(os.path.join(vis_dir, f"{tag}_gt.png"), gt_u8)
            cv2.imwrite(os.path.join(vis_dir, f"{tag}_pred0.png"), p0_u8)
            cv2.imwrite(os.path.join(vis_dir, f"{tag}_pred1.png"), p1_u8)
            cv2.imwrite(os.path.join(vis_dir, f"{tag}_ovGT.png"), ov_gt)
            cv2.imwrite(os.path.join(vis_dir, f"{tag}_ovP0.png"), ov_p0)
            cv2.imwrite(os.path.join(vis_dir, f"{tag}_ovP1_click.png"), ov_p1)

        if (k + 1) % 10 == 0:
            print(f"[{k+1:04d}/{len(chosen)}] Dice0={np.mean(dices0):.4f} Dice1={np.mean(dices1):.4f} | IoU0={np.mean(ious0):.4f} IoU1={np.mean(ious1):.4f}")

    # summary
    mean_d0 = float(np.mean(dices0)) if dices0 else 0.0
    mean_d1 = float(np.mean(dices1)) if dices1 else 0.0
    mean_j0 = float(np.mean(ious0)) if ious0 else 0.0
    mean_j1 = float(np.mean(ious1)) if ious1 else 0.0

    print("============================================================")
    print(f"[Done] N={len(chosen)}")
    print(f"  Dice@0click: {mean_d0:.4f}")
    print(f"  Dice@1click: {mean_d1:.4f}")
    print(f"  IoU @0click: {mean_j0:.4f}")
    print(f"  IoU @1click: {mean_j1:.4f}")
    print(f"[Saved] visualizations in: {vis_dir}")
    print("============================================================")

    # save a small txt
    with open(os.path.join(CFG.OUTDIR, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"N={len(chosen)}\n")
        f.write(f"Dice0={mean_d0:.6f}\nDice1={mean_d1:.6f}\n")
        f.write(f"IoU0={mean_j0:.6f}\nIoU1={mean_j1:.6f}\n")

    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    main()