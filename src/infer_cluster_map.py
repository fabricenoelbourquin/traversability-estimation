#!/usr/bin/env python3
"""
Infer a cluster map over a SwissImage chip using DINOv3 and pre-fitted KMeans.
Modes:
  - dino:  assign directly on DINO features
  - stego: project DINO features through STEGO head, then assign
  - both:  do both in one run (single DINO forward)

Outputs:
  - cluster_kmeans<K>_dino.tif/.png and/or cluster_kmeans<K>_stego.tif/.png
  - *_rgb.png previews if enabled

Example Use:
python src/infer_cluster_map.py --mission ETH-1
"""

from __future__ import annotations
import argparse, io, json, os, re
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
import joblib

import torch
import torch.nn as nn
import torch.nn.functional as F

import rasterio
from rasterio.io import MemoryFile

import yaml
from transformers import AutoImageProcessor, AutoModel

# ---------- project paths ----------
try:
    from utils.paths import get_paths
except ImportError:
    from paths import get_paths

P = get_paths()
VALID_IMG_EXT = (".tif", ".tiff", ".png", ".jpg", ".jpeg")

# ---------- config ----------
def load_yaml(p: Path) -> dict:
    return yaml.safe_load(p.read_text()) if p.exists() else {}

def str2bool(x: Optional[str]) -> Optional[bool]:
    if x is None:
        return None
    s = str(x).strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return True
    if s in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got {x}")

# ---------- device ----------
def pick_device(arg: str = "auto") -> str:
    if arg != "auto":
        return arg
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

# ---------- model path resolution ----------
def resolve_under_models(candidate: str) -> Path:
    """
    Accept absolute paths; otherwise interpret relative to DATA_ROOT/models.
    """
    cand = Path(candidate)
    if cand.is_absolute():
        if not cand.exists():
            raise FileNotFoundError(f"Model not found: {candidate}")
        return cand
    models_root = P.get("MODELS")
    if not models_root:
        raise FileNotFoundError("P['MODELS'] is not set; check paths.py / paths.yaml")
    p = Path(models_root) / candidate
    if p.exists():
        return p
    # best effort: if user passed a bare filename, try common subdirs
    for sub in ("kmeans", "heads"):
        q = Path(models_root) / sub / candidate
        if q.exists():
            return q
    raise FileNotFoundError(f"Model not found under MODELS: {candidate}")

# ---------- IO helpers ----------
def to_uint8_rgb(arr_hw3: np.ndarray) -> np.ndarray:
    x = arr_hw3.astype(np.float32)
    if x.dtype != np.uint8 and x.max() > 1.5:
        x = np.clip(x / 10000.0, 0.0, 1.0)
    x = np.clip(x, 0.0, 1.0)
    lo, hi = np.percentile(x, 2), np.percentile(x, 98)
    if hi - lo > 1e-6:
        x = np.clip((x - lo) / (hi - lo), 0, 1)
    return (x * 255.0 + 0.5).astype(np.uint8)

def _read_geotiff_rgb(path_or_mem) -> tuple[np.ndarray, rasterio.DatasetReader]:
    src = rasterio.open(path_or_mem)
    try:
        bands = src.count
        if bands >= 3:
            arr = np.stack([src.read(1), src.read(2), src.read(3)], axis=-1)
        elif bands == 1:
            single = src.read(1); arr = np.stack([single, single, single], axis=-1)
        else:
            raise SystemExit(f"[error] unexpected band count: {bands}")
        return arr, src
    except Exception:
        src.close()
        raise

def load_image_from_path(path: str, stride: int) -> tuple[np.ndarray, Optional[rasterio.DatasetReader], tuple[int,int]]:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".tif", ".tiff"):
        rgb, ds = _read_geotiff_rgb(path)
        rgb8 = to_uint8_rgb(rgb)
    else:
        im = Image.open(path).convert("RGB")
        rgb8, ds = np.array(im, dtype=np.uint8), None
    H, W = rgb8.shape[:2]
    Hs, Ws = (H // stride) * stride, (W // stride) * stride
    return rgb8[:Hs, :Ws], ds, (Hs, Ws)

def write_uint16_label_map(out_base: Path, labels: np.ndarray, ref_ds: Optional[rasterio.DatasetReader]) -> Path:
    if ref_ds is not None:
        meta = ref_ds.meta.copy()
        meta.update({
            "count": 1, "dtype": "uint16", "compress": "deflate", "predictor": 2,
            "nodata": None, "width": labels.shape[1], "height": labels.shape[0],
        })
        out_tif = out_base.with_suffix(".tif")
        with rasterio.open(out_tif, "w", **meta) as dst:
            dst.write(labels.astype(np.uint16), 1)
        return out_tif
    else:
        out_png = out_base.with_suffix(".png")
        Image.fromarray(labels.astype(np.uint16), mode="I;16").save(out_png)
        return out_png

def make_palette(n: int) -> np.ndarray:
    import colorsys
    cols = []
    for i in range(max(1, n)):
        h = (i / float(n)) % 1.0
        s, v = 0.65, 0.95
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        cols.append([int(r*255), int(g*255), int(b*255)])
    return np.array(cols, dtype=np.uint8)

def save_color_png(out_base: Path, labels: np.ndarray, palette: np.ndarray) -> Path:
    rgb = palette[labels % len(palette)]
    p = out_base.with_name(out_base.name + "_rgb").with_suffix(".png")
    Image.fromarray(rgb).save(p)
    return p

# ---------- DINO features ----------
@torch.no_grad()
def dinov3_patch_features(rgb8: np.ndarray, model: AutoModel, processor: AutoImageProcessor, device: str, stride: int) -> np.ndarray:
    H, W, _ = rgb8.shape
    Hs, Ws = (H // stride) * stride, (W // stride) * stride
    rgb8 = rgb8[:Hs, :Ws, :]
    inputs = processor(images=rgb8, return_tensors="pt", do_resize=False, do_center_crop=False).to(device)
    model.eval()
    out = model(**inputs).last_hidden_state  # [1, 1+R+N, C]
    R = getattr(model.config, "num_register_tokens", 4)
    patch_tok = out[:, 1 + R:, :]
    C = patch_tok.shape[-1]
    h_feat, w_feat = Hs // stride, Ws // stride
    grid = patch_tok.reshape(1, h_feat, w_feat, C).squeeze(0)
    grid = F.normalize(grid, dim=-1).cpu().numpy()  # (hf, wf, C) L2
    return grid

# ---------- STEGO head loader ----------
def load_stego_head(path: Path, device: str) -> nn.Module:
    obj = torch.load(path, map_location=device)
    if isinstance(obj, nn.Module):
        return obj.eval().to(device)
    if isinstance(obj, dict):
        def count_linear(sd):
            import torch
            return sum(1 for k,v in sd.items() if isinstance(v, torch.Tensor) and v.ndim==2 and k.endswith(".weight"))
        best, cnt = None, 0
        def visit(x):
            nonlocal best, cnt
            if isinstance(x, dict):
                has_t = any(hasattr(v, "ndim") for v in x.values())
                if has_t:
                    c = count_linear(x)
                    if c > cnt: best, cnt = x, c
                for v in x.values(): visit(v)
        visit(obj)
        if not best or cnt == 0:
            raise SystemExit("[error] cannot infer MLP from checkpoint dict")
        ws = sorted([(k,v) for k,v in best.items() if hasattr(v, "ndim") and getattr(v, "ndim", 0)==2 and k.endswith(".weight")],
                    key=lambda kv: re.split(r'(\d+)', kv[0]))
        shapes = [w.shape for _, w in ws]
        dims = [shapes[0][1]] + [s[0] for s in shapes]
        layers = []
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1], bias=True))
            if i < len(dims)-2: layers.append(nn.ReLU(inplace=True))
        mlp = nn.Sequential(*layers).to(device).eval()
        li = 0
        for name, w in ws:
            while li < len(mlp) and not isinstance(mlp[li], nn.Linear):
                li += 1
            lin = mlp[li]; li += 1
            with torch.no_grad():
                lin.weight.copy_(w)
                bname = name.replace(".weight", ".bias")
                if bname in best:
                    lin.bias.copy_(best[bname])
                else:
                    lin.bias.zero_()
        return mlp
    raise SystemExit(f"[error] unsupported STEGO head format: {path}")

# ---------- KMeans loader (joblib/npz) ----------
class CentersOnly:
    def __init__(self, centers: np.ndarray):
        self.cluster_centers_ = np.asarray(centers, dtype=np.float32)
        self.n_clusters = int(self.cluster_centers_.shape[0])

def load_kmeans_any(path: Path):
    suf = path.suffix.lower()
    if suf in {".joblib", ".pkl"}:
        return joblib.load(path)
    if suf == ".npz":
        npz = np.load(path)
        # accept a few common keys
        for key in ("cluster_centers_", "centers", "C"):
            if key in npz:
                return CentersOnly(npz[key])
        raise SystemExit(f"[error] .npz KMeans missing centers (expected one of cluster_centers_, centers, C): {path}")
    raise SystemExit(f"[error] unsupported KMeans format: {path}")

# ---------- core ----------
@torch.no_grad()
def assign_labels_from_feats(Z: np.ndarray, kmeans_model, device: str, out_hw: tuple[int,int]) -> np.ndarray:
    H, W = out_hw
    Fsmall = torch.from_numpy(Z).permute(2,0,1).unsqueeze(0).to(device)   # [1,D,hf,wf]
    Fhi = F.interpolate(Fsmall, size=(H, W), mode="bilinear", align_corners=False)
    Fhi = F.normalize(Fhi, dim=1).squeeze(0).permute(1,2,0).contiguous()  # [H,W,D]

    centers = np.asarray(getattr(kmeans_model, "cluster_centers_", None), dtype=np.float32)
    if centers is None:
        raise SystemExit("[error] KMeans model lacks cluster_centers_")
    Ct = torch.from_numpy(centers).to(Fhi.device)
    Ct = F.normalize(Ct, dim=1)
    logits = torch.einsum("hwc,kc->hwk", Fhi, Ct)
    labels = torch.argmax(logits, dim=-1).cpu().numpy().astype(np.uint16)
    return labels

# ---------- mission/image resolution ----------
def resolve_mission_and_input(mission: Optional[str], input_path: Optional[str], out_subdir: str) -> tuple[Path, str, str, str]:
    """
    Returns:
      out_dir (where outputs go),
      chosen_input_path (string),
      mission_id ('' if none),
      short_folder ('' if none)
    """
    if input_path:
        out_dir = Path(input_path).parent / out_subdir   # <— use configured subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir, input_path, "", ""

    # mission-driven
    mj = P["REPO_ROOT"] / "config" / "missions.json"
    meta = json.loads(mj.read_text())
    mission_id = meta.get("_aliases", {}).get(mission, mission)
    entry = meta.get(mission_id)
    if not entry:
        raise KeyError(f"Mission '{mission}' not found in missions.json")
    short = entry["folder"]

    # latest swissimg chip
    img_dir = P["MAPS"] / short / "swissimg"
    cands = sorted(img_dir.glob("*_rgb8.tif"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        raise FileNotFoundError(f"No '*_rgb8.tif' under {img_dir}. Run get_swissimg_patch.py first or pass --input.")
    picked = str(cands[0])

    out_dir = P["MAPS"] / short / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir, picked, mission_id, short

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Infer cluster map from DINOv3 with optional STEGO head and KMeans (joblib/npz).")
    # inputs
    ap.add_argument("--mission", help="Mission alias or UUID (auto-picks latest SwissImage chip)")
    ap.add_argument("--input", help="Explicit path to an image (.tif/.tiff/.png/.jpg)")
    # model/config
    ap.add_argument("--config", default=None, help="Path to config/imagery.yaml (default: repo config)")
    ap.add_argument("--model-id", default="", help="HuggingFace model id override (else from config)")
    ap.add_argument("--device", default="auto", choices=["auto","cuda","mps","cpu"])

    # mode selector
    ap.add_argument("--mode", choices=["dino","stego","both"], default=None,
                    help="Override embedding.mode; 'both' runs DINO once and branches.")

    # model path overrides (optional)
    ap.add_argument("--kmeans-dino",  default="", help="Override DINO KMeans path (used when mode=dino/both).")
    ap.add_argument("--kmeans-stego", default="", help="Override STEGO KMeans path (used when mode=stego/both).")
    ap.add_argument("--stego-head",   default="", help="Override STEGO head (used when mode=stego/both).")

    # output
    ap.add_argument("--save-rgb", action="store_true", help="Force saving color PNG (overrides config).")
    ap.add_argument("--out-subdir", default="", help="Override output subdir (default from config).")

    args = ap.parse_args()

    if not args.mission and not args.input:
        raise SystemExit("Provide either --mission or --input.")

    # load config
    cfg_path = Path(args.config) if args.config else (P["REPO_ROOT"] / "config" / "imagery.yaml")
    cfg = load_yaml(cfg_path)
    emb = cfg.get("embedding", {}) or {}
    clu = cfg.get("clustering", {}) or {}

    stride   = int(emb.get("stride", 16))
    model_id = args.model_id or emb.get("model_id", "facebook/dinov3-vitl16-pretrain-sat493m")

    # REQUIRED in cfg or via CLI
    cfg_mode = emb.get("mode", None)
    if cfg_mode is None and args.mode is None:
        raise SystemExit("[error] embedding.mode must be set in config or overridden via --mode (dino|stego|both).")
    mode = args.mode or cfg_mode  # final mode: "dino" | "stego" | "both"

    defaults  = (clu.get("default_kmeans") or {})
    out_subdir = args.out_subdir or clu.get("out_subdir", "clusters")
    save_rgb   = args.save_rgb or bool(clu.get("save_rgb", True))

    # resolve mission/input (unchanged)
    out_dir, input_path, mission_id, short = resolve_mission_and_input(args.mission, args.input, out_subdir)

    # device + models
    device = pick_device(args.device)
    print(f"[info] device = {device}")
    processor  = AutoImageProcessor.from_pretrained(model_id)
    dino_model = AutoModel.from_pretrained(model_id).to(device).eval()

    # resolve model paths per mode
    kmeans_path_dino  = None
    kmeans_path_stego = None
    stego_head        = None

    if mode in ("dino", "both"):
        km_rel_d = args.kmeans_dino or defaults.get("dino", "")
        if not km_rel_d:
            raise SystemExit("[error] Set clustering.default_kmeans.dino or pass --kmeans-dino.")
        kmeans_path_dino = resolve_under_models(km_rel_d)

    if mode in ("stego", "both"):
        km_rel_s = args.kmeans_stego or defaults.get("stego", "")
        if not km_rel_s:
            raise SystemExit("[error] Set clustering.default_kmeans.stego or pass --kmeans-stego.")
        kmeans_path_stego = resolve_under_models(km_rel_s)

        sh_rel = args.stego_head or emb.get("default_stego_head", "")
        if not sh_rel:
            raise SystemExit("[error] mode includes STEGO but no head provided. Set embedding.default_stego_head or pass --stego-head.")
        stego_head_path = resolve_under_models(sh_rel)
        stego_head = load_stego_head(stego_head_path, device)

    # load KMeans models
    kmeans_dino  = load_kmeans_any(kmeans_path_dino)  if kmeans_path_dino  else None
    kmeans_stego = load_kmeans_any(kmeans_path_stego) if kmeans_path_stego else None

    rgb8, ref_ds, _ = load_image_from_path(input_path, stride=stride)
    H, W = rgb8.shape[:2]

    # one DINO pass
    feat = dinov3_patch_features(rgb8, dino_model, processor, device, stride)  # (hf, wf, C), L2-normalized

    written_paths = []
    palette_cache = {}

    def _out_basename(k: int, tag: str) -> str:
        return f"cluster_kmeans{k}_{tag}"

    if mode in ("dino", "both"):
        labels_d = assign_labels_from_feats(feat, kmeans_dino, device, (H, W))
        Kd = getattr(kmeans_dino, "n_clusters", len(getattr(kmeans_dino, "cluster_centers_", [])) or 0)
        base_d = out_dir / _out_basename(Kd, "dino")
        lab_p_d = write_uint16_label_map(base_d, labels_d, ref_ds); written_paths.append(lab_p_d)
        if save_rgb:
            palette = palette_cache.setdefault(Kd, make_palette(Kd if Kd else 100))
            rgb_p_d = save_color_png(base_d, labels_d, palette); written_paths.append(rgb_p_d)

    if mode in ("stego", "both"):
        hf, wf, C = feat.shape
        X = torch.from_numpy(feat).reshape(-1, C).to(device)
        with torch.no_grad():
            Y = stego_head(X)
            Y = F.normalize(Y, dim=-1)
        Z = Y.detach().cpu().numpy().reshape(hf, wf, -1)
        labels_s = assign_labels_from_feats(Z, kmeans_stego, device, (H, W))
        Ks = getattr(kmeans_stego, "n_clusters", len(getattr(kmeans_stego, "cluster_centers_", [])) or 0)
        base_s = out_dir / _out_basename(Ks, "stego")
        lab_p_s = write_uint16_label_map(base_s, labels_s, ref_ds); written_paths.append(lab_p_s)
        if save_rgb:
            palette = palette_cache.setdefault(Ks, make_palette(Ks if Ks else 100))
            rgb_p_s = save_color_png(base_s, labels_s, palette); written_paths.append(rgb_p_s)

    for p in written_paths:
        print("[ok] wrote", p)

    if ref_ds is not None:
        ref_ds.close()


if __name__ == "__main__":
    main()
