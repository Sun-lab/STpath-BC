#!/usr/bin/env python3

# ======================================================================
# ADD-ON: Feature extraction only (does NOT modify any original function)
# ======================================================================

import numpy as np
from tqdm import tqdm
from PIL import Image
from PIL import UnidentifiedImageError
import torch
import timm
from conch.open_clip_custom import create_model_from_pretrained
from huggingface_hub import login
from datetime import datetime
from timm.layers import SwiGLUPacked
from torchvision import transforms
import pickle
import os
import glob
import pandas as pd
import traceback, csv, shutil
import argparse

def _iter_patch_images_in_dir(patch_dir):
    exts = (".tif", ".tiff", ".jpg", ".jpeg", ".png")
    for fn in sorted(os.listdir(patch_dir)):
        if fn.lower().endswith(exts):
            yield os.path.join(patch_dir, fn)

def _make_patch_rec_from_file(img_path, patch_size_guess=240):
    im = Image.open(img_path).convert("RGB")
    arr = np.array(im)
    # patch_id = filename (no ext)
    pid = os.path.splitext(os.path.basename(img_path))[0]
    h, w = arr.shape[:2]
    return {
        "patch_id": pid,
        "row": None, "col": None,
        "x_start": None, "y_start": None,
        "x_end": None, "y_end": None,
        "patch_width": w, "patch_height": h,
        "patch_image": im,
        "patch_array": arr,
    }

def log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def load_Conch_model():
    hf_token = "YOUR_HUGGINGFACE_TOKEN_HERE"
    login(token=hf_token)
    model, transform = create_model_from_pretrained('conch_ViT-B-16', "hf_hub:MahmoodLab/conch", hf_auth_token=hf_token)
    return model, transform

def load_UNI2h_model():
    hf_token = "YOUR_HUGGINGFACE_TOKEN_HERE"
    login(token=hf_token) 
    timm_kwargs = {
        'img_size': 224, 
        'patch_size': 14, 
        'depth': 24,
        'num_heads': 24,
        'init_values': 1e-5, 
        'embed_dim': 1536,
        'mlp_ratio': 2.66667*2,
        'num_classes': 0, 
        'no_embed_class': True,
        'mlp_layer': timm.layers.SwiGLUPacked, 
        'act_layer': torch.nn.SiLU, 
        'reg_tokens': 8, 
        'dynamic_img_size': True
    }
    model_UNI2h = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
    return model_UNI2h

def load_ProvGigapath_model():
    hf_token = "YOUR_HUGGINGFACE_TOKEN_HERE"
    login(token=hf_token)
    tile_encoder = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
    return tile_encoder

def load_Virchow_model():
    hf_token = "YOUR_HUGGINGFACE_TOKEN_HERE"
    login(token=hf_token)
    model_Virchow = timm.create_model("hf-hub:paige-ai/Virchow", pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
    model_Virchow = model_Virchow.eval()
    return model_Virchow

def load_Virchow2_model():
    hf_token = "YOUR_HUGGINGFACE_TOKEN_HERE"
    login(token=hf_token)
    model_Virchow2 = timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
    model_Virchow2 = model_Virchow2.eval()
    return model_Virchow2

def create_patches_240x240(image_path):
    """Create 240x240 pixel patches from WSI"""
    Image.MAX_IMAGE_PIXELS = None
    image = Image.open(image_path)
    image_np = np.array(image)
    
    height, width = image_np.shape[:2]
    log(f"Original image size: {width} x {height}")
    
    # Fixed patch size: 240x240 pixels
    patch_size = 240
    
    # Calculate how many patches we can fit
    num_patches_width = width // patch_size
    num_patches_height = height // patch_size
    
    log(f"Creating {num_patches_height} x {num_patches_width} = {num_patches_height * num_patches_width} patches")
    log(f"Each patch size: {patch_size} x {patch_size} pixels")
    
    # Calculate discarded border sizes
    discarded_width = width % patch_size
    discarded_height = height % patch_size
    log(f"Discarded border: {discarded_width} pixels (width) x {discarded_height} pixels (height)")
    
    patch_list = []
    
    for row in range(num_patches_height):
        for col in range(num_patches_width):
            y_start = row * patch_size
            y_end = y_start + patch_size
            x_start = col * patch_size
            x_end = x_start + patch_size
            
            patch_np = image_np[y_start:y_end, x_start:x_end]
            patch_image = Image.fromarray(patch_np)
            patch_id = f"row{row:03d}_col{col:03d}"
            
            patch_info = {
                'patch_id': patch_id,
                'row': row,
                'col': col,
                'x_start': x_start,
                'y_start': y_start,
                'x_end': x_end,
                'y_end': y_end,
                'patch_width': patch_size,
                'patch_height': patch_size,
                'patch_image': patch_image,
                'patch_array': patch_np
            }
            
            patch_list.append(patch_info)
    
    log(f"Successfully created {len(patch_list)} patches")
    return patch_list


# ======================================================================
# ADD-ON: Feature extraction only (embeddings; includes Conch support)
# ======================================================================

def _log_fe(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [FE] {msg}")

def _setup_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def _load_backbones_only(use_models=("uni2h","virchow","virchow2","provgigapath","conch"),
                         hf_token=None,
                         device=None):
    """
    Loads backbones + transforms WITHOUT modifying any original functions.
    For Conch, we load directly here using hf_token (avoids editing your original loader).
    """
    from timm.layers import SwiGLUPacked
    import timm

    models = {}
    tmap = {}

    if device is None:
        device = _setup_device()

    # ----- UNI2-h -----
    if "uni2h" in use_models:
        timm_kwargs = {
            'img_size': 224, 'patch_size': 14, 'depth': 24,
            'num_heads': 24, 'init_values': 1e-5, 'embed_dim': 1536,
            'mlp_ratio': 2.66667*2, 'num_classes': 0, 'no_embed_class': True,
            'mlp_layer': SwiGLUPacked, 'act_layer': torch.nn.SiLU,
            'reg_tokens': 8, 'dynamic_img_size': True
        }
        m_uni = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs).eval().to(device)
        tfm_std = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        models["uni2h"] = m_uni; tmap["uni2h"] = tfm_std

    # ----- Virchow -----
    if "virchow" in use_models:
        m_v = timm.create_model("hf-hub:paige-ai/Virchow", pretrained=True,
                                mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU).eval().to(device)
        tfm_std = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        models["virchow"] = m_v; tmap["virchow"] = tfm_std

    # ----- Virchow2 -----
    if "virchow2" in use_models:
        m_v2 = timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=True,
                                 mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU).eval().to(device)
        tfm_std = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        models["virchow2"] = m_v2; tmap["virchow2"] = tfm_std

    # ----- Prov-Gigapath -----
    if "provgigapath" in use_models:
        m_p = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True).eval().to(device)
        tfm_pg = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        models["provgigapath"] = m_p; tmap["provgigapath"] = tfm_pg

    # ----- Conch (loaded here to avoid editing your original loader) -----
    if "conch" in use_models:
        if hf_token is None:
            hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            _log_fe("Conch requested but HF_TOKEN is not set. Skipping Conch.")
        else:
            from huggingface_hub import login
            from conch.open_clip_custom import create_model_from_pretrained
            login(token=hf_token)
            m_conch, tfm_conch = create_model_from_pretrained(
                'conch_ViT-B-16', "hf_hub:MahmoodLab/conch", hf_auth_token=hf_token
            )
            models["conch"] = m_conch.eval().to(device)
            tmap["conch"] = tfm_conch

    return models, tmap, device

def _is_white_patch_simple(patch_array, gray_thresh=220, white_ratio=0.40):
    if patch_array.ndim == 3:
        gray = patch_array.mean(axis=2)
    else:
        gray = patch_array
    return (gray > gray_thresh).mean() > white_ratio

def extract_embeddings_for_patches(patch_list, models, tmap, device):
    out = []
    with torch.no_grad():
        for p in patch_list:
            if _is_white_patch_simple(p["patch_array"]):
                continue
            rec = {
                "patch_id": p["patch_id"],
                "row": p["row"], "col": p["col"],
                "x_start": p["x_start"], "y_start": p["y_start"],
                "x_end": p["x_end"], "y_end": p["y_end"],
                "embeddings": {}
            }
            im = p["patch_image"]

            if "uni2h" in models:
                x = tmap["uni2h"](im).unsqueeze(0).to(device)
                feats = models["uni2h"](x)
                rec["embeddings"]["uni2h"] = feats.cpu().numpy().reshape(-1)

            if "virchow" in models:
                x = tmap["virchow"](im).unsqueeze(0).to(device)
                emb = models["virchow"](x)
                v = torch.cat([emb[:, 0], emb[:, 5:].mean(1)], dim=-1)
                rec["embeddings"]["virchow"] = v.cpu().numpy().reshape(-1)

            if "virchow2" in models:
                x = tmap["virchow2"](im).unsqueeze(0).to(device)
                emb = models["virchow2"](x)
                v = torch.cat([emb[:, 0], emb[:, 5:].mean(1)], dim=-1)
                rec["embeddings"]["virchow2"] = v.cpu().numpy().reshape(-1)

            if "provgigapath" in models:
                x = tmap["provgigapath"](im).unsqueeze(0).to(device)
                feats = models["provgigapath"](x)
                rec["embeddings"]["provgigapath"] = feats.cpu().numpy().reshape(-1)

            if "conch" in models:
                x = tmap["conch"](im).unsqueeze(0).to(device)
                feats = models["conch"].encode_image(x, proj_contrast=False, normalize=False)
                rec["embeddings"]["conch"] = feats.cpu().numpy().reshape(-1)

            out.append(rec)
    return out

def save_embeddings_pt(wsi_path, embeddings, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    wsi_name = os.path.splitext(os.path.basename(wsi_path))[0]
    out_path = os.path.join(output_dir, f"{wsi_name}.emb.pt")
    torch.save({"wsi_name": wsi_name, "patch_size": 240, "patches": embeddings}, out_path)
    _log_fe(f"Saved embeddings: {out_path}")

# --- add near top of add-on section ---


def _record_problematic(wsi_path, reason, problem_dir, rows):
    os.makedirs(problem_dir, exist_ok=True)
    rows.append({"wsi": wsi_path, "reason": reason})
    # keep a symlink to the problematic file (cheap and easy to inspect)
    link_path = os.path.join(problem_dir, os.path.basename(wsi_path))
    try:
        if not os.path.exists(link_path):
            os.symlink(wsi_path, link_path)
    except Exception as e:
        _log_fe(f"[WARN] Could not symlink {wsi_path} -> {link_path}: {e}")

def run_feature_extraction_only(input_dir, output_dir,
                                use_models=("uni2h","virchow","virchow2","provgigapath","conch"),
                                hf_token=None):

    device = _setup_device()
    _log_fe(f"Using device: {device}")

    # authenticate once so HF-backed models (UNI2-h, Virchow*) don’t 401
    try:
        if hf_token is None:
            hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        if hf_token:
            from huggingface_hub import login as _hf_login
            _hf_login(token=hf_token)
    except Exception as e:
        _log_fe(f"[WARN] HF login failed: {e}")

    models, tmap, device = _load_backbones_only(use_models=use_models, hf_token=hf_token, device=device)

    # Gather WSIs
    tiffs = glob.glob(os.path.join(input_dir, "*.tif")) + glob.glob(os.path.join(input_dir, "*.tiff"))
    _log_fe(f"Found {len(tiffs)} WSIs")

    problem_dir = os.path.join(output_dir, "_problematic")
    problematic_rows = []

    for i, wsi_path in enumerate(tiffs, 1):
        wsi_name = os.path.splitext(os.path.basename(wsi_path))[0]
        out_pt = os.path.join(output_dir, f"{wsi_name}.emb.pt")
        if os.path.exists(out_pt):
            _log_fe(f"[{i}/{len(tiffs)}] {wsi_name}: embeddings exist, skipping")
            continue

        _log_fe(f"[{i}/{len(tiffs)}] Tiling {wsi_name}")
        try:
            patches = create_patches_240x240(wsi_path)  # original function
        except UnidentifiedImageError as e:
            reason = f"UnidentifiedImageError while opening image: {e}"
            _log_fe(f"[{i}/{len(tiffs)}] SKIP {wsi_name}: {reason}")
            _record_problematic(wsi_path, reason, problem_dir, problematic_rows)
            continue
        except Exception as e:
            reason = f"tiling failed: {e.__class__.__name__}: {e}"
            _log_fe(f"[{i}/{len(tiffs)}] SKIP {wsi_name}: {reason}")
            _record_problematic(wsi_path, reason, problem_dir, problematic_rows)
            continue

        _log_fe(f"[{i}/{len(tiffs)}] Extracting embeddings for {wsi_name} (n_patches={len(patches)})")
        try:
            embeddings = extract_embeddings_for_patches(patches, models, tmap, device)
        except Exception as e:
            # If a model hiccups on this slide, skip the slide and keep going
            tb = traceback.format_exc(limit=2)
            reason = f"embedding failed: {e.__class__.__name__}: {e}"
            _log_fe(f"[{i}/{len(tiffs)}] SKIP {wsi_name}: {reason}\n{tb}")
            _record_problematic(wsi_path, reason, problem_dir, problematic_rows)
            continue

        _log_fe(f"[{i}/{len(tiffs)}] {wsi_name}: kept {len(embeddings)} non-white patches")
        try:
            save_embeddings_pt(wsi_path, embeddings, output_dir)
        except Exception as e:
            reason = f"saving failed: {e.__class__.__name__}: {e}"
            _log_fe(f"[{i}/{len(tiffs)}] SKIP {wsi_name}: {reason}")
            _record_problematic(wsi_path, reason, problem_dir, problematic_rows)
            continue

    # persist the list of problematic inputs
    if problematic_rows:
        csv_path = os.path.join(problem_dir, "problematic_wsi.csv")
        txt_path = os.path.join(problem_dir, "problematic_wsi.txt")
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["wsi", "reason"])
            w.writeheader()
            w.writerows(problematic_rows)
        with open(txt_path, "w") as f:
            for r in problematic_rows:
                f.write(f"{r['wsi']}\t{r['reason']}\n")
        _log_fe(f"Recorded {len(problematic_rows)} problematic WSIs in {problem_dir}")
    else:
        _log_fe("No problematic WSIs encountered 🎉")


def _merge_embeddings(existing_recs, new_recs):
    """Merge embeddings dicts per patch by patch_id (falls back to row/col)."""
    by_id = {r["patch_id"]: r for r in existing_recs}
    # also map by (row,col) in case someone edited patch_id format
    by_rc = {(r["row"], r["col"]): r for r in existing_recs}

    merged = existing_recs  # mutate in place
    for nr in new_recs:
        key = nr["patch_id"]
        target = by_id.get(key) or by_rc.get((nr["row"], nr["col"]))
        if target is None:
            # If a new (non-white) patch appears due to threshold tweaks, append it.
            merged.append(nr)
            continue
        # add/overwrite only the new model keys
        for k, v in nr["embeddings"].items():
            target["embeddings"][k] = v
    return merged

def run_update_embeddings(
    input_dir,
    output_dir,
    add_models=("virchow","virchow2"),
    hf_token=None
):
    """
    Incrementally add embeddings for `add_models` into existing *.emb.pt files.
    Only computes the requested models; pre-existing model vectors are preserved.
    """
    device = _setup_device()
    _log_fe(f"[UPDATE] Using device: {device}")

    # HF login once
    try:
        if hf_token is None:
            hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        if hf_token:
            from huggingface_hub import login as _hf_login
            _hf_login(token=hf_token)
    except Exception as e:
        _log_fe(f"[WARN] HF login failed: {e}")

    # load backbones/transforms for ONLY the models we’re adding
    models, tmap, device = _load_backbones_only(use_models=add_models, hf_token=hf_token, device=device)

    if not models:
        _log_fe("[UPDATE] No models loaded for update (check add_models/HF_TOKEN). Nothing to do.")
        return

    tiffs = glob.glob(os.path.join(input_dir, "*.tif")) + glob.glob(os.path.join(input_dir, "*.tiff"))
    _log_fe(f"[UPDATE] Found {len(tiffs)} WSIs")

    problem_dir = os.path.join(output_dir, "_problematic_update")
    problematic_rows = []

    for i, wsi_path in enumerate(tiffs, 1):
        wsi_name = os.path.splitext(os.path.basename(wsi_path))[0]
        out_pt = os.path.join(output_dir, f"{wsi_name}.emb.pt")

        if not os.path.exists(out_pt):
            _log_fe(f"[{i}/{len(tiffs)}] {wsi_name}: no existing .emb.pt found; skipping (use full run to create).")
            continue

        # Load existing embeddings
        try:
            blob = torch.load(out_pt, map_location="cpu")
            existing_patches = blob.get("patches", [])
            _log_fe(f"[{i}/{len(tiffs)}] {wsi_name}: existing patches = {len(existing_patches)}")
        except Exception as e:
            reason = f"load existing failed: {e.__class__.__name__}: {e}"
            _log_fe(f"[{i}/{len(tiffs)}] SKIP {wsi_name}: {reason}")
            _record_problematic(wsi_path, reason, problem_dir, problematic_rows)
            continue

        # Fast path: if BOTH virchow and virchow2 already present, skip
        have_all = True
        sample = next((p for p in existing_patches if p.get("embeddings")), None)
        if sample:
            for m in add_models:
                if m not in sample["embeddings"]:
                    have_all = False
                    break
        else:
            have_all = False

        if have_all:
            _log_fe(f"[{i}/{len(tiffs)}] {wsi_name}: already has {add_models}, skipping.")
            continue

        # Recreate the exact tiling and white filtering, then extract ONLY the new models
        try:
            patches = create_patches_240x240(wsi_path)
            _log_fe(f"[{i}/{len(tiffs)}] Extracting new embeddings {add_models} (n_patches={len(patches)})")
            new_recs = extract_embeddings_for_patches(patches, models, tmap, device)
        except Exception as e:
            tb = traceback.format_exc(limit=2)
            reason = f"update embedding failed: {e.__class__.__name__}: {e}"
            _log_fe(f"[{i}/{len(tiffs)}] SKIP {wsi_name}: {reason}\n{tb}")
            _record_problematic(wsi_path, reason, problem_dir, problematic_rows)
            continue

        # Merge & save
        merged = _merge_embeddings(existing_patches, new_recs)
        blob["patches"] = merged
        # keep original metadata
        if "patch_size" not in blob:
            blob["patch_size"] = 240
        if "wsi_name" not in blob:
            blob["wsi_name"] = wsi_name

        # Backup then overwrite
        try:
            backup_path = out_pt + ".bak"
            if not os.path.exists(backup_path):
                shutil.copy2(out_pt, backup_path)
            torch.save(blob, out_pt)
            _log_fe(f"[{i}/{len(tiffs)}] UPDATED {wsi_name}: added {add_models} → {out_pt}")
        except Exception as e:
            reason = f"saving updated failed: {e.__class__.__name__}: {e}"
            _log_fe(f"[{i}/{len(tiffs)}] SKIP {wsi_name}: {reason}")
            _record_problematic(wsi_path, reason, problem_dir, problematic_rows)
            continue

    if problematic_rows:
        csv_path = os.path.join(problem_dir, "problematic_update.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["wsi", "reason"])
            w.writeheader()
            w.writerows(problematic_rows)
        _log_fe(f"[UPDATE] Recorded {len(problematic_rows)} issues in {problem_dir}")
    else:
        _log_fe("[UPDATE] All updates completed 🎉")

def run_feature_extraction_on_patch_dir(patch_dir, output_dir,
                                        use_models=("uni2h","virchow","virchow2","provgigapath","conch"),
                                        hf_token=None):
    device = _setup_device()
    _log_fe(f"[PATCHES] Using device: {device}")
    os.makedirs(output_dir, exist_ok=True)

    # HF login once (if needed)
    try:
        if hf_token is None:
            hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        if hf_token:
            from huggingface_hub import login as _hf_login
            _hf_login(token=hf_token)
    except Exception as e:
        _log_fe(f"[PATCHES][WARN] HF login failed: {e}")

    models, tmap, device = _load_backbones_only(use_models=use_models, hf_token=hf_token, device=device)

    # collect patch files
    patch_files = list(_iter_patch_images_in_dir(patch_dir))
    _log_fe(f"[PATCHES] {patch_dir}: found {len(patch_files)} patch files")
    if not patch_files:
        _log_fe(f"[PATCHES] {patch_dir}: nothing to do")
        return

    # output name = subfolder name
    subdir_name = os.path.basename(os.path.normpath(patch_dir))
    out_pt = os.path.join(output_dir, f"{subdir_name}.emb.pt")
    if os.path.exists(out_pt):
        _log_fe(f"[PATCHES] {subdir_name}: embeddings exist, skipping -> {out_pt}")
        return

    # build in-memory "patch_list" from existing files, apply white filter in extractor
    patch_list = []
    for p in patch_files:
        try:
            rec = _make_patch_rec_from_file(p)
            patch_list.append(rec)
        except UnidentifiedImageError as e:
            _log_fe(f"[PATCHES][WARN] skip unreadable: {p} ({e})")
        except Exception as e:
            _log_fe(f"[PATCHES][WARN] skip {p}: {e}")

    _log_fe(f"[PATCHES] {subdir_name}: extracting embeddings (candidate patches={len(patch_list)})")
    embeddings = extract_embeddings_for_patches(patch_list, models, tmap, device)
    _log_fe(f"[PATCHES] {subdir_name}: kept {len(embeddings)} non-white patches")

    # save
    os.makedirs(output_dir, exist_ok=True)
    torch.save({"wsi_name": subdir_name, "patch_size": None, "patches": embeddings}, out_pt)
    _log_fe(f"[PATCHES] saved -> {out_pt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=False)

    # mode: update (your existing incremental add)
    p_upd = sub.add_parser("update", help="Update existing .emb.pt with new models")
    p_upd.add_argument("--input-dir", required=True)
    p_upd.add_argument("--output-dir", required=True)
    p_upd.add_argument("--add-models", default="uni2h,virchow,virchow2,provgigapath,conch")
    p_upd.add_argument("--hf-token", default=None)

    # mode: wsi (your existing WSI tiling + FE)
    p_wsi = sub.add_parser("wsi", help="Tile WSIs in a folder and extract")
    p_wsi.add_argument("--input-dir", required=True)
    p_wsi.add_argument("--output-dir", required=True)
    p_wsi.add_argument("--models", default="uni2h,virchow,virchow2,provgigapath,conch")
    p_wsi.add_argument("--hf-token", default=None)

    # mode: patches (NEW) – run on a single subfolder of patch images
    p_pch = sub.add_parser("patches", help="Extract from an existing patch folder (one subdir)")
    p_pch.add_argument("--patch-dir", required=True, help="One subdirectory under patch root")
    p_pch.add_argument("--output-dir", required=True, help="Where to write <subdir>.emb.pt")
    p_pch.add_argument("--models", default="uni2h,virchow,virchow2,provgigapath,conch")
    p_pch.add_argument("--hf-token", default=None)

    # defaults to 'patches' mode if none provided and old hard-coded paths existed
    args = parser.parse_args()

    if args.mode == "update":
        add_models = tuple(x.strip() for x in args.add_models.split(",") if x.strip())
        run_update_embeddings(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            add_models=add_models,
            hf_token=args.hf_token
        )
    elif args.mode == "wsi":
        use_models = tuple(x.strip() for x in args.models.split(",") if x.strip())
        run_feature_extraction_only(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            use_models=use_models,
            hf_token=args.hf_token
        )
    else:
        # default to patches mode
        use_models = tuple(x.strip() for x in args.models.split(",") if x.strip())
        run_feature_extraction_on_patch_dir(
            patch_dir=args.patch_dir,
            output_dir=args.output_dir,
            use_models=use_models,
            hf_token=args.hf_token
        )
