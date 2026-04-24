# %% [markdown]
# # Lesion Clustering — Trained / Classical / Intersection
#
# This script produces THREE independent sets of lesion clusters from every
# image, one per leaf-segmentation strategy:
#
#   TRAINED      — lesions detected inside the *trained segmenter's* leaf mask.
#   CLASSICAL    — lesions detected inside the *classical colour-heuristic* mask.
#   INTERSECTION — lesions in the region where BOTH masks agree (most reliable).
#
# Why three?
# ----------
# The trained model (mIoU 0.737) and the classical heuristic each make
# different errors.  The intersection is the conservative "we are sure this
# is a leaf pixel" region; trained-only lesions test what the model uniquely
# adds; classical-only lesions show what the heuristic catches that the model
# currently misses.  Comparing all three sets shows whether severity estimates
# are stable across methods.
#
# Outputs (per strategy)
# ----------------------
#   models/clusters/<strategy>/lesion_cluster_model.joblib
#   models/clusters/<strategy>/lesion_features_fit.csv
#   models/clusters/<strategy>/cluster_summary.csv
#
# Per image: one 6-panel figure (3 strategies × [mask overlay | disease class])
#
# Summary CSV has per-strategy prefixed columns:
#   trained_severity_pct, classical_severity_pct, intersection_severity_pct …
#
# Usage
# -----
#   # Fit all three cluster models:
#   python lesion_clustering.py --mode fit --data Data --out models/clusters
#
#   # Apply all three to produce figures + CSV:
#   python lesion_clustering.py --mode apply --data Data --out Results_v2 \
#       --cluster-model-dir models/clusters
#   # or
#   python lesion_clustering.py --mode apply --data Data --out Results_v2 --cluster-model-dir models/clusters
#
#   # Use only one strategy:
#   python lesion_clustering.py --mode apply --data Data --out Results_v2 \
#       --cluster-model-dir models/clusters --strategies trained


# %% Cell 1 — Imports and config
from rice_disease_analysis import (
    isolate_leaves as classical_isolate_leaves,
    detect_scene_mode, LOCATION_META, severity_band,
    MAX_SIDE, MIN_LEAF_AREA_FRAC,
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from skimage.feature import graycomatrix, graycoprops
from skimage import measure, morphology
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
import joblib
import cv2
import argparse
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

matplotlib.use("Agg")


try:
    import hdbscan
    HAVE_HDBSCAN = True
except ImportError:
    HAVE_HDBSCAN = False

try:
    import torch
    from train_leaf_segmenter import build_model as build_leaf_model, build_transforms
    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False

# ── Constants ──────────────────────────────────────────────────────────────
MIN_LESION_AREA_PX = 25
STRATEGIES = ["trained", "classical", "intersection"]

DISEASE_ADVICE = {
    "Blast":
        "Blast-like lesions (elongated, grey centre) detected. Drain the "
        "field briefly if flooded, avoid late nitrogen top-dressing, and "
        "consult your extension officer about a triazole/strobilurin spray "
        "if severity is rising.",
    "Brown spot":
        "Brown-spot-like lesions (round, dark brown) detected. Often a sign "
        "of potassium/silicon deficiency or weak seedlings. Check soil "
        "nutrients, remove severely affected leaves, and consider a "
        "mancozeb or propiconazole spray if severity is rising.",
    "Other":
        "Non-specific lesions detected. Scout the field, photograph a few "
        "close-ups of the worst leaves, and consult your extension officer "
        "before applying any spray.",
    "Healthy":
        "No significant lesions detected. Continue routine monitoring.",
}


# %% Cell 2 — Dual leaf segmentation
# Returns trained mask, classical mask, AND their pixel-wise intersection.
_LEAF_MODEL_CACHE: dict = {"model": None, "tf": None, "path": None}


def segment_leaf_all(img_bgr, model_path=None, img_size=512):
    """
    Run BOTH the trained segmenter and the classical heuristic on img_bgr.

    Returns a dict:
        trained      : uint8 (0/255) mask from the trained model
        classical    : uint8 (0/255) mask from the colour heuristic
        intersection : bitwise AND of trained & classical
                       (pixels where BOTH methods agree = leaf)
        trained_src  : str label describing the source
        classical_src: str label describing the source

    When the trained model is unavailable, trained == classical so that
    intersection == classical and downstream code still runs unchanged.
    """
    # ── Classical mask ──────────────────────────────────────────────────────
    mode = detect_scene_mode(img_bgr)
    classical_mask, _ = classical_isolate_leaves(img_bgr, mode)
    classical_src = f"classical_{mode}"

    # ── Trained mask ────────────────────────────────────────────────────────
    trained_mask = None
    trained_src = "trained_unavailable"

    if model_path and HAVE_TORCH and Path(model_path).exists():
        if _LEAF_MODEL_CACHE["path"] != str(model_path):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = build_leaf_model(num_classes=2).to(device)

            # Cross-platform checkpoint load: checkpoints pickled on Windows
            # contain WindowsPath objects that can't be unpickled on Linux.
            # Temporarily alias WindowsPath → PosixPath for the duration of
            # the torch.load() call only.
            import pathlib
            import platform
            _patched = False
            if platform.system() != "Windows":
                _orig_wp = pathlib.WindowsPath
                pathlib.WindowsPath = pathlib.PosixPath
                _patched = True
            try:
                ckpt = torch.load(model_path, map_location=device,
                                  weights_only=False)
            finally:
                if _patched:
                    pathlib.WindowsPath = _orig_wp

            model.load_state_dict(ckpt["model"])
        model.eval()
        _LEAF_MODEL_CACHE.update({
            "model": model, "device": device, "path": str(model_path),
            "tf": build_transforms(img_size, train=False),
        })
        model = _LEAF_MODEL_CACHE["model"]
        tf = _LEAF_MODEL_CACHE["tf"]
        device = _LEAF_MODEL_CACHE["device"]
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]
        dummy = np.zeros((H, W), np.uint8)
        tensor = tf(image=rgb, mask=dummy)["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor)["out"]
            pred_small = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
        trained_mask = cv2.resize(pred_small, (W, H),
                                  interpolation=cv2.INTER_NEAREST) * 255
        trained_src = "trained_segmenter"

    # Fallback: if model unavailable, treat trained as identical to classical.
    if trained_mask is None:
        trained_mask = classical_mask.copy()
        trained_src = f"trained_fallback({classical_src})"

    # ── Intersection (pixel-wise AND) ───────────────────────────────────────
    intersection_mask = cv2.bitwise_and(trained_mask, classical_mask)

    return {
        "trained":       trained_mask,
        "classical":     classical_mask,
        "intersection":  intersection_mask,
        "trained_src":   trained_src,
        "classical_src": classical_src,
    }


def segment_leaf(img_bgr, model_path=None, img_size=512):
    """
    Legacy single-mask entry point kept so other scripts importing this
    function continue to work without modification.
    Returns (mask, source_str) using the trained model when available.
    """
    seg = segment_leaf_all(img_bgr, model_path, img_size)
    if "fallback" not in seg["trained_src"]:
        return seg["trained"], seg["trained_src"]
    return seg["classical"], seg["classical_src"]


# %% Cell 3 — Lesion candidate detection (leaf-adaptive)
def find_lesion_candidates(img_bgr, leaf_mask):
    """
    Inside leaf_mask, flag pixels whose Lab colour deviates from the leaf's
    own median by more than ~2 robust standard deviations.
    Variety-agnostic — no fixed HSV thresholds.
    Returns uint8 binary candidate mask.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    leaf = leaf_mask > 0
    if leaf.sum() < 100:
        return np.zeros_like(leaf, dtype=np.uint8)

    Lm, Am, Bm = (np.median(L[leaf]), np.median(A[leaf]), np.median(B[leaf]))
    dL = L.astype(np.float32) - Lm
    dA = A.astype(np.float32) - Am
    dB = B.astype(np.float32) - Bm
    dist = np.sqrt(dL*dL + dA*dA + dB*dB)
    med = np.median(dist[leaf])
    mad = np.median(np.abs(dist[leaf] - med)) + 1e-6
    thr = med + 2.0 * 1.4826 * mad
    cand = ((dist > thr) & leaf).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN,  k, iterations=1)
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, k, iterations=1)
    return cand


# %% Cell 4 — Per-lesion feature extraction (23-D, unchanged)
FEATURE_NAMES = [
    "area_px", "perimeter_px", "eccentricity", "solidity",
    "circularity", "equiv_diameter", "aspect_ratio", "extent",
    "spindle_score",
    "dL_vs_leaf", "dA_vs_leaf", "dB_vs_leaf",
    "mean_saturation", "mean_hue",
    "glcm_contrast", "glcm_homogeneity", "glcm_energy", "glcm_correlation",
    "interior_L_std", "core_vs_margin_L_diff",
    "edge_distance_norm", "darkness_drop", "relative_area_of_leaf",
]


def extract_lesion_features(img_bgr, leaf_mask, cand_mask):
    """Returns (list-of-dicts, label_image). One dict per connected lesion."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    H, S, V = cv2.split(hsv)

    leaf = leaf_mask > 0
    if leaf.sum() == 0:
        return [], np.zeros(img_bgr.shape[:2], dtype=np.int32)

    Lm, Am, Bm = np.median(L[leaf]), np.median(A[leaf]), np.median(B[leaf])
    leaf_area_px = int(leaf.sum())
    edge_dist = cv2.distanceTransform(leaf.astype(np.uint8), cv2.DIST_L2, 5)
    max_edge = float(edge_dist.max()) + 1e-6

    bin_ = cv2.bitwise_and(cand_mask, leaf_mask)
    bin_ = morphology.remove_small_objects(
        bin_.astype(bool), min_size=MIN_LESION_AREA_PX).astype(np.uint8)
    labels = measure.label(bin_, connectivity=2)

    rows = []
    for region in measure.regionprops(labels, intensity_image=gray):
        if region.area < MIN_LESION_AREA_PX:
            continue
        minr, minc, maxr, maxc = region.bbox
        mask_local = labels[minr:maxr, minc:maxc] == region.label

        # Geometry
        area = float(region.area)
        perim = float(region.perimeter) if region.perimeter > 0 else 1.0
        circularity = 4.0 * np.pi * area / (perim * perim)
        bh, bw = maxr - minr, maxc - minc
        aspect = max(bh, bw) / max(1, min(bh, bw))

        # Spindle score
        spindle_score = 0.0
        try:
            orient = region.orientation
            ys_l, xs_l = np.where(mask_local)
            cy, cx = mask_local.shape[0] / 2, mask_local.shape[1] / 2
            cos_t = np.cos(-orient + np.pi / 2)
            sin_t = np.sin(-orient + np.pi / 2)
            rx = (xs_l - cx) * cos_t - (ys_l - cy) * sin_t
            ry = (xs_l - cx) * sin_t + (ys_l - cy) * cos_t
            if len(rx) > 20:
                bins = np.linspace(rx.min(), rx.max(), 6)
                widths = []
                for i in range(5):
                    m = (rx >= bins[i]) & (rx < bins[i + 1])
                    if m.sum() > 2:
                        widths.append(ry[m].max() - ry[m].min())
                if len(widths) == 5:
                    middle = widths[2] + 1e-6
                    ends = 0.5 * (widths[0] + widths[4]) + 1e-6
                    spindle_score = float(middle / ends)
        except Exception:
            spindle_score = 0.0

        # Colour
        ys, xs = np.where(mask_local)
        ys += minr
        xs += minc
        dL = float(np.mean(L[ys, xs].astype(np.float32) - Lm))
        dA = float(np.mean(A[ys, xs].astype(np.float32) - Am))
        dB = float(np.mean(B[ys, xs].astype(np.float32) - Bm))
        mean_s = float(np.mean(S[ys, xs]))
        mean_h = float(np.mean(H[ys, xs]))

        # Interior structure
        lm_full = np.zeros_like(leaf_mask, dtype=np.uint8)
        lm_full[ys, xs] = 1
        erode_r = max(1, int(np.sqrt(area) / 6))
        k_e = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                        (2*erode_r+1, 2*erode_r+1))
        core = cv2.erode(lm_full, k_e, iterations=1)
        margin = cv2.bitwise_and(lm_full, cv2.bitwise_not(core))
        interior_L_std = float(np.std(L[ys, xs])) if len(ys) > 1 else 0.0
        if core.sum() > 5 and margin.sum() > 5:
            core_vs_margin_L_diff = (float(np.mean(L[core > 0]))
                                     - float(np.mean(L[margin > 0])))
        else:
            core_vs_margin_L_diff = 0.0

        # Texture (GLCM)
        patch = gray[minr:maxr, minc:maxc]
        if patch.size >= 16:
            q = (patch // 8).astype(np.uint8)
            glcm = graycomatrix(q, distances=[1], angles=[0],
                                levels=32, symmetric=True, normed=True)
            contrast = float(graycoprops(glcm, "contrast")[0, 0])
            homogeneity = float(graycoprops(glcm, "homogeneity")[0, 0])
            energy = float(graycoprops(glcm, "energy")[0, 0])
            correlation = float(graycoprops(glcm, "correlation")[0, 0])
        else:
            contrast = homogeneity = energy = correlation = np.nan

        edge_d = float(np.mean(edge_dist[ys, xs])) / max_edge
        darkness_drop = float(Lm - np.mean(L[ys, xs]))
        rel_area = area / leaf_area_px

        rows.append({
            "lesion_id_in_image":    int(region.label),
            "bbox_minr": int(minr),  "bbox_minc": int(minc),
            "bbox_maxr": int(maxr),  "bbox_maxc": int(maxc),
            "centroid_y":            float(region.centroid[0]),
            "centroid_x":            float(region.centroid[1]),
            "area_px":               area,
            "perimeter_px":          perim,
            "eccentricity":          float(region.eccentricity),
            "solidity":              float(region.solidity),
            "circularity":           float(circularity),
            "equiv_diameter":        float(region.equivalent_diameter),
            "aspect_ratio":          float(aspect),
            "extent":                float(region.extent),
            "spindle_score":         float(spindle_score),
            "dL_vs_leaf":            dL,
            "dA_vs_leaf":            dA,
            "dB_vs_leaf":            dB,
            "mean_saturation":       mean_s,
            "mean_hue":              mean_h,
            "glcm_contrast":         contrast,
            "glcm_homogeneity":      homogeneity,
            "glcm_energy":           energy,
            "glcm_correlation":      correlation,
            "interior_L_std":        float(interior_L_std),
            "core_vs_margin_L_diff": float(core_vs_margin_L_diff),
            "edge_distance_norm":    edge_d,
            "darkness_drop":         darkness_drop,
            "relative_area_of_leaf": rel_area,
        })
    return rows, labels


# %% Cell 5 — Cluster model (K=3 KMeans + heuristic disease mapping)
# override after expert review: {0:"Blast", ...}
DISEASE_MAPPING_OVERRIDE = None

BLAST_SIGNATURE = {
    "eccentricity":          "+",
    "spindle_score":         "+",
    "aspect_ratio":          "+",
    "area_px":               "+",
    "interior_L_std":        "+",
    "core_vs_margin_L_diff": "+",
    "circularity":           "-",
}
BROWN_SPOT_SIGNATURE = {
    "circularity":   "+",
    "solidity":      "+",
    "eccentricity":  "-",
    "spindle_score": "-",
    "aspect_ratio":  "-",
    "dA_vs_leaf":    "+",
    "darkness_drop": "+",
    "area_px":       "-",
}
DISEASE_NAMES = ["Blast", "Brown spot", "Other"]


def _signature_score(cluster_medians, signature):
    score = 0.0
    for feat, sign in signature.items():
        if feat not in cluster_medians:
            continue
        score += cluster_medians[feat] if sign == "+" else - \
            cluster_medians[feat]
    return score


def heuristic_cluster_to_disease(summary_df):
    feats = [c for c in summary_df.columns
             if c not in ("cluster_id", "n_lesions")]
    X = summary_df[feats].values.astype(np.float32)
    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-6
    Z = (X - mu) / sd
    z_df = summary_df.copy()
    z_df[feats] = Z

    mapping, used = {}, set()

    def pick(sig, label):
        best_cid, best_score = None, -1e9
        for _, row in z_df.iterrows():
            cid = int(row["cluster_id"])
            if cid in used:
                continue
            s = _signature_score(row, sig)
            if s > best_score:
                best_score, best_cid = s, cid
        if best_cid is not None:
            mapping[best_cid] = label
            used.add(best_cid)

    pick(BLAST_SIGNATURE,      "Blast")
    pick(BROWN_SPOT_SIGNATURE, "Brown spot")
    for _, row in z_df.iterrows():
        cid = int(row["cluster_id"])
        if cid not in used:
            mapping[cid] = "Other"
    return mapping


def fit_cluster_model(feature_df, n_components_pca=8, k=3):
    X = feature_df[FEATURE_NAMES].values.astype(np.float32)
    col_median = np.nanmedian(X, axis=0)
    nan_idx = np.where(np.isnan(X))
    X[nan_idx] = np.take(col_median, nan_idx[1])

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    n_comp = min(n_components_pca, Xs.shape[1], max(1, Xs.shape[0] - 1))
    pca = PCA(n_components=n_comp, random_state=42).fit(Xs)
    Xp = pca.transform(Xs)
    eff_k = min(k, max(2, len(Xp) // 2))
    kmeans = KMeans(n_clusters=eff_k, n_init=20, random_state=42).fit(Xp)

    feature_df = feature_df.copy()
    feature_df["cluster_id"] = kmeans.labels_
    summary = (feature_df.groupby("cluster_id")[FEATURE_NAMES]
               .median().reset_index())
    summary["n_lesions"] = feature_df.groupby("cluster_id").size().values

    cid_to_disease = (dict(DISEASE_MAPPING_OVERRIDE)
                      if DISEASE_MAPPING_OVERRIDE is not None
                      else heuristic_cluster_to_disease(summary))

    feature_df["disease_label"] = feature_df["cluster_id"].map(cid_to_disease)
    summary["disease_label"] = summary["cluster_id"].map(cid_to_disease)

    return {
        "scaler": scaler, "pca": pca, "clusterer": kmeans,
        "algo":   f"kmeans_k{eff_k}",
        "feature_names":      FEATURE_NAMES,
        "col_median_for_nan": col_median,
        "summary":            summary,
        "cid_to_disease":     cid_to_disease,
    }, feature_df


def apply_cluster_model(feature_df, model):
    X = feature_df[model["feature_names"]].values.astype(np.float32)
    nan_idx = np.where(np.isnan(X))
    X[nan_idx] = np.take(model["col_median_for_nan"], nan_idx[1])
    Xs = model["scaler"].transform(X)
    Xp = model["pca"].transform(Xs)
    labels = model["clusterer"].predict(Xp)
    feature_df = feature_df.copy()
    feature_df["cluster_id"] = labels
    feature_df["disease_label"] = (
        pd.Series(labels)
        .map(model.get("cid_to_disease", {}))
        .fillna("Other")
        .values
    )
    return feature_df


def format_cluster_label(cluster_id, cid_to_disease=None):
    if cid_to_disease and cluster_id in cid_to_disease:
        return cid_to_disease[cluster_id]
    return "Other" if cluster_id == -1 else f"Type-{int(cluster_id)+1}"


def save_cluster_model(model, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_cluster_model(path):
    return joblib.load(path)


# %% Cell 6 — Colours
DISEASE_COLORS_BGR = {
    "Blast":      (180, 105, 255),   # pink/magenta
    "Brown spot": (0,    90, 200),   # brown/orange
    "Other":      (200, 200, 200),   # grey
}
# Tint colour used for each strategy's mask overlay panel.
STRATEGY_OVERLAY_BGR = {
    "trained":      (220, 120,   0),   # blue tint
    "classical":    (0, 200,   0),   # green tint
    "intersection": (0, 200, 220),   # yellow tint
}


def color_for_disease(label_str):
    return DISEASE_COLORS_BGR.get(label_str, DISEASE_COLORS_BGR["Other"])


def color_for_cluster(label_str):   # backward-compat alias
    return color_for_disease(label_str)


# %% Cell 7 — Per-image figure (3 strategies × 2 panels, plus original)
def _mask_overlay(img_bgr, mask, color_bgr, bg_dim=0.35):
    """Colour-tint the masked region; dim the rest."""
    layer = np.zeros_like(img_bgr)
    layer[:] = color_bgr
    result = np.where(
        mask[..., None] > 0,
        cv2.addWeighted(img_bgr, 0.4, layer, 0.6, 0),
        (img_bgr * bg_dim).astype(np.uint8),
    )
    return result


def _draw_disease_contours(img_bgr, labels_img, lesion_df):
    """Outline each lesion in its disease colour. Returns (viz, count_dict)."""
    viz = img_bgr.copy()
    counts = {}
    if len(lesion_df):
        for _, row in lesion_df.iterrows():
            lid = int(row["lesion_id_in_image"])
            lbl = str(row.get("disease_label", "Other"))
            color = color_for_disease(lbl)
            m = (labels_img == lid).astype(np.uint8) * 255
            cts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(viz, cts, -1, color, 2)
            counts[lbl] = counts.get(lbl, 0) + 1
    return viz, counts


def _disease_legend_patches(counts):
    patches = []
    for lbl in DISEASE_NAMES:
        b, g, r = color_for_disease(lbl)
        patches.append(Patch(facecolor=(r/255, g/255, b/255),
                             label=f"{lbl} (n={counts.get(lbl, 0)})"))
    return patches


def save_per_image_cluster_figure(
        img_bgr,
        masks,           # {strategy: uint8 mask}
        labels_imgs,     # {strategy: label image}
        lesion_dfs,      # {strategy: DataFrame}
        out_path,
        title,
        strategies_used=None,
):
    """
    Layout (n_strategies rows × 3 columns):
      col 0 — original image (first row only; remaining rows blank)
      col 1 — mask overlay for that strategy
      col 2 — disease-class outlines for that strategy

    Each strategy row shows its own severity and dominant disease.
    """
    if strategies_used is None:
        strategies_used = [s for s in STRATEGIES if s in masks]

    n_rows = max(len(strategies_used), 1)
    fig, axes = plt.subplots(n_rows, 3,
                             figsize=(15, 5 * n_rows),
                             squeeze=False)

    # Column 0: show original only in row 0, hide the rest.
    axes[0, 0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Original image", fontsize=11)
    axes[0, 0].axis("off")
    for r in range(1, n_rows):
        axes[r, 0].axis("off")

    for row_idx, strat in enumerate(strategies_used):
        mask = masks.get(strat,      np.zeros(img_bgr.shape[:2], np.uint8))
        lbl_img = labels_imgs.get(strat, np.zeros(img_bgr.shape[:2], np.int32))
        ldf = lesion_dfs.get(strat,  pd.DataFrame())
        col_bgr = STRATEGY_OVERLAY_BGR.get(strat, (200, 200, 200))

        leaf_px = int((mask > 0).sum())
        n_les = len(ldf)
        les_area = int(ldf["area_px"].sum()) if n_les else 0
        sev = 100.0 * les_area / max(leaf_px, 1)
        band, _ = severity_band(sev)

        # --- Mask overlay panel ---
        mask_viz = _mask_overlay(img_bgr, mask, col_bgr)
        axes[row_idx, 1].imshow(cv2.cvtColor(mask_viz, cv2.COLOR_BGR2RGB))
        axes[row_idx, 1].set_title(
            f"[{strat.upper()}] Leaf mask\n"
            f"{leaf_px:,} px  |  {n_les} lesions  |  "
            f"Severity {sev:.1f}% ({band})",
            fontsize=10)
        axes[row_idx, 1].axis("off")

        # --- Disease-class panel ---
        if n_les and "disease_label" in ldf.columns:
            dominant = (ldf.groupby("disease_label")["area_px"].sum()
                        .sort_values(ascending=False).index[0])
        else:
            dominant = "Healthy"
        dis_viz, dcounts = _draw_disease_contours(img_bgr, lbl_img, ldf)
        axes[row_idx, 2].imshow(cv2.cvtColor(dis_viz, cv2.COLOR_BGR2RGB))
        axes[row_idx, 2].set_title(
            f"[{strat.upper()}] Disease classes\n"
            f"Dominant: {dominant}", fontsize=10)
        axes[row_idx, 2].axis("off")
        axes[row_idx, 2].legend(
            handles=_disease_legend_patches(dcounts),
            loc="lower center", bbox_to_anchor=(0.5, -0.22),
            ncol=3, fontsize=8, frameon=False)

    fig.suptitle(title, fontsize=12, y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# %% Cell 8 — Constrained re-clustering hook (future expert feedback)
def recluster_with_constraints(feature_df, must_link=None, cannot_link=None,
                               n_components_pca=8, k=None):
    """
    Stub — wire in COPKMeans once expert must-link/cannot-link pairs exist.
    Install 'active-semi-supervised-clustering' and replace KMeans with
    COPKMeans using the must_link / cannot_link pair lists.
    """
    raise NotImplementedError(
        "Constrained clustering will be wired in once expert labels exist. "
        "Install 'active-semi-supervised-clustering' and use COPKMeans.")


# %% Cell 9 — Per-image processing (all three strategies in one pass)
def process_image(img_path, location_key, leaf_model_path,
                  cluster_models, strategies=None):
    """
    Full pipeline for one image across all requested strategies.

    cluster_models : {strategy: loaded_model_or_None}
    strategies     : list subset of STRATEGIES (default = all three)

    Returns (summary_dict, all_lesion_records, viz_dict).
    """
    if strategies is None:
        strategies = STRATEGIES

    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(str(img_path))
    H0, W0 = img.shape[:2]
    scale = MAX_SIDE / max(H0, W0) if max(H0, W0) > MAX_SIDE else 1.0
    if scale < 1.0:
        img = cv2.resize(img, (int(W0*scale), int(H0*scale)),
                         interpolation=cv2.INTER_AREA)

    # One segmentation call produces all three masks.
    seg = segment_leaf_all(img, model_path=leaf_model_path)
    masks = {s: seg[s] for s in STRATEGIES}

    labels_imgs = {}
    lesion_dfs = {}
    all_lesion_records = []

    for strat in strategies:
        mask = masks[strat]
        cand = find_lesion_candidates(img, mask)
        rows, lbl_img = extract_lesion_features(img, mask, cand)
        labels_imgs[strat] = lbl_img

        df = pd.DataFrame(rows)
        if len(df):
            cl = cluster_models.get(strat)
            df = apply_cluster_model(df, cl) if cl is not None else (
                df.assign(cluster_id=-1, disease_label="Other"))
        else:
            df["cluster_id"] = pd.Series(dtype=int)
            df["disease_label"] = pd.Series(dtype=str)

        df["image_name"] = img_path.name
        df["location_key"] = location_key
        df["mask_strategy"] = strat
        lesion_dfs[strat] = df
        all_lesion_records.extend(df.to_dict(orient="records"))

    # ── Build summary (per-strategy prefixed columns) ──────────────────────
    meta = LOCATION_META.get(location_key, {})
    total_px = img.shape[0] * img.shape[1]
    summary: dict = {
        "image_name":    img_path.name,
        "location_key":  location_key,
        "site":          meta.get("site", "Unknown"),
        "latitude":      meta.get("lat",  np.nan),
        "longitude":     meta.get("lon",  np.nan),
        "date":          meta.get("date", ""),
        "total_area_px": total_px,
        "trained_src":   seg["trained_src"],
        "classical_src": seg["classical_src"],
    }

    for strat in strategies:
        mask = masks[strat]
        ldf = lesion_dfs[strat]
        leaf_px = int((mask > 0).sum())
        les_area = int(ldf["area_px"].sum()) if len(ldf) else 0
        sev = 100.0 * les_area / max(leaf_px, 1)
        band, _ = severity_band(sev)

        # Per-disease columns — always emit all 3, even when count = 0.
        per_n = {f"{strat}_n_{d}": 0 for d in DISEASE_NAMES}
        per_area = {f"{strat}_area_{d}": 0 for d in DISEASE_NAMES}
        per_pct = {f"{strat}_pct_{d}": 0.0 for d in DISEASE_NAMES}
        if len(ldf) and "disease_label" in ldf.columns:
            for lbl, sub in ldf.groupby("disease_label"):
                if lbl not in DISEASE_NAMES:
                    continue
                per_n[f"{strat}_n_{lbl}"] = int(len(sub))
                per_area[f"{strat}_area_{lbl}"] = int(sub["area_px"].sum())
                per_pct[f"{strat}_pct_{lbl}"] = (
                    100.0 * sub["area_px"].sum() / max(leaf_px, 1))

        dominant = (
            max(DISEASE_NAMES, key=lambda d: per_area[f"{strat}_area_{d}"])
            if les_area > 0 else "Healthy"
        )
        summary.update({
            f"{strat}_leaf_px":          leaf_px,
            f"{strat}_leaf_frac_pct":    100.0 * leaf_px / total_px,
            f"{strat}_n_lesions":        len(ldf),
            f"{strat}_lesion_area_px":   les_area,
            f"{strat}_severity_pct":     sev,
            f"{strat}_severity_band":    band,
            f"{strat}_dominant_disease": dominant,
            f"{strat}_farmer_advice":    DISEASE_ADVICE.get(dominant, ""),
            **per_n, **per_area, **per_pct,
        })

    # Mask agreement metrics
    t_px = int((masks["trained"] > 0).sum())
    c_px = int((masks["classical"] > 0).sum())
    i_px = int((masks["intersection"] > 0).sum())
    summary["intersection_pct_of_trained"] = 100.0 * i_px / max(t_px, 1)
    summary["intersection_pct_of_classical"] = 100.0 * i_px / max(c_px, 1)
    summary["trained_classical_agreement_pct"] = (
        100.0 * 2 * i_px / max(t_px + c_px, 1))  # Dice-style agreement

    viz = {
        "img":         img,
        "masks":       masks,
        "labels_imgs": labels_imgs,
        "lesion_dfs":  lesion_dfs,
    }
    return summary, all_lesion_records, viz


# %% Cell 10 — Dataset walker
def discover_images(data_root):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    items = []
    for sub in sorted(p for p in Path(data_root).iterdir() if p.is_dir()):
        for img in sorted(sub.iterdir()):
            if img.suffix.lower() in exts:
                items.append((img, sub.name))
    return items


# %% Cell 11 — FIT mode: one cluster model per strategy
def run_fit(data_root, out_dir, leaf_model_path, strategies=None):
    """
    Walk all images, run segmentation, extract features, fit K=3 cluster
    model independently for EACH strategy, and save to
        out_dir/<strategy>/lesion_cluster_model.joblib
    """
    if strategies is None:
        strategies = STRATEGIES
    print(f"[FIT] Strategies : {strategies}")
    print(f"[FIT] Scanning   : {data_root}")

    items = discover_images(data_root)
    if not items:
        print("No images found.")
        return

    all_rows = {s: [] for s in strategies}

    for img_path, loc in items:
        print(f"  {loc}/{img_path.name}")
        img = cv2.imread(str(img_path))
        if img is None:
            print("    [skip] cannot read")
            continue
        H0, W0 = img.shape[:2]
        scale = MAX_SIDE / max(H0, W0) if max(H0, W0) > MAX_SIDE else 1.0
        if scale < 1.0:
            img = cv2.resize(img, (int(W0*scale), int(H0*scale)),
                             interpolation=cv2.INTER_AREA)

        seg = segment_leaf_all(img, model_path=leaf_model_path)
        masks = {s: seg[s] for s in STRATEGIES}

        for strat in strategies:
            mask = masks[strat]
            cand = find_lesion_candidates(img, mask)
            rows, _ = extract_lesion_features(img, mask, cand)
            for r in rows:
                r["image_name"] = img_path.name
                r["location_key"] = loc
                r["mask_strategy"] = strat
            all_rows[strat].extend(rows)
            print(f"    {strat}: {len(rows)} lesions")

    out_dir = Path(out_dir)
    fitted_models = {}

    for strat in strategies:
        rows = all_rows[strat]
        if not rows:
            print(f"\n[FIT] {strat}: no lesions — skipping.")
            continue

        df = pd.DataFrame(rows)
        print(f"\n[FIT] {strat}: {len(df)} lesions — fitting K=3 ...")
        model, df = fit_cluster_model(df)
        fitted_models[strat] = model

        print(f"  Algorithm: {model['algo']}   "
              f"Classes: {df['disease_label'].nunique()}")
        for cid, name in sorted(model["cid_to_disease"].items()):
            n = int((df["cluster_id"] == cid).sum())
            print(f"    cluster {cid} → {name}  (n={n})")

        strat_dir = out_dir / strat
        strat_dir.mkdir(parents=True, exist_ok=True)
        save_cluster_model(model, strat_dir / "lesion_cluster_model.joblib")
        df.to_csv(strat_dir / "lesion_features_fit.csv",  index=False)
        model["summary"].to_csv(strat_dir / "cluster_summary.csv", index=False)
        print(f"  Saved → {strat_dir}/")
        print(model["summary"][
            ["cluster_id", "disease_label", "n_lesions",
             "area_px", "eccentricity", "circularity",
             "spindle_score", "core_vs_margin_L_diff",
             "dA_vs_leaf", "darkness_drop"]
        ].to_string(index=False))

    print(f"\n[FIT] Done. Models in {out_dir}/<strategy>/")
    return fitted_models


# %% Cell 12 — APPLY mode
def run_apply(data_root, out_dir, leaf_model_path, cluster_model_dir,
              strategies=None):
    """
    Apply the saved cluster model(s) to every image.

    cluster_model_dir: directory whose sub-folders are named after each
                       strategy and contain lesion_cluster_model.joblib.
    """
    if strategies is None:
        strategies = STRATEGIES

    items = discover_images(data_root)
    if not items:
        print("No images found.")
        return

    cluster_model_dir = Path(cluster_model_dir)
    cluster_models = {}
    for strat in strategies:
        path = cluster_model_dir / strat / "lesion_cluster_model.joblib"
        if path.exists():
            cluster_models[strat] = load_cluster_model(path)
            print(f"Loaded cluster model [{strat}]: {path}")
        else:
            cluster_models[strat] = None
            print(f"[WARN] No cluster model for '{strat}' at {path} — "
                  "will label all lesions 'Other'.")

    out_dir = Path(out_dir)
    all_summary, all_lesions = [], []

    for img_path, loc in items:
        try:
            summary, lesion_records, viz = process_image(
                img_path, loc, leaf_model_path, cluster_models, strategies)
        except Exception as e:
            print(f"  [ERROR] {img_path.name}: {e}")
            continue

        loc_out = out_dir / loc
        loc_out.mkdir(parents=True, exist_ok=True)
        fig_path = loc_out / f"{img_path.stem}_clusters.png"

        save_per_image_cluster_figure(
            viz["img"],
            viz["masks"],
            viz["labels_imgs"],
            viz["lesion_dfs"],
            out_path=fig_path,
            title=(f"{img_path.name} | "
                   f"{summary.get('site', '?')} | "
                   f"{summary.get('date', '?')}"),
            strategies_used=strategies,
        )

        all_summary.append(summary)
        all_lesions.extend(lesion_records)

        parts = []
        for strat in strategies:
            sev = summary.get(f"{strat}_severity_pct", 0)
            dom = summary.get(f"{strat}_dominant_disease", "-")
            n = summary.get(f"{strat}_n_lesions",  0)
            parts.append(f"{strat[:3]}:sev={sev:.1f}%,dom={dom},n={n}")
        agree = summary.get("trained_classical_agreement_pct", 0)
        print(f"  [OK] {loc}/{img_path.name}  "
              + "  |  ".join(parts)
              + f"  |  T∩C_agree={agree:.1f}%")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sum_csv = out_dir / f"report_summary_{stamp}.csv"
    les_csv = out_dir / f"report_lesions_{stamp}.csv"
    pd.DataFrame(all_summary).to_csv(sum_csv, index=False)
    pd.DataFrame(all_lesions).to_csv(les_csv, index=False)
    print(f"\n[APPLY] Summary CSV : {sum_csv}")
    print(f"[APPLY] Lesion CSV  : {les_csv}")
    print(f"[APPLY] Figures in  : {out_dir}/<location>/")


# %% Cell 13 — CLI
def main():
    ap = argparse.ArgumentParser(
        description=(
            "Lesion clustering — three independent mask strategies: "
            "trained / classical / intersection."))
    ap.add_argument("--mode", choices=["fit", "apply"], required=True,
                    help="fit: build cluster models.  apply: score images.")
    ap.add_argument("--data",              default="Data",
                    help="Data root with date-subfolders.")
    ap.add_argument("--out",               default="Results_v2")
    ap.add_argument("--leaf-model",
                    default="models/leaf_segmenter_best.pt")
    ap.add_argument("--cluster-model-dir", default="models/clusters",
                    help="Directory containing trained/classical/intersection "
                         "sub-folders, each with lesion_cluster_model.joblib")
    ap.add_argument("--cluster-model",     default=None,
                    help="[legacy] single joblib path — ignored in multi-strategy mode.")
    ap.add_argument("--strategies",        default="all",
                    help="Comma-separated: trained,classical,intersection  "
                         "(default: all three)")
    args = ap.parse_args()

    strategies = (STRATEGIES if args.strategies == "all"
                  else [s.strip() for s in args.strategies.split(",")
                        if s.strip() in STRATEGIES])
    if not strategies:
        print("No valid strategies specified.  "
              "Choose from: trained, classical, intersection")
        return

    print(f"Torch available  : {HAVE_TORCH}")
    print(f"HDBSCAN available: {HAVE_HDBSCAN}")
    leaf_model = args.leaf_model if Path(args.leaf_model).exists() else None
    if leaf_model:
        print(f"Leaf model       : {leaf_model}")
    else:
        print("No leaf model found — 'trained' strategy will mirror classical.")

    if args.mode == "fit":
        out = args.out if args.out != "Results_v2" else "models/clusters"
        run_fit(args.data, out, leaf_model, strategies)
    else:
        run_apply(args.data, args.out, leaf_model,
                  args.cluster_model_dir, strategies)


if __name__ == "__main__":
    main()
