# %% [markdown]
# # Rice Disease Monitoring System
# Per-image, per-location analysis pipeline for field-captured rice images.
#
# **Folder structure expected:**
# ```
# Data/
#   Rice_Diseases_Sep_26_2025/   # location 1 (date folder)
#   Rice_Diseases_Dec_02_2025/   # location 2
#   Rice_Diseases_Sep_03_2024/   # location 3
# Results/                       # auto-created, one sub-folder per date/location
# ```
# The script works for 1 image in 1 folder, or many images across many folders.

# %% Cell 1 — Imports and configuration
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from skimage import measure, morphology, color
from skimage.feature import graycomatrix, graycoprops
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ---- Paths (edit these two lines if your project lives elsewhere) ----
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "Data"
RESULTS_DIR = PROJECT_ROOT / "Results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---- Location metadata (matches your three folders) ----
LOCATION_META = {
    "Rice_Diseases_Sep_26_2025": {
        "site": "ICAR-RCER Plandu Farm B, Ranchi",
        "lat":  23 + 17/60,
        "lon":  85 + 24/60,
        "date": "2025-09-26",
    },
    "Rice_Diseases_Dec_02_2025": {
        "site": "ICAR-NBPGR Regional Station, Ranchi",
        "lat":  23 + 16/60,
        "lon":  85 + 20/60,
        "date": "2025-12-02",
    },
    "Rice_Diseases_Sep_03_2024": {
        "site": "ICAR-IIAB, Farm B, Ranchi",
        "lat":  23 + 16/60,
        "lon":  85 + 20/60,
        "date": "2024-09-03",
    },
}

# ---- Processing parameters ----
MAX_SIDE = 1600           # downscale large photos for speed; keeps good detail
MIN_LEAF_AREA_FRAC = 0.0005   # ignore leaf blobs smaller than this fraction of image
# ignore lesion specks smaller than this (in px, after resize)
MIN_LESION_AREA_PX = 25

print("Configuration loaded.")
print(f"Data dir   : {DATA_DIR}")
print(f"Results dir: {RESULTS_DIR}")


# %% Cell 2 — Scene mode detection and focus mask
def detect_scene_mode(img_bgr):
    """
    Decide whether the image is:
      - 'specimen' : a cut leaf/panicle on a neutral backdrop (lab photo, paper, cloth)
      - 'field'    : a live plant photographed in situ
    This matters because a dry senesced specimen is NOT diseased — it's mature/harvested.
    In specimen mode we switch off the "brown = lesion" assumption.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    # Neutral backdrop fraction: low-saturation pixels occupying >40% of frame
    neutral = ((S < 35) & ((V > 160) | (V < 60))).sum() / H.size
    # Plant fraction: green + yellow hues
    plant = (((H >= 20) & (H <= 95)) & (S > 40)).sum() / H.size
    if neutral > 0.40 and plant < 0.25:
        return "specimen"
    return "field"


def focus_mask(img_bgr, tile=32):
    """
    Return an HxW float map in [0,1] of local sharpness.
    Computed by block-wise variance of the Laplacian — high variance = in-focus.
    Used to downweight blurred-bokeh background pixels.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    lap2 = lap * lap
    # Local mean of squared Laplacian via box filter = local variance proxy
    local = cv2.boxFilter(lap2, ddepth=-1, ksize=(tile, tile), normalize=True)
    # Normalise robustly (95th pct → 1.0)
    p95 = np.percentile(local, 95) + 1e-6
    f = np.clip(local / p95, 0.0, 1.0)
    return f


# %% Cell 3 — Leaf isolation (removes hand, soil, tools, sky, blurred background)
def isolate_leaves(img_bgr, mode):
    """
    Separate rice-leaf / straw pixels from everything else.

    In 'field' mode: remove skin, sky, tools, blurred bokeh background.
    In 'specimen' mode: keep anything that isn't the neutral backdrop, because
      the photographer has already done the isolation — we just need to pick up
      the specimen silhouette regardless of its colour (including dry straw).

    Returns (leaf_mask, subject_mask) — both uint8 (0/255).
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    H, S, V = cv2.split(hsv)

    if mode == "specimen":
        # The specimen sits on a near-uniform backdrop, so Otsu on luminance
        # distance from the backdrop gives a much better silhouette than
        # colour thresholding (which fails for pale tan straw on off-white paper).
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        L = lab[:, :, 0]
        # Estimate backdrop as the modal luminance (most common value).
        hist = cv2.calcHist([L], [0], None, [256], [0, 256]).flatten()
        bg_L = int(np.argmax(hist))
        # Pixels whose L differs noticeably from the backdrop are the specimen.
        dist = cv2.absdiff(L, np.full_like(L, bg_L))
        # Require a minimum absolute luminance gap so shadows aren't counted.
        clear = (dist > 25).astype(np.uint8) * 255
        _, otsu = cv2.threshold(
            dist, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        spec = cv2.bitwise_and(clear, otsu)
        # Also include pixels that are clearly coloured (any hue with decent sat),
        # which catches the darker panicle grains.
        hsv_s = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)[:, :, 1]
        coloured = (hsv_s > 55).astype(np.uint8) * 255
        leaf_like = cv2.bitwise_or(spec, coloured)

        # Remove low-saturation "shadow on paper" zones that slipped through:
        # the true specimen always has either colour or a sharp luminance step,
        # but the shadow is a smooth gradient of low-sat grey.
        shadow = ((hsv_s < 25) & (dist < 60)).astype(np.uint8) * 255
        leaf_like = cv2.bitwise_and(leaf_like, cv2.bitwise_not(shadow))

        # Remove skin (specimens are sometimes held by hand).
        skin = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
        leaf_like = cv2.bitwise_and(leaf_like, cv2.bitwise_not(skin))

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        leaf_like = cv2.morphologyEx(
            leaf_like, cv2.MORPH_OPEN,  k, iterations=1)
        leaf_like = cv2.morphologyEx(
            leaf_like, cv2.MORPH_CLOSE, k, iterations=2)
    else:
        # Field mode: colour-based plant mask (green / yellow / brown tissue).
        green = cv2.inRange(hsv, (25, 25, 25), (95, 255, 255))
        yellow = cv2.inRange(hsv, (15, 25, 60), (34, 255, 255))
        brown = cv2.inRange(hsv, (5,  20, 40), (28, 255, 230))
        leaf_like = cv2.bitwise_or(cv2.bitwise_or(green, yellow), brown)

        # Remove hand (skin), sky, bright/dark tools.
        skin = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
        bright_gray = cv2.inRange(hsv,   (0, 0, 180),  (179, 40, 255))
        deep_black = cv2.inRange(hsv,   (0, 0, 0),    (179, 60, 35))
        for rej in (skin, bright_gray, deep_black):
            leaf_like = cv2.bitwise_and(leaf_like, cv2.bitwise_not(rej))

        # Remove blurred out-of-focus background (bokeh). This is the key fix
        # for whole-plant photos against a paddy-field backdrop.
        fmap = focus_mask(img_bgr, tile=32)
        in_focus = (fmap > 0.15).astype(np.uint8) * 255
        leaf_like = cv2.bitwise_and(leaf_like, in_focus)

        k_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        leaf_like = cv2.morphologyEx(
            leaf_like, cv2.MORPH_OPEN,  k_small, iterations=1)
        leaf_like = cv2.morphologyEx(
            leaf_like, cv2.MORPH_CLOSE, k_close, iterations=2)

    # Drop tiny blobs (noise).
    min_area = int(MIN_LEAF_AREA_FRAC * leaf_like.size)
    nlab, lab, stats, _ = cv2.connectedComponentsWithStats(
        leaf_like, connectivity=8)
    leaf_mask = np.zeros_like(leaf_like)
    for i in range(1, nlab):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            leaf_mask[lab == i] = 255

    # Subject mask = largest 3 connected components (the leaves under inspection).
    nlab, lab, stats, _ = cv2.connectedComponentsWithStats(
        leaf_mask, connectivity=8)
    subject = np.zeros_like(leaf_mask)
    if nlab > 1:
        areas = sorted(((i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, nlab)),
                       key=lambda x: -x[1])
        for i, _ in areas[:3]:
            subject[lab == i] = 255

    return leaf_mask, subject


# %% Cell 4 — Lesion detection and classification within the leaf mask
def detect_lesions(img_bgr, leaf_mask, mode):
    """
    Within leaf tissue, assign each pixel to one of:
      1 = healthy      (vivid green)
      2 = chlorotic    (yellow, early disease / nutrient stress)
      3 = brown lesion (necrotic brown-spot, blast, sheath-blight)
      4 = dark lesion  (very dark necrotic core, bacterial streak)
      5 = senesced     (mature/harvested dry tissue — NOT a disease)

    In 'specimen' mode the tan/brown mature-straw colour is mapped to class 5
    instead of class 3, so post-harvest specimens aren't flagged as diseased.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    H, S, V = cv2.split(hsv)
    L, A, B = cv2.split(lab)

    leaf = leaf_mask > 0
    cm = np.zeros(img_bgr.shape[:2], dtype=np.uint8)

    healthy = leaf & (H >= 35) & (H <= 90) & (S >= 50) & (V >= 60)
    chlorotic = leaf & (H >= 18) & (H <= 34) & (S >= 60) & (V >= 80)

    # Brown/necrotic — two complementary rules (HSV and Lab) with moderately dark L.
    brown_hsv = leaf & (H >= 5) & (H <= 20) & (
        S >= 60) & (V >= 40) & (V <= 200)
    brown_lab = leaf & (A > 132) & (B > 140) & (L > 50) & (L < 160)
    brown = brown_hsv | brown_lab

    # Dark lesion — deeply darker than a healthy leaf (tight threshold, was too loose).
    dark = leaf & (L < 45) & (S < 200)

    # Senesced/mature tissue — pale tan/straw, low-to-mid saturation.
    senesced = leaf & (H >= 15) & (H <= 32) & (S >= 15) & (S < 80) & (V >= 90)

    # Precedence (lowest → highest): healthy → senesced → chlorotic → brown → dark.
    cm[healthy] = 1
    cm[senesced] = 5
    cm[chlorotic] = 2
    cm[brown] = 3
    cm[dark] = 4

    # In specimen mode, reclassify "brown lesion" pixels as senesced (mature tissue).
    # This fixes the dry post-harvest panicle case.
    if mode == "specimen":
        cm[cm == 3] = 5

    # Clean small speckles per class.
    for cls in (2, 3, 4, 5):
        m = (cm == cls).astype(np.uint8) * 255
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        cm[(cm == cls) & (m == 0)] = 1
        cm[(m > 0) & leaf] = cls

    cm[leaf & (cm == 0)] = 1
    return cm


# %% Cell 5 — Per-lesion feature extraction (count, size, shape, texture)
def extract_lesion_features(img_bgr, class_map, cls_value, cls_name):
    """
    For one lesion class, label connected lesion blobs and compute features
    (area, perimeter, eccentricity, solidity, texture) for each blob.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    binary = (class_map == cls_value).astype(np.uint8)
    # Remove tiny specks
    binary = morphology.remove_small_objects(binary.astype(bool),
                                             min_size=MIN_LESION_AREA_PX).astype(np.uint8)
    labels = measure.label(binary, connectivity=2)
    rows = []
    for region in measure.regionprops(labels, intensity_image=gray):
        if region.area < MIN_LESION_AREA_PX:
            continue
        minr, minc, maxr, maxc = region.bbox
        patch = gray[minr:maxr, minc:maxc]
        # Texture: GLCM contrast and homogeneity (one-liner summary).
        if patch.size >= 16:
            glcm = graycomatrix((patch // 16).astype(np.uint8), distances=[1],
                                angles=[0], levels=16, symmetric=True, normed=True)
            contrast = float(graycoprops(glcm, "contrast")[0, 0])
            homogeneity = float(graycoprops(glcm, "homogeneity")[0, 0])
        else:
            contrast, homogeneity = np.nan, np.nan
        rows.append({
            "lesion_class":  cls_name,
            "area_px":       int(region.area),
            "perimeter_px":  float(region.perimeter),
            "eccentricity":  float(region.eccentricity),
            "solidity":      float(region.solidity),
            "equiv_diameter_px": float(region.equivalent_diameter),
            "glcm_contrast": contrast,
            "glcm_homogeneity": homogeneity,
            "centroid_y":    float(region.centroid[0]),
            "centroid_x":    float(region.centroid[1]),
        })
    return rows, labels


# %% Cell 6 — Severity classification (farmer-friendly bands)
def severity_band(pct):
    """IRRI-style severity bands, simplified for farmer messaging."""
    if pct < 1:
        return "Healthy",          "No action needed."
    if pct < 5:
        return "Very mild",        "Monitor weekly; no spray yet."
    if pct < 15:
        return "Mild",             "Scout the field; remove worst leaves."
    if pct < 30:
        return "Moderate",         "Consider a fungicide spray soon."
    if pct < 50:
        return "Severe",           "Spray immediately; consult extension officer."
    return "Very severe",      "Urgent: treat and check neighbouring plots."


# %% Cell 7 — Visualisation for a single image
CLASS_COLORS = {
    1: (0, 180,   0),   # healthy   → green
    2: (0, 230, 255),   # chlorotic → yellow (BGR)
    3: (0,  90, 200),   # brown     → orange-brown
    4: (30,  30,  30),   # dark      → near-black
    5: (150, 200, 230),   # senesced  → pale tan (not a disease)
}
CLASS_NAMES = {1: "Healthy", 2: "Chlorotic", 3: "Brown lesion",
               4: "Dark lesion", 5: "Senesced (mature)"}


def make_overlay(img_bgr, class_map, mode="field"):
    """
    Build a class-coloured overlay.
    In 'field' mode the non-leaf background is dimmed to dark grey so leaves pop.
    In 'specimen' mode the background is kept near its original brightness so a
    pale specimen on a white/grey backdrop remains visible to the eye.
    """
    overlay = img_bgr.copy()
    for cls, color_bgr in CLASS_COLORS.items():
        overlay[class_map == cls] = color_bgr
    blended = cv2.addWeighted(img_bgr, 0.45, overlay, 0.55, 0)
    non_leaf = class_map == 0
    if mode == "specimen":
        # Keep backdrop bright so the specimen contour stays readable.
        dimmed = (img_bgr * 0.85 + 30).clip(0, 255).astype(np.uint8)
    else:
        dimmed = (img_bgr * 0.35).astype(np.uint8)
    blended[non_leaf] = dimmed[non_leaf]
    return blended


def save_per_image_figure(img_bgr, leaf_mask, subject_mask, class_map, stats, out_path, title):
    """One figure per image (as requested) — never combined across images."""
    mode = stats.get("scene_mode", "field")
    overlay = make_overlay(img_bgr, class_map, mode=mode)

    # Lesion-only highlight: draw contours of brown/dark/chlorotic regions so
    # the farmer sees exactly what the algorithm counted as infected.
    highlight = img_bgr.copy()
    for cls_val, color_bgr in [(3, (0, 90, 220)), (4, (30, 30, 30)), (2, (0, 220, 230))]:
        m = ((class_map == cls_val).astype(np.uint8)) * 255
        contours, _ = cv2.findContours(
            m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(highlight, contours, -1, color_bgr, 2)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("1. Original field image", fontsize=11)
    axes[0, 0].axis("off")

    # Leaf mask visualization — choose highlight colour by scene mode so the
    # specimen doesn't get painted green-on-green.
    if mode == "specimen":
        highlight_color_bgr = np.array(
            [255, 100, 200], dtype=np.uint8)   # magenta
        bg_scale = 0.75
    else:
        highlight_color_bgr = np.array(
            [0, 255, 0], dtype=np.uint8)       # green
        bg_scale = 0.35
    hl_layer = np.zeros_like(img_bgr)
    hl_layer[:] = highlight_color_bgr
    blended_leaf = np.where(leaf_mask[..., None] > 0,
                            cv2.addWeighted(img_bgr, 0.45, hl_layer, 0.55, 0),
                            (img_bgr * bg_scale).astype(np.uint8))
    axes[0, 1].imshow(cv2.cvtColor(blended_leaf, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title(
        f"2. Isolated leaves  (leaf area = {stats['leaf_area_px']:,} px, "
        f"{stats['leaf_frac_pct']:.1f}% of frame)", fontsize=11)
    axes[0, 1].axis("off")

    axes[1, 0].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title("3. Per-pixel lesion class map", fontsize=11)
    axes[1, 0].axis("off")
    legend_elements = [
        Patch(facecolor=(0.0, 0.70, 0.0),
              label=f"Healthy  ({stats['pct_healthy']:.1f}%)"),
        Patch(facecolor=(1.0, 0.90, 0.0),
              label=f"Chlorotic ({stats['pct_chlorotic']:.1f}%)"),
        Patch(facecolor=(0.78, 0.35, 0.0),
              label=f"Brown lesion ({stats['pct_brown']:.1f}%)"),
        Patch(facecolor=(0.12, 0.12, 0.12),
              label=f"Dark lesion ({stats['pct_dark']:.1f}%)"),
        Patch(facecolor=(0.90, 0.78, 0.58),
              label=f"Senesced ({stats['pct_senesced']:.1f}%)"),
    ]
    axes[1, 0].legend(handles=legend_elements, loc="lower center",
                      bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=8, frameon=False)

    axes[1, 1].imshow(cv2.cvtColor(highlight, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title(
        f"4. Detected lesions outlined\n"
        f"Brown: {stats['n_brown_lesions']}   Dark: {stats['n_dark_lesions']}   "
        f"Chlorotic patches: {stats['n_chlorotic_patches']}", fontsize=11)
    axes[1, 1].axis("off")

    band, advice = severity_band(stats["severity_subject_pct"])
    fig.suptitle(
        f"{title}  |  scene mode: {stats['scene_mode']}\n"
        f"Severity (whole frame): {stats['severity_pct']:.1f}%   |   "
        f"Severity (main leaves): {stats['severity_subject_pct']:.1f}% → {band}\n"
        f"Advice: {advice}",
        fontsize=12, y=1.00)
    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# %% Cell 8 — Single-image pipeline
def analyze_image(image_path, location_key, out_dir):
    """
    Full pipeline on one image. Produces:
      - <stem>_analysis.png  (3-panel figure)
      - returns a dict (summary row) + a list of per-lesion dicts
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read {image_path}")

    # Downscale very large images for speed (keep aspect ratio).
    h0, w0 = img.shape[:2]
    scale = MAX_SIDE / max(h0, w0) if max(h0, w0) > MAX_SIDE else 1.0
    if scale < 1.0:
        img = cv2.resize(img, (int(w0 * scale), int(h0 * scale)),
                         interpolation=cv2.INTER_AREA)

    # Detect scene mode — this gates the senesced-vs-brown-lesion logic.
    mode = detect_scene_mode(img)

    leaf_mask, subject_mask = isolate_leaves(img, mode)
    class_map = detect_lesions(img, leaf_mask, mode)

    # Pixel accounting — full leaf mask (denominator = all plant tissue in frame).
    # Note: senesced pixels count as LEAF area but NOT as infected area.
    total_px = img.shape[0] * img.shape[1]
    leaf_px = int((class_map > 0).sum())
    healthy_px = int((class_map == 1).sum())
    chlorotic_px = int((class_map == 2).sum())
    brown_px = int((class_map == 3).sum())
    dark_px = int((class_map == 4).sum())
    senesced_px = int((class_map == 5).sum())
    infected_px = chlorotic_px + brown_px + dark_px
    # Severity denominator excludes senesced tissue (it isn't living leaf).
    living_px = max(leaf_px - senesced_px, 1)
    severity_pct = 100.0 * infected_px / living_px

    # Subject-focused severity (main 1–3 leaves only).
    sub_mask_bool = subject_mask > 0
    sub_total = int(sub_mask_bool.sum())
    if sub_total > 0:
        sub_class = class_map.copy()
        sub_class[~sub_mask_bool] = 0
        sub_healthy = int((sub_class == 1).sum())
        sub_chlorotic = int((sub_class == 2).sum())
        sub_brown = int((sub_class == 3).sum())
        sub_dark = int((sub_class == 4).sum())
        sub_senesced = int((sub_class == 5).sum())
        sub_infected = sub_chlorotic + sub_brown + sub_dark
        sub_living = max(sub_total - sub_senesced, 1)
        severity_subject_pct = 100.0 * sub_infected / sub_living
    else:
        sub_healthy = sub_chlorotic = sub_brown = sub_dark = sub_senesced = sub_infected = 0
        severity_subject_pct = 0.0

    # Per-lesion features for the two "real lesion" classes
    lesion_rows = []
    for cls_val, cls_name in [(3, "Brown lesion"), (4, "Dark lesion"), (2, "Chlorotic patch")]:
        rows, _ = extract_lesion_features(img, class_map, cls_val, cls_name)
        lesion_rows.extend(rows)

    stats = {
        "image_name":       image_path.name,
        "location_key":     location_key,
        "site":             LOCATION_META.get(location_key, {}).get("site", "Unknown"),
        "latitude":         LOCATION_META.get(location_key, {}).get("lat", np.nan),
        "longitude":        LOCATION_META.get(location_key, {}).get("lon", np.nan),
        "date":             LOCATION_META.get(location_key, {}).get("date", ""),
        "scene_mode":       mode,
        "image_width_px":   img.shape[1],
        "image_height_px":  img.shape[0],
        "total_area_px":    total_px,
        "leaf_area_px":     leaf_px,
        "subject_area_px":  sub_total,
        "leaf_frac_pct":    100.0 * leaf_px / total_px,
        "healthy_px":       healthy_px,
        "chlorotic_px":     chlorotic_px,
        "brown_px":         brown_px,
        "dark_px":          dark_px,
        "senesced_px":      senesced_px,
        "infected_px":      infected_px,
        "severity_pct":         severity_pct,
        "severity_subject_pct": severity_subject_pct,
        "pct_healthy":      100.0 * healthy_px / leaf_px if leaf_px else 0.0,
        "pct_chlorotic":    100.0 * chlorotic_px / leaf_px if leaf_px else 0.0,
        "pct_brown":        100.0 * brown_px / leaf_px if leaf_px else 0.0,
        "pct_dark":         100.0 * dark_px / leaf_px if leaf_px else 0.0,
        "pct_senesced":     100.0 * senesced_px / leaf_px if leaf_px else 0.0,
        "n_lesions":        len(lesion_rows),
        "n_brown_lesions":  sum(1 for r in lesion_rows if r["lesion_class"] == "Brown lesion"),
        "n_dark_lesions":   sum(1 for r in lesion_rows if r["lesion_class"] == "Dark lesion"),
        "n_chlorotic_patches": sum(1 for r in lesion_rows if r["lesion_class"] == "Chlorotic patch"),
    }
    band, advice = severity_band(severity_subject_pct)
    stats["severity_band"] = band
    stats["farmer_advice"] = advice

    # Save per-image figure (never combined across images — as requested)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / f"{image_path.stem}_analysis.png"
    save_per_image_figure(
        img, leaf_mask, subject_mask, class_map, stats,
        out_path=fig_path,
        title=f"{image_path.name}  |  {stats['site']}  |  {stats['date']}"
    )
    stats["result_figure"] = str(fig_path.relative_to(PROJECT_ROOT))

    # Attach image_name to each per-lesion row
    for r in lesion_rows:
        r["image_name"] = image_path.name
        r["location_key"] = location_key
    return stats, lesion_rows


# %% Cell 9 — Folder / dataset orchestration
def discover_images(data_root):
    """Return list of (image_path, location_key). Handles 1 folder / 1 image gracefully."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    items = []
    if not data_root.exists():
        return items
    for sub in sorted(p for p in data_root.iterdir() if p.is_dir()):
        for img_path in sorted(sub.iterdir()):
            if img_path.suffix.lower() in exts:
                items.append((img_path, sub.name))
    return items


def run_pipeline(data_root=DATA_DIR, results_root=RESULTS_DIR):
    items = discover_images(data_root)
    if not items:
        print(f"No images found under {data_root}")
        return None, None

    all_summary, all_lesions = [], []
    for img_path, loc_key in items:
        out_dir = results_root / loc_key
        try:
            summary, lesions = analyze_image(img_path, loc_key, out_dir)
        except Exception as e:
            print(f"  [ERROR] {img_path.name}: {e}")
            continue
        all_summary.append(summary)
        all_lesions.extend(lesions)
        print(f"  [OK] {loc_key}/{img_path.name}  "
              f"severity(frame)={summary['severity_pct']:.1f}%  "
              f"severity(main)={summary['severity_subject_pct']:.1f}%  "
              f"({summary['severity_band']})  "
              f"lesions={summary['n_lesions']}")

    # Write CSV reports (per-image summary + per-lesion details).
    summary_df = pd.DataFrame(all_summary)
    lesions_df = pd.DataFrame(all_lesions)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_csv = results_root / f"report_summary_{stamp}.csv"
    lesions_csv = results_root / f"report_lesions_{stamp}.csv"
    summary_df.to_csv(summary_csv, index=False)
    if not lesions_df.empty:
        lesions_df.to_csv(lesions_csv, index=False)

    print(f"\nSummary CSV : {summary_csv}")
    print(
        f"Lesion CSV  : {lesions_csv if not lesions_df.empty else '(no lesions detected)'}")
    print(f"Figures in  : {results_root}/<location>/")
    return summary_df, lesions_df


# %% Cell 10 — Run it
if __name__ == "__main__":
    summary_df, lesions_df = run_pipeline()
    if summary_df is not None:
        print("\n=== Per-image summary ===")
        cols = ["image_name", "location_key", "scene_mode",
                "severity_subject_pct", "severity_band",
                "n_brown_lesions", "n_dark_lesions", "pct_senesced"]
        print(summary_df[cols].to_string(index=False))
