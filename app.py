"""
Rice Disease Analysis — Streamlit Application
==============================================
Upload one or more rice leaf images, choose your segmentation strategy
(Trained / Classical / Intersection), run the full lesion-clustering
pipeline, view per-image results, and download the annotated image + CSV.

Usage
-----
    streamlit run app.py

Folder layout expected beside this file
----------------------------------------
    app.py
    rice_disease_analysis.py
    lesion_clustering.py
    train_leaf_segmenter.py          ← needed if you have a .pt checkpoint
    models/
        leaf_segmenter_best.pt       ← optional; classical fallback used if absent
        clusters/
            trained/lesion_cluster_model.joblib
            classical/lesion_cluster_model.joblib
            intersection/lesion_cluster_model.joblib
"""

import streamlit as st
import pandas as pd
import numpy as np
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import io
import sys
import tempfile
import traceback
from pathlib import Path
import urllib.request

# ── Model auto-download ─────────────────────────────────────────────
MODEL_URL = "https://drive.google.com/uc?export=download&id=1D5moFE9mvm2fj3wQJwaNUiKelelAI0AH"

MODEL_PATH = Path(__file__).resolve().parent / "models" / "leaf_segmenter_best.pt"

def ensure_model():
    if not MODEL_PATH.exists():
        import streamlit as st
        st.info("Downloading leaf segmentation model... ⏳")

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

        try:
            urllib.request.urlretrieve(MODEL_URL, str(MODEL_PATH))
            st.success("Model downloaded successfully ✅")
        except Exception as e:
            st.error(f"Failed to download model: {e}")

# Call it immediately
ensure_model()
import cv2
import matplotlib
matplotlib.use("Agg")

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Rice Disease Analysis",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject custom CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Main accent: forest green + gold */
:root {
    --forest: #1a3a2a;
    --gold: #c9a84c;
}
[data-testid="stSidebar"] {
    background: #1a3a2a !important;
}
[data-testid="stSidebar"] * {
    color: #f0ebe0 !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stMultiSelect label,
[data-testid="stSidebar"] .stCheckbox label,
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] p, [data-testid="stSidebar"] span {
    color: #c9a84c !important;
    font-weight: 600;
}
.metric-box {
    background: #f4f1e8;
    border-left: 4px solid #c9a84c;
    border-radius: 6px;
    padding: 12px 16px;
    margin-bottom: 10px;
}
.metric-box .metric-val {
    font-size: 28px; font-weight: 700; color: #1a3a2a;
    line-height: 1.1;
}
.metric-box .metric-lab {
    font-size: 12px; color: #5a5a4a; text-transform: uppercase;
    letter-spacing: 0.8px; margin-top: 2px;
}
.disease-pill {
    display: inline-block; padding: 4px 12px;
    border-radius: 12px; font-size: 13px; font-weight: 600;
    margin-right: 6px; margin-bottom: 6px;
}
.pill-blast { background: #f3d6ff; color: #5a1580; }
.pill-brownspot { background: #ffe0cc; color: #7a3010; }
.pill-other { background: #e8e8e8; color: #444; }
.pill-healthy { background: #d4f0dc; color: #1a5c2a; }
.advice-box {
    background: #f9f5e8; border: 1px solid #c9a84c;
    border-radius: 8px; padding: 14px 18px; margin-top: 12px;
}
.advice-box p { font-size: 14px; color: #3a3a2a; line-height: 1.6; margin: 0; }
.section-header {
    font-size: 13px; font-weight: 700; color: #1a3a2a;
    text-transform: uppercase; letter-spacing: 1.2px;
    border-bottom: 2px solid #c9a84c; padding-bottom: 4px;
    margin: 18px 0 12px;
}
.stDownloadButton > button {
    background: #1a3a2a !important;
    color: #f0ebe0 !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
}
.stDownloadButton > button:hover {
    background: #2d5a3d !important;
}
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Project root & sys.path ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── Lazy import of pipeline (only after sys.path is set) ─────────────────────


@st.cache_resource(show_spinner="Loading pipeline modules …")
def _import_pipeline():
    """Import all heavy modules once and cache the callables."""
    try:
        from lesion_clustering import (
            segment_leaf_all,
            find_lesion_candidates,
            extract_lesion_features,
            apply_cluster_model,
            load_cluster_model,
            DISEASE_NAMES,
            DISEASE_ADVICE,
            DISEASE_COLORS_BGR,
            STRATEGY_OVERLAY_BGR,
            severity_band,
            color_for_disease,
        )
        from rice_disease_analysis import MAX_SIDE
        return {
            "segment_leaf_all":      segment_leaf_all,
            "find_lesion_candidates": find_lesion_candidates,
            "extract_lesion_features": extract_lesion_features,
            "apply_cluster_model":   apply_cluster_model,
            "load_cluster_model":    load_cluster_model,
            "DISEASE_NAMES":         DISEASE_NAMES,
            "DISEASE_ADVICE":        DISEASE_ADVICE,
            "DISEASE_COLORS_BGR":    DISEASE_COLORS_BGR,
            "STRATEGY_OVERLAY_BGR":  STRATEGY_OVERLAY_BGR,
            "severity_band":         severity_band,
            "color_for_disease":     color_for_disease,
            "MAX_SIDE":              MAX_SIDE,
            "ok":                    True,
        }
    except Exception as e:
        return {"ok": False, "error": str(e), "trace": traceback.format_exc()}


# ── Load cluster models (cached per path) ────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _load_cluster_model_cached(path_str):
    try:
        p = _import_pipeline()
        if not p["ok"]:
            return None
        model = p["load_cluster_model"](path_str)
        return model
    except Exception:
        return None


def get_cluster_models(cluster_dir: Path):
    """Return {strategy: model_or_None} for all three strategies."""
    strategies = ["trained", "classical", "intersection"]
    models = {}
    for s in strategies:
        path = cluster_dir / s / "lesion_cluster_model.joblib"
        if path.exists():
            models[s] = _load_cluster_model_cached(str(path))
        else:
            models[s] = None
    return models


# ── Core analysis for one image (numpy array BGR) ────────────────────────────
def run_analysis(img_bgr: np.ndarray, strategies: list,
                 leaf_model_path: str, cluster_models: dict) -> dict:
    """
    Run segmentation + feature extraction + clustering for the selected strategies.
    Returns a rich result dict.
    """
    p = _import_pipeline()

    # Downscale if needed
    H0, W0 = img_bgr.shape[:2]
    scale = p["MAX_SIDE"] / max(H0, W0) if max(H0, W0) > p["MAX_SIDE"] else 1.0
    if scale < 1.0:
        img_bgr = cv2.resize(img_bgr, (int(W0*scale), int(H0*scale)),
                             interpolation=cv2.INTER_AREA)

    # Segmentation — one call returns all three masks
    seg = p["segment_leaf_all"](img_bgr,
                                model_path=leaf_model_path if leaf_model_path else None)
    masks = {s: seg[s] for s in ["trained", "classical", "intersection"]}

    results = {}  # keyed by strategy
    for strat in strategies:
        mask = masks[strat]
        cand = p["find_lesion_candidates"](img_bgr, mask)
        rows, lbl_img = p["extract_lesion_features"](img_bgr, mask, cand)

        df = pd.DataFrame(rows)
        if len(df):
            cl = cluster_models.get(strat)
            if cl is not None:
                df = p["apply_cluster_model"](df, cl)
            else:
                df["cluster_id"] = -1
                df["disease_label"] = "Other"
        else:
            df["cluster_id"] = pd.Series(dtype=int)
            df["disease_label"] = pd.Series(dtype=str)

        leaf_px = int((mask > 0).sum())
        les_area = int(df["area_px"].sum()) if len(df) else 0
        sev = 100.0 * les_area / max(leaf_px, 1)
        band, _ = p["severity_band"](sev)

        # Per-disease counts / area / pct
        disease_stats = {}
        for d in p["DISEASE_NAMES"]:
            sub = df[df["disease_label"] == d] if len(df) else pd.DataFrame()
            disease_stats[d] = {
                "count":    int(len(sub)),
                "area_px":  int(sub["area_px"].sum()) if len(sub) else 0,
                "pct":      100.0 * sub["area_px"].sum() / max(leaf_px, 1) if len(sub) else 0.0,
            }

        dominant = (max(p["DISEASE_NAMES"],
                        key=lambda d: disease_stats[d]["area_px"])
                    if les_area > 0 else "Healthy")

        results[strat] = {
            "mask":          mask,
            "labels_img":    lbl_img,
            "lesion_df":     df,
            "leaf_px":       leaf_px,
            "total_px":      img_bgr.shape[0] * img_bgr.shape[1],
            "les_area":      les_area,
            "severity_pct":  sev,
            "severity_band": band,
            "dominant":      dominant,
            "disease_stats": disease_stats,
            "advice":        p["DISEASE_ADVICE"].get(dominant, ""),
            "src":           seg["trained_src"] if strat == "trained"
            else seg["classical_src"]
            if strat == "classical"
            else f"intersection({seg['trained_src']}∩{seg['classical_src']})",
        }

    # Agreement metrics
    t_px = int((masks["trained"] > 0).sum())
    c_px = int((masks["classical"] > 0).sum())
    i_px = int((masks["intersection"] > 0).sum())
    agreement = {
        "intersection_pct_of_trained":   100.0 * i_px / max(t_px, 1),
        "intersection_pct_of_classical": 100.0 * i_px / max(c_px, 1),
        "dice_agreement":                100.0 * 2 * i_px / max(t_px + c_px, 1),
    }

    return {
        "img":       img_bgr,
        "seg":       seg,
        "masks":     masks,
        "results":   results,
        "agreement": agreement,
        "strategies": strategies,
    }


# ── Render annotated image (returns RGB numpy array) ─────────────────────────
def render_result_image(img_bgr: np.ndarray, strat_result: dict,
                        strat: str, show_overlay=True, show_contours=True,
                        contour_thickness=2) -> np.ndarray:
    """Build the annotated BGR image for a single strategy."""
    p = _import_pipeline()
    img_out = img_bgr.copy()
    mask = strat_result["mask"]
    lbl_img = strat_result["labels_img"]
    ldf = strat_result["lesion_df"]
    col_bgr = p["STRATEGY_OVERLAY_BGR"].get(strat, (200, 200, 200))

    if show_overlay:
        layer = np.zeros_like(img_bgr)
        layer[:] = col_bgr
        img_out = np.where(
            mask[..., None] > 0,
            cv2.addWeighted(img_bgr, 0.4, layer, 0.6, 0),
            (img_bgr * 0.35).astype(np.uint8),
        )

    if show_contours and len(ldf):
        for _, row in ldf.iterrows():
            lid = int(row["lesion_id_in_image"])
            lbl = str(row.get("disease_label", "Other"))
            color = p["color_for_disease"](lbl)
            m = (lbl_img == lid).astype(np.uint8) * 255
            cts, _ = cv2.findContours(
                m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img_out, cts, -1, color, contour_thickness)

    return cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)


# ── Build a matplotlib figure for saving ─────────────────────────────────────
def build_save_figure(img_bgr, analysis, selected_strats,
                      fname, show_overlay, show_contours) -> bytes:
    """
    Build a multi-panel matplotlib figure (original + per-strategy panels)
    and return PNG bytes for download.
    """
    p = _import_pipeline()
    n_strats = len(selected_strats)
    fig, axes = plt.subplots(n_strats + 1, 2,
                             figsize=(12, 4.5 * (n_strats + 1)),
                             squeeze=False)

    # Row 0: original image (full width span)
    ax_orig = axes[0, 0]
    ax_orig.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    ax_orig.set_title(f"Original: {fname}", fontsize=11, fontweight="bold")
    ax_orig.axis("off")
    axes[0, 1].axis("off")

    for row_i, strat in enumerate(selected_strats, start=1):
        sr = analysis["results"][strat]
        mask = sr["mask"]
        lbl_img = sr["labels_img"]
        ldf = sr["lesion_df"]
        col_bgr = p["STRATEGY_OVERLAY_BGR"].get(strat, (200, 200, 200))

        # Mask overlay panel
        layer = np.zeros_like(img_bgr)
        layer[:] = col_bgr
        mask_viz = np.where(
            mask[..., None] > 0,
            cv2.addWeighted(img_bgr, 0.4, layer, 0.6, 0),
            (img_bgr * 0.35).astype(np.uint8))
        axes[row_i, 0].imshow(cv2.cvtColor(mask_viz, cv2.COLOR_BGR2RGB))
        axes[row_i, 0].set_title(
            f"[{strat.upper()}] Leaf mask  ({sr['leaf_px']:,} px)\n"
            f"Severity: {sr['severity_pct']:.1f}% — {sr['severity_band']}",
            fontsize=9)
        axes[row_i, 0].axis("off")

        # Disease-class contours panel
        dis_viz = img_bgr.copy()
        counts = {}
        if len(ldf):
            for _, row in ldf.iterrows():
                lid = int(row["lesion_id_in_image"])
                lbl = str(row.get("disease_label", "Other"))
                col = p["color_for_disease"](lbl)
                m = (lbl_img == lid).astype(np.uint8) * 255
                cts, _ = cv2.findContours(
                    m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(dis_viz, cts, -1, col, 2)
                counts[lbl] = counts.get(lbl, 0) + 1

        axes[row_i, 1].imshow(cv2.cvtColor(dis_viz, cv2.COLOR_BGR2RGB))
        axes[row_i, 1].set_title(
            f"[{strat.upper()}] Disease classes  Dominant: {sr['dominant']}",
            fontsize=9)
        axes[row_i, 1].axis("off")

        # Legend
        legend_patches = []
        for lbl in p["DISEASE_NAMES"]:
            b, g, r = p["color_for_disease"](lbl)
            n = counts.get(lbl, 0)
            legend_patches.append(
                Patch(facecolor=(r/255, g/255, b/255), label=f"{lbl} (n={n})"))
        axes[row_i, 1].legend(handles=legend_patches,
                              loc="lower center", bbox_to_anchor=(0.5, -0.22),
                              ncol=3, fontsize=8, frameon=False)

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ── Severity badge colour ─────────────────────────────────────────────────────
def severity_color(band: str) -> str:
    return {
        "Healthy":      "#1e8449",
        "Very mild":    "#27ae60",
        "Mild":         "#f39c12",
        "Moderate":     "#e67e22",
        "Severe":       "#c0392b",
        "Very severe":  "#7b241c",
    }.get(band, "#888")


def disease_pill_class(disease: str) -> str:
    return {
        "Blast":      "pill-blast",
        "Brown spot": "pill-brownspot",
        "Other":      "pill-other",
        "Healthy":    "pill-healthy",
    }.get(disease, "pill-other")


# ════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🌾 Rice Disease Analysis")
    st.markdown("---")

    # ── Model paths ──
    st.markdown("### Model Paths")
    leaf_model_path = st.text_input(
        "Leaf segmenter (.pt)",
        value=str(PROJECT_ROOT / "models" / "leaf_segmenter_best.pt"),
        help="Path to leaf_segmenter_best.pt. Leave blank to use classical fallback.")
    cluster_dir = st.text_input(
        "Cluster models directory",
        value=str(PROJECT_ROOT / "models" / "clusters"),
        help="Directory with trained/ classical/ intersection/ sub-folders.")

    # Resolve paths
    lm_path = leaf_model_path.strip()
    lm_path = lm_path if (lm_path and Path(lm_path).exists()) else None
    cl_dir = Path(cluster_dir.strip()) if cluster_dir.strip() else None

    # Status indicators
    st.markdown("**Model status:**")
    if lm_path:
        st.success(f"✓ Segmenter found")
    else:
        st.warning("⚠ No .pt file — using classical")

    if cl_dir and cl_dir.exists():
        found = [s for s in ["trained", "classical", "intersection"]
                 if (cl_dir / s / "lesion_cluster_model.joblib").exists()]
        if found:
            st.success(f"✓ Cluster models: {', '.join(found)}")
        else:
            st.warning("⚠ No cluster models found")
    else:
        st.warning("⚠ Cluster dir not found")

    st.markdown("---")

    # ── Strategy selection ──
    st.markdown("### Segmentation Strategy")
    strategy_choice = st.radio(
        "Show results for:",
        options=["All three (compare)", "Trained only",
                 "Classical only", "Intersection only"],
        index=0,
        help="Trained: uses your .pt model.\n"
             "Classical: colour-heuristic.\n"
             "Intersection: only pixels both methods agree on.")

    strategy_map = {
        "All three (compare)": ["trained", "classical", "intersection"],
        "Trained only":        ["trained"],
        "Classical only":      ["classical"],
        "Intersection only":   ["intersection"],
    }
    selected_strategies = strategy_map[strategy_choice]

    st.markdown("---")

    # ── Visualisation options ──
    st.markdown("### Display Options")
    show_mask_overlay = st.checkbox("Show leaf mask overlay", value=True)
    show_disease_contours = st.checkbox("Show disease contours", value=True)
    contour_thickness = st.slider("Contour thickness", 1, 5, 2)

    st.markdown("---")

    # ── Save options ──
    st.markdown("### Export")
    save_annotated = st.checkbox("Save annotated image", value=True)
    save_csv = st.checkbox("Save summary CSV",     value=True)
    save_lesion_csv = st.checkbox("Save per-lesion CSV",  value=True)


# ════════════════════════════════════════════════════════════════════════════
#  MAIN PAGE
# ════════════════════════════════════════════════════════════════════════════
st.markdown("# 🌾 Rice Disease Analysis")
st.markdown(
    "Upload one or more rice leaf images. The system segments the leaf, "
    "detects lesion candidates using leaf-adaptive thresholding, "
    "clusters them into **Blast / Brown spot / Other** using a saved K=3 model, "
    "and reports severity per image.")

# Check pipeline loaded OK
pipeline = _import_pipeline()
if not pipeline["ok"]:
    st.error("Pipeline import failed. Check that all scripts are in the same folder.")
    with st.expander("Error details"):
        st.code(pipeline.get("trace", pipeline.get("error", "")))
    st.stop()

# Load cluster models
cluster_models = {}
if cl_dir and cl_dir.exists():
    cluster_models = get_cluster_models(cl_dir)
    missing = [s for s in ["trained", "classical", "intersection"]
               if cluster_models.get(s) is None]
    if missing:
        st.info(f"ℹ No cluster model for: **{', '.join(missing)}**. "
                "Those strategies will label all lesions as 'Other'. "
                "Run `lesion_clustering.py --mode fit` to create them.")

# ── Upload ───────────────────────────────────────────────────────────────────
uploaded_files = st.file_uploader(
    "Upload rice leaf images (JPG, PNG, BMP, TIFF)",
    type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
    accept_multiple_files=True)

if not uploaded_files:
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **What this app does:**
        - Separates leaf from background (hand, soil, weeds, sky)
        - Detects lesion candidate pixels using leaf-adaptive colour deviation
        - Classifies lesions into Blast / Brown spot / Other via K=3 KMeans
        - Reports severity %, dominant disease, and farmer advice
        """)
    with col2:
        st.markdown("""
        **Three segmentation strategies:**
        - 🔵 **Trained** — DeepLabV3+MobileNetV3 checkpoint
        - 🟢 **Classical** — colour heuristic (HSV + Lab + focus mask)
        - 🟡 **Intersection** — pixels both methods agree on (most reliable)
        """)
    with col3:
        st.markdown("""
        **Expected folder layout:**
        ```
        app.py
        rice_disease_analysis.py
        lesion_clustering.py
        train_leaf_segmenter.py
        models/
          leaf_segmenter_best.pt
          clusters/
            trained/
            classical/
            intersection/
        ```
        """)
    st.stop()

# ── Process each uploaded file ───────────────────────────────────────────────
st.markdown("---")
all_summaries = []
all_lesions = []

for uf in uploaded_files:
    st.markdown(f"## 📸 {uf.name}")
    fname = uf.name

    # Decode image
    file_bytes = np.frombuffer(uf.read(), np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None:
        st.error(f"Could not decode {fname}")
        continue

    # Run analysis with spinner
    with st.spinner(f"Analysing {fname} …"):
        try:
            analysis = run_analysis(
                img_bgr,
                strategies=selected_strategies,
                leaf_model_path=lm_path,
                cluster_models=cluster_models,
            )
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            with st.expander("Traceback"):
                st.code(traceback.format_exc())
            continue

    res_dict = analysis["results"]

    # ── Original image ────────────────────────────────────────────────────
    col_orig, col_info = st.columns([1.5, 1])
    with col_orig:
        st.markdown('<div class="section-header">Original image</div>',
                    unsafe_allow_html=True)
        st.image(cv2.cvtColor(analysis["img"], cv2.COLOR_BGR2RGB),
                 use_container_width=True)

    with col_info:
        st.markdown('<div class="section-header">Image info</div>',
                    unsafe_allow_html=True)
        H, W = analysis["img"].shape[:2]
        st.markdown(f"""
        | Property | Value |
        |---|---|
        | Filename | `{fname}` |
        | Dimensions | {W} × {H} px |
        | Total pixels | {W*H:,} |
        | Scene mode | `{analysis['seg']['classical_src']}` |
        | Segmenter | `{analysis['seg']['trained_src']}` |
        """)

        # Mask agreement
        agr = analysis["agreement"]
        st.markdown('<div class="section-header">Mask agreement</div>',
                    unsafe_allow_html=True)
        st.markdown(f"""
        | Metric | Value |
        |---|---|
        | Dice agreement | **{agr['dice_agreement']:.1f}%** |
        | Intersection / Trained | {agr['intersection_pct_of_trained']:.1f}% |
        | Intersection / Classical | {agr['intersection_pct_of_classical']:.1f}% |
        """)
        if agr["dice_agreement"] < 50:
            st.warning("Low agreement: trained and classical masks differ substantially. "
                       "Consider labelling more images for the segmenter.")
        elif agr["dice_agreement"] > 85:
            st.success("High agreement: both methods are consistent.")

    # ── Per-strategy results ──────────────────────────────────────────────
    st.markdown('<div class="section-header">Results by strategy</div>',
                unsafe_allow_html=True)

    tabs = st.tabs([s.capitalize() for s in selected_strategies])

    for tab, strat in zip(tabs, selected_strategies):
        with tab:
            sr = res_dict[strat]

            # Metric cards row
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(f"""
                <div class="metric-box">
                  <div class="metric-val">{sr['severity_pct']:.1f}%</div>
                  <div class="metric-lab">Severity</div>
                </div>""", unsafe_allow_html=True)
            with m2:
                sev_col = severity_color(sr['severity_band'])
                st.markdown(f"""
                <div class="metric-box">
                  <div class="metric-val" style="color:{sev_col};font-size:20px">
                    {sr['severity_band']}
                  </div>
                  <div class="metric-lab">Severity band</div>
                </div>""", unsafe_allow_html=True)
            with m3:
                st.markdown(f"""
                <div class="metric-box">
                  <div class="metric-val">{sr['les_area']:,}</div>
                  <div class="metric-lab">Lesion area (px)</div>
                </div>""", unsafe_allow_html=True)
            with m4:
                st.markdown(f"""
                <div class="metric-box">
                  <div class="metric-val">{len(sr['lesion_df'])}</div>
                  <div class="metric-lab">Lesions detected</div>
                </div>""", unsafe_allow_html=True)

            # Disease breakdown
            st.markdown("**Disease breakdown:**")
            pill_html = ""
            for d, ds in sr["disease_stats"].items():
                if ds["count"] > 0:
                    cls = disease_pill_class(d)
                    pill_html += (
                        f'<span class="disease-pill {cls}">'
                        f'{d}: {ds["count"]} lesions ({ds["pct"]:.1f}%)</span>')
            if not pill_html:
                pill_html = '<span class="disease-pill pill-healthy">No lesions detected</span>'
            st.markdown(pill_html, unsafe_allow_html=True)

            # Annotated image
            annotated_rgb = render_result_image(
                analysis["img"], sr, strat,
                show_overlay=show_mask_overlay,
                show_contours=show_disease_contours,
                contour_thickness=contour_thickness)
            st.image(annotated_rgb, caption=f"{strat.capitalize()} — annotated",
                     use_container_width=True)

            # Leaf mask stats
            leaf_frac = 100.0 * sr["leaf_px"] / max(sr["total_px"], 1)
            st.caption(
                f"Leaf area: {sr['leaf_px']:,} px ({leaf_frac:.1f}% of frame)  |  "
                f"Source: `{sr['src']}`")

            # Farmer advice
            if sr["dominant"] != "Healthy" and sr["severity_pct"] >= 1.0:
                st.markdown(
                    f'<div class="advice-box">'
                    f'<p>🌿 <strong>Advice ({sr["dominant"]}):</strong> {sr["advice"]}</p>'
                    f'</div>',
                    unsafe_allow_html=True)

            # Per-lesion table (collapsible)
            if len(sr["lesion_df"]) > 0:
                with st.expander(f"Per-lesion feature table ({len(sr['lesion_df'])} lesions)"):
                    display_cols = [
                        "lesion_id_in_image", "disease_label",
                        "area_px", "eccentricity", "circularity",
                        "spindle_score", "darkness_drop",
                        "core_vs_margin_L_diff", "glcm_contrast",
                    ]
                    disp_df = sr["lesion_df"][[c for c in display_cols
                                              if c in sr["lesion_df"].columns]].copy()
                    for col in disp_df.select_dtypes(include=float).columns:
                        disp_df[col] = disp_df[col].round(3)
                    st.dataframe(disp_df, use_container_width=True)

    # ── Collect summary rows ──────────────────────────────────────────────
    row = {"image_name": fname, "width_px": W, "height_px": H,
           "trained_src": analysis["seg"]["trained_src"],
           "classical_src": analysis["seg"]["classical_src"],
           **{f"dice_agreement": analysis["agreement"]["dice_agreement"]}}
    for strat in selected_strategies:
        sr = res_dict[strat]
        row[f"{strat}_severity_pct"] = round(sr["severity_pct"], 2)
        row[f"{strat}_severity_band"] = sr["severity_band"]
        row[f"{strat}_n_lesions"] = len(sr["lesion_df"])
        row[f"{strat}_lesion_area_px"] = sr["les_area"]
        row[f"{strat}_leaf_px"] = sr["leaf_px"]
        row[f"{strat}_dominant_disease"] = sr["dominant"]
        for d in pipeline["DISEASE_NAMES"]:
            ds = sr["disease_stats"][d]
            row[f"{strat}_n_{d}"] = ds["count"]
            row[f"{strat}_pct_{d}"] = round(ds["pct"], 2)
    all_summaries.append(row)

    # Add lesion rows
    for strat in selected_strategies:
        ldf = res_dict[strat]["lesion_df"].copy()
        ldf["mask_strategy"] = strat
        all_lesions.append(ldf)

    # ── Download buttons for this image ──────────────────────────────────
    st.markdown('<div class="section-header">Export for this image</div>',
                unsafe_allow_html=True)
    dl1, dl2, dl3 = st.columns(3)

    if save_annotated:
        with dl1:
            fig_bytes = build_save_figure(
                analysis["img"], analysis, selected_strategies,
                fname, show_mask_overlay, show_disease_contours)
            st.download_button(
                label="⬇ Download annotated figure",
                data=fig_bytes,
                file_name=f"{Path(fname).stem}_analysis.png",
                mime="image/png",
                key=f"dl_fig_{fname}")

    if save_csv:
        with dl2:
            single_csv = pd.DataFrame([row]).to_csv(index=False).encode()
            st.download_button(
                label="⬇ Download summary CSV",
                data=single_csv,
                file_name=f"{Path(fname).stem}_summary.csv",
                mime="text/csv",
                key=f"dl_sum_{fname}")

    if save_lesion_csv:
        with dl3:
            if all_lesions:
                last_lesion_df = all_lesions[-1]
                if len(last_lesion_df):
                    les_csv = last_lesion_df.to_csv(index=False).encode()
                    st.download_button(
                        label="⬇ Download lesion CSV",
                        data=les_csv,
                        file_name=f"{Path(fname).stem}_lesions.csv",
                        mime="text/csv",
                        key=f"dl_les_{fname}")

    st.markdown("---")

# ── Batch export (all uploaded images together) ───────────────────────────────
if len(uploaded_files) > 1 and all_summaries:
    st.markdown("## 📊 Batch export — all images")

    batch_col1, batch_col2 = st.columns(2)
    with batch_col1:
        batch_sum = pd.DataFrame(all_summaries)
        st.dataframe(batch_sum, use_container_width=True)
        st.download_button(
            label="⬇ Download combined summary CSV",
            data=batch_sum.to_csv(index=False).encode(),
            file_name="batch_summary.csv",
            mime="text/csv",
            key="dl_batch_sum")

    with batch_col2:
        if any(len(ldf) > 0 for ldf in all_lesions):
            batch_les = pd.concat(
                [ldf for ldf in all_lesions if len(ldf)], ignore_index=True)
            st.dataframe(batch_les.head(100), use_container_width=True)
            st.caption(
                f"Showing first 100 of {len(batch_les)} total lesion rows")
            st.download_button(
                label="⬇ Download combined lesion CSV",
                data=batch_les.to_csv(index=False).encode(),
                file_name="batch_lesions.csv",
                mime="text/csv",
                key="dl_batch_les")

    # Bar chart: severity comparison across images × strategies
    if len(all_summaries) > 0:
        st.markdown("### Severity comparison")
        chart_data = pd.DataFrame([
            {"Image": r["image_name"],
             "Strategy": strat,
             "Severity (%)": r.get(f"{strat}_severity_pct", 0)}
            for r in all_summaries
            for strat in selected_strategies
        ])
        try:
            import altair as alt
            chart = (
                alt.Chart(chart_data)
                .mark_bar()
                .encode(
                    x=alt.X("Image:N", title="Image"),
                    y=alt.Y("Severity (%):Q", title="Severity (%)"),
                    color=alt.Color("Strategy:N",
                                    scale=alt.Scale(
                                        domain=["trained", "classical",
                                                "intersection"],
                                        range=["#3b82f6", "#22c55e", "#f59e0b"])),
                    xOffset="Strategy:N",
                    tooltip=["Image", "Strategy", "Severity (%)"])
                .properties(height=300)
            )
            st.altair_chart(chart, use_container_width=True)
        except ImportError:
            st.bar_chart(chart_data.pivot(index="Image", columns="Strategy",
                                          values="Severity (%)"))

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<small>Rice Disease Monitoring System · ICAR-IIAB · "
    "Ranchi, Jharkhand · Phase 1</small>",
    unsafe_allow_html=True)
