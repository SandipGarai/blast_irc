# 🌾 Rice Disease Analysis

A **Streamlit web application** for detecting and analyzing rice leaf diseases using lesion clustering.

Upload one or more rice leaf images, choose a segmentation strategy, run the full pipeline, and download annotated images along with CSV summaries.

---

## Features

- **Three segmentation strategies**
- **Trained** — Deep learning model (PyTorch)
- **Classical** — Computer vision-based segmentation
- **Intersection** — Combined, more reliable mask

- **Automated lesion detection & clustering**

- **Per-image analysis with severity metrics**

- **Annotated visual outputs**

- **Downloadable results (images + CSV)**

---

## Repository Structure

```
blast_irc/
│
├── app.py                              # Streamlit entry point
├── rice_disease_analysis.py            # Core analysis pipeline
├── lesion_clustering.py                # Lesion clustering logic
├── train_leaf_segmenter.py             # Training script (optional)
│
├── requirements.txt                    # Python dependencies (REQUIRED)
├── packages.txt                        # System dependencies (optional)
├── .python-version                     # Python version for Streamlit Cloud
├── README.md
│
├── models/
│   ├── leaf_segmenter_best.pt          # Optional trained model
│   └── clusters/
│       ├── trained/
│       │   └── lesion_cluster_model.joblib
│       ├── classical/
│       │   └── lesion_cluster_model.joblib
│       └── intersection/
│           └── lesion_cluster_model.joblib
```

---

## Run Locally

```bash
git clone https://github.com/SandipGarai/blast_irc.git
cd blast_irc
pip install -r requirements.txt
streamlit run app.py
```

Then open:

```
http://localhost:8501
```

---

## Important Notes (Local Setup)

### 1. Python Version

Use **Python 3.10 or 3.11**
(PyTorch is not stable on Python 3.13)

---

### 2. Torch Warning (Optional)

If you see this warning:

```
torch.classes __path__ error
```

Run Streamlit with:

```bash
streamlit run app.py --server.fileWatcherType=none
```

---

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub
2. Go to: https://share.streamlit.io
3. Click **New app**
4. Select:
   - Repository
   - Branch
   - `app.py` as entry point

5. Click **Deploy**

Streamlit Cloud will automatically use:

- `requirements.txt`
- `packages.txt`
- `.python-version`

---

## requirements.txt (Critical)

Ensure this file exists in root:

```
streamlit
numpy
pandas
matplotlib
opencv-python-headless
scikit-learn
joblib
Pillow
torch
torchvision
altair
```

---

## Model Files

The `models/` directory is required at runtime.

### If files are large (>100MB):

- Use **Git LFS**, OR
- Host externally and download at runtime

### If missing:

- App will fall back to **classical segmentation**

---

## packages.txt (Optional)

Only needed if system packages are required:

```
libgl1
```

---

## Tech Stack

- Streamlit
- PyTorch
- OpenCV
- scikit-learn
- NumPy
- pandas
- Matplotlib
- Altair

---

## Summary

This app provides a complete pipeline for:

- Leaf segmentation
- Lesion detection
- Disease classification
- Severity estimation

Designed for **research and field-level disease monitoring**.

---

## Organization

**ICAR-IIAB, Ranchi, Jharkhand**
Rice Disease Monitoring System · Phase 1

---
