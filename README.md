# Robust Multimodal Age Verification for Online Gaming Chat

This project builds an audio-based age verification pipeline for online gaming chat.  
The system prepares speech data from Common Voice, MyST, and VoxCeleb, then trains models to classify speakers as **minor** or **adult**, with robustness against spoofing and noise.

## Project Workflow

Run the project in this order:

### 1. Dataset Inventory Setup

Each dataset has a separate inventory setup file.  
These files prepare file-level and speaker-level metadata before the main pipeline.

| Dataset | File | Purpose |
|---|---|---|
| Common Voice | `cv_inventory_setup.py` | Builds Common Voice file metadata and speaker inventory |
| MyST | `myst_inventory_setup.py` | Builds MyST file metadata and speaker inventory |
| VoxCeleb | `voxceleb_inventory_setup.py` | Builds VoxCeleb segment metadata and speaker inventory |

These files operate before preprocessing and mainly use raw file durations and metadata. :contentReference[oaicite:0]{index=0} :contentReference[oaicite:1]{index=1} :contentReference[oaicite:2]{index=2}

---

### 2. Build Speaker Pools

After creating the dataset inventories, run:

```text
build_pools_py.py
