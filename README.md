# Robust Multimodal Age Verification for Online Gaming Chat

This project implements a robust audio-based age verification pipeline designed for online gaming chat environments. The system aims to classify speakers as **minor** or **adult** using speech signals while improving robustness against spoofing attacks and noisy audio conditions commonly encountered in real-world voice communication platforms.

The pipeline integrates speech data from multiple datasets, including **Common Voice**, **MyST**, and **VoxCeleb**, and supports large-scale dataset construction, preprocessing, speaker-level splitting, spoof target preparation, spoof generation, and model training. In addition to standard clean speech processing, the project includes dedicated pipelines for spoof-aware experimentation using methods such as **Text-to-Speech (TTS)**, **Voice Conversion (VC)**, and replay-based attacks.

The repository is organized into modular stages covering:
- dataset inventory construction
- speaker pool generation
- preprocessing and segmentation
- spoof target validation
- spoof generation
- model training and evaluation

## Project Workflow

Run the project in this order:

### 1. Dataset Inventory Setup

All dataset inventory notebooks are located inside:

```text
dataset_setup/
```

These files prepare file-level and speaker-level metadata before the main pipeline.

| Dataset | File | Purpose |
|---|---|---|
| Common Voice | `dataset_setup/cv_inventory_setup.ipynb` | Builds Common Voice file metadata and speaker inventory |
| MyST | `dataset_setup/myst_inventory_setup.ipynb` | Builds MyST file metadata and speaker inventory |
| VoxCeleb | `dataset_setup/voxceleb_inventory_setup.ipynb` | Builds VoxCeleb segment metadata and speaker inventory |

These files operate before preprocessing and mainly use raw file durations and metadata.

---

### 2. Build Speaker Pools

After creating the dataset inventories, run:

```text
build_pools.ipynb
```

This file merges the speaker inventories from all datasets and creates the main speaker pools:

- `adult_real_candidates`
- `minor_real_candidates`
- `adult_spoof_targets`
- `minor_spoof_targets`

It also balances the adult pools and saves the final pool files used by the main pipeline.

---

### 3. Main Pipeline

After building the speaker pools, run:

```text
main.ipynb
```

This is the main project pipeline. It includes:

- speaker-level train/validation/test splitting
- unified file manifest construction
- real candidate preprocessing
- spoof target preprocessing
- backup data processing
- clean dataset preparation
- balanced clean/spoof split
- local runtime setup
- model training and evaluation

---

### 4. Spoof Target VAD Filtering

After the standard spoof target preprocessing, an external VAD-based filtering file is used:

```text
spoof_target_vad_filtering.ipynb
```

This step keeps only spoof target files that satisfy:

```python
vad_status == "success"
speech_duration_sec >= 7
```

These filtered files are then used as valid reference recordings for spoof generation.

---

### 5. Spoof Generation

Spoof generation is handled in a separate external file.

This step uses:

- `spoof_source_clean`
- `spoof_targets`

to generate spoofed samples using methods such as:

- Text-to-Speech (TTS)
- Voice Conversion (VC)
- Replay attacks

