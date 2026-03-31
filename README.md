# MDF-Net: Decoding the FVH-DWI Mismatch for Acute Ischemic Stroke

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📖 Overview
The FVH-DWI mismatch is a crucial neuroimaging biomarker for identifying the salvageable ischemic penumbra in acute ischemic stroke (AIS). **MDF-Net** is a novel, end-to-end multimodal segmentation framework tailored to automate and quantify this mismatch. 

It explicitly addresses three critical bottlenecks in current CAD systems:
1. **Severe cross-modal feature dilution** between high-contrast DWI and low-SNR FLAIR.
2. **Epistemic uncertainty** caused by complex vascular artifacts (e.g., slow anterograde flow mimicking true functional FVH).
3. **Optimization conflict (Gradient Competition)** between macroscopic region extraction and microscopic boundary localization.

## ✨ Key Highlights
* **Dual-Stream Encoder:** Independently preserves modality-specific semantics to prevent premature feature dilution.
* **Evidential Deep Learning (EDL) Fusion:** Dynamically quantifies voxel-wise spatial uncertainty using Dirichlet priors, adaptively suppressing misleading low-SNR artifacts while ensuring robust evidence aggregation.
* **Boundary-Aware Task Decoupling (BD-Net):** Imposes explicit first-derivative edge supervision using spatial gradient operators, effectively resolving gradient competition to yield high-fidelity lesion contours.

## 🗂️ Repository Structure

```text
MDF-Net/
├── models/
│   ├── __init__.py
│   ├── mdf_net.py          # The main MDF-Net architecture
│   ├── modules.py          # Dual-stream encoder & BD-Net Decoder
│   └── evidential.py       # EDL Fusion & Uncertainty quantification
├── utils/
│   └── losses.py           # Composite loss (KL Divergence + Boundary BCE)
└── README.md