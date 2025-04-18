# PPN: Polar Coordinate Attention Network for Ptychographic Imaging

Official implementation of **PPN (Physics-Inspired Ptychographic Network)**, proposed in our IEEE TCI paper:

> *A Physics-Inspired Deep Learning Framework with Polar Coordinate Attention for Ptychographic Imaging*  
> Han Yue, Jun Cheng, Yu-Xuan Ren, Chien-Chun Chen, Grant A. van Riessen, Philip H.W. Leong, Steve Feng Shu  
> IEEE Transactions on Computational Imaging, 2024  
> [Paper Link (when available)]()

## 🔬 Overview

Ptychographic imaging suffers from a geometric mismatch between conventional deep neural networks and the underlying diffraction physics. PPN addresses this gap through a dual-branch architecture that incorporates a **Polar Coordinate Attention (PoCA)** mechanism for modeling global coherence in reciprocal space, and a local ViT branch for capturing spatial features.

**Key Features:**
- Physics-informed attention mechanism aligned with diffraction geometry
- >1000× inference speedup over iterative methods like ePIE
- Superior high-frequency preservation under low-overlap scanning
- Efficient deployment with 11× fewer parameters than baseline ViT models

## 🛠 Code Structure

- `models/` – PPN architecture and modules (ViT branch, PoCA branch, decoder)
- `scripts/` – Training, evaluation, and stitching routines
- `datasets/` – Data loading pipelines for simulated and experimental datasets
- `utils/` – Loss functions, metrics, and preprocessing

## 📖 Citation

If you use this code or models in your research, please cite:

```bibtex
@article{yue2025ppn,
  title={A Physics-Inspired Deep Learning Framework with Polar Coordinate Attention for Ptychographic Imaging},
  author={Yue, Han and Cheng, Jun and Ren, Yu-Xuan and Chen, Chien-Chun and van Riessen, Grant A. and Leong, Philip H.W. and Shu, Steve Feng},
  journal={IEEE Transactions on Computational Imaging},
  year={2025}
}
