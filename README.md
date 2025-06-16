# A Polar Coordinate Attention Transformer for Ptychographic Imaging

Official implementation of **PPN (Physics-Inspired Ptychographic Network)**, accepted in our IEEE TCI paper:

> *A Physics-Inspired Deep Learning Framework with Polar Coordinate Attention for Ptychographic Imaging*  
> Han Yue, Jun Cheng, Yu-Xuan Ren, Chien-Chun Chen, Grant A. van Riessen, Philip H.W. Leong, Steve Feng Shu  
> IEEE Transactions on Computational Imaging, 2025  
> [Paper Link](https://ieeexplore.ieee.org/document/11027575)

## 🔬 Overview

Deep Learning based ptychographic imaging suffers from a geometric mismatch between conventional deep neural networks and the underlying diffraction physics. PPN addresses this gap through a dual-branch architecture that incorporates a **Polar Coordinate Attention (PoCA)** mechanism for modeling global coherence in reciprocal space, and a local ViT branch for capturing spatial features.

**Key Features:**
- Physics-inspired attention mechanism aligned with diffraction geometry
- >1000× inference speedup over iterative methods like ePIE
- Superior high-frequency preservation under low-overlap scanning
- Efficient deployment with 11× fewer parameters than baseline ViT models

**Data and Code References:**
- The amplitude and phase data used in this experiment are sourced from: [AD_LTEM](https://github.com/danielzt12/AD_LTEM).
- The code in this repository is inspired by and references the following works:
  - [PtychoNN](https://github.com/mcherukara/PtychoNN), associated with the paper: M. Cherukara et al., *APL*, 2021, [DOI: 10.1063/5.0013065](https://aip.scitation.org/doi/full/10.1063/5.0013065).
  - [Deep Phase Imaging](https://github.com/dillanchang/deep-phase-imaging), associated with the paper: D. Chang et al., *Physical Review Letters*, 2023, [DOI: 10.1103/PhysRevLett.130.016101](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.130.016101).
  - [PtyNet](https://github.com/paidaxinbao/PtyNet), associated with the paper: *Efficient ptychography reconstruction strategy by fine tuning of large pre-trained deep learning model*.

  
## 🛠 Code Structure

- `models/` – PPN model
- `train/` – Training and evaluation scripts
- `utils/` – Data loading, preprocessing, visualization, and custom callbacks
- `config.py` – Centralized configuration for paths and hyperparameters
- `main.py` – Main script to run the training and visualization pipeline


## 📖 Citation

If you use this code or models in your research, please cite:

```bibtex
@article{yue2025ppn,
  title={A Physics-Inspired Deep Learning Framework with Polar Coordinate Attention for Ptychographic Imaging},
  author={Yue, Han and Cheng, Jun and Ren, Yu-Xuan and Chen, Chien-Chun and van Riessen, Grant A. and Leong, Philip H.W. and Shu, Steve Feng},
  journal={IEEE Transactions on Computational Imaging},
  year={2025}
}
