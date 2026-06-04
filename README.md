# FAL-Net
# FAL-Net: A Hybrid LoRA-Enhanced Architecture with Spatial Attention for Robust Fingerprint Liveness Detection

Official implementation of the paper:

**"FAL-Net: A Hybrid LoRA-Enhanced Architecture with Spatial Attention for Robust Fingerprint Liveness Detection"**

## Overview

Fingerprint Presentation Attack Detection (PAD), also known as Fingerprint Liveness Detection (FLD), is a critical component of modern biometric security systems. Existing deep learning approaches often suffer from limited generalization to unseen spoof materials, cross-sensor variability, or high computational costs.

To address these challenges, we propose **FAL-Net**, a hybrid deep learning architecture that combines:

* A **ResNet50 encoder** for hierarchical feature extraction.
* A **U-Net decoder** for multi-scale feature reconstruction.
* **Soft Spatial Attention (SSA)** modules to automatically focus on spoof-sensitive regions without explicit fingerprint segmentation.
* **Low-Rank Adaptation (LoRA)** for parameter-efficient task adaptation.
* A **Hierarchical Feature Fusion** mechanism that integrates local texture information and global ridge patterns.

The proposed framework is designed to improve robustness against unknown spoof materials and sensor variability while maintaining efficient adaptation capabilities.

## Main Contributions

* Attention-guided multi-scale fingerprint analysis.
* Parameter-efficient adaptation through LoRA.
* Hierarchical fusion of texture and semantic fingerprint features.
* Comprehensive evaluation under ISO/IEC 30107-3 biometric PAD protocols.
* State-of-the-art performance on the LivDet 2017 benchmark.
* Strong cross-sensor generalization on LivDet datasets.

## Architecture

The architecture consists of:

1. Constraint-Driven Attention-Based Representation Learning.
2. Coupled Parameter-Efficient Adaptation Strategy.
3. Hierarchical Texture--Semantic Fusion.
4. ISO-Compliant Evaluation under Cross-Sensor Protocols.
5. Robust Cross-Domain Generalization.

## Datasets

Experiments were conducted using the publicly available LivDet datasets:

* LivDet 2011
* LivDet 2013
* LivDet 2015
* LivDet 2017

Dataset access information is available from: https://livdet.org/

## Experimental Results

FAL-Net achieves competitive performance across multiple benchmark datasets and establishes state-of-the-art performance on LivDet 2017.

| Dataset     | Accuracy |
| ----------- | -------- |
| LivDet 2011 | 98.35%   |
| LivDet 2013 | 99.52%   |
| LivDet 2015 | 96.08%   |
| LivDet 2017 | 99.49%   |

## Installation

```bash
git clone https://github.com/your_username/FAL-Net.git
cd FAL-Net

pip install -r requirements.txt
```

## Training

```bash
python train_ablation.py
```

## Evaluation

```bash
python ablation_runner.py
```

## Reproducibility

To facilitate reproducibility and independent verification of the reported results, this repository provides:

* Source code
* Model architecture
* Training configuration
* Evaluation scripts
* Pretrained weights (to be released)

All experiments reported in the paper can be reproduced using the provided scripts and settings.

## Citation

If you use this work in your research, please cite:

```bibtex
@article{ammar2026falnet,
  title={FAL-Net: A Hybrid LoRA-Enhanced Architecture with Spatial Attention for Robust Fingerprint Liveness Detection},
  author={Bouhamed, Sonda Ammar and others},
  journal={IEEE Transactions on Biometrics, Behavior, and Identity Science},
  year={2026}
}
```

## License

This project is released for academic and research purposes.

