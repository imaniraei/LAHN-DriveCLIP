<h1 align="center">
LAHN-DriveCLIP: Localization-Aware CLIP Adaptation with CAM-Guided Hard Negative Mining for Autonomous Driving
</h1>

<p align="center">
<a href="https://imaniraei.github.io/">Iman Iraei</a>,
<a href="https://users.encs.concordia.ca/~omair/">M. Omair Ahmad</a>,
<a href="https://users.encs.concordia.ca/~swamy/">M. N. S. Swamy</a>
</p>

<p align="center">
Department of Electrical and Computer Engineering<br>
Concordia University, Montreal, QC, Canada
</p>

<p align="center">
<a href="#"><img src="https://img.shields.io/badge/Paper-RAS-green"></a>
<a href="#overview"><img src="https://img.shields.io/badge/Overview-Method-blue"></a>
<a href="#datasets"><img src="https://img.shields.io/badge/Datasets-4%20Benchmarks-orange"></a>
<a href="#"><img src="https://img.shields.io/badge/Code-Coming%20Soon-red"></a>
<a href="#"><img src="https://img.shields.io/badge/Status-Under%20Review-yellow"></a>
<a href="#citation"><img src="https://img.shields.io/badge/Citation-BibTeX-purple"></a>
</p>

---

# Overview

<p align="center">
<img src="assets/pipeline.png" width="100%">
</p>

**LAHN-DriveCLIP** is a localization-aware vision-language framework designed for weakly supervised object localization in autonomous driving. The proposed framework extends the CLIP foundation model through parameter-efficient adaptation while preserving the powerful visual-semantic knowledge learned during large-scale pretraining.

Unlike conventional CLIP adaptation methods that primarily optimize global image-text similarity, LAHN-DriveCLIP explicitly incorporates spatial supervision into the learning process. The framework introduces Gaussian CAM Alignment to encourage spatial consistency between class activation maps and object locations, together with a Localization-Aware Hard Negative Mining strategy that emphasizes spatially confusing image-text pairs during contrastive learning.

Instead of fully fine-tuning CLIP, lightweight Low-Rank Adaptation (LoRA) modules are inserted into both the vision and text transformers, allowing efficient domain adaptation while keeping the pretrained backbone frozen. This design substantially reduces computational overhead while improving both localization accuracy and cross-modal retrieval performance.

Extensive experiments demonstrate that LAHN-DriveCLIP consistently outperforms pretrained CLIP and CLIP Surgery across multiple autonomous driving datasets while maintaining interpretable activation maps and strong vision-language alignment.

---

# Method

The proposed training pipeline consists of four tightly coupled components:

### 1. LoRA-based CLIP Adaptation

The pretrained CLIP vision encoder and text encoder remain frozen during training. Instead, lightweight Low-Rank Adaptation (LoRA) modules are inserted into the transformer attention layers, allowing efficient adaptation with only a small number of trainable parameters.


### 2. Gaussian CAM Alignment

To introduce localization awareness without adding an object detector, LAHN-DriveCLIP generates gScoreCAM activation maps from CLIP similarity scores.

Ground-truth bounding boxes are converted into Gaussian spatial targets, and the generated activation maps are aligned with these targets through a CAM alignment loss, encouraging spatially accurate visual attention while preserving CLIP's original contrastive learning paradigm.


### 3. CAM-Guided Hard Negative Mining

Rather than treating all negative image-text pairs equally, LAHN-DriveCLIP identifies difficult negatives according to the spatial overlap between their activation maps.

Image-text pairs producing similar activation regions receive larger weights during contrastive learning, enabling stronger discrimination between visually confusing driving scenes.


### 4. Joint Optimization

The final training objective jointly optimizes:

- Contrastive image-text alignment
- Gaussian CAM alignment
- CAM-guided hard negative mining

During optimization, gradients update only the LoRA parameters while the original CLIP backbone remains frozen.

---

# Key Features

- 🚗 Localization-aware adaptation of CLIP for autonomous driving
- 🧠 Parameter-efficient fine-tuning using LoRA
- 🔥 Gaussian CAM Alignment for weakly supervised localization
- 🎯 CAM-guided Hard Negative Mining
- 🌍 Strong cross-domain generalization
- ⚡ Lightweight training with frozen CLIP encoders
- 📈 Improved cross-modal retrieval and localization performance
- 🔍 Interpretable class activation maps

---

# Datasets

The proposed framework is evaluated on four public autonomous driving datasets.

| Dataset | Purpose |
|----------|----------|
| Talk2Car | In-domain training and evaluation |
| BDD-X | Cross-domain retrieval and localization |
| KITTI | Cross-domain localization |
| Udacity Self-Driving Car | Cross-domain localization |

### Official Dataset Links

- Talk2Car Dataset
- BDD-X Dataset
- KITTI Vision Benchmark
- Udacity Self-Driving Car Dataset

---

# Highlights

- Localization-aware CLIP adaptation
- Weakly supervised vision-language grounding
- Frozen CLIP backbone with LoRA adaptation
- Gaussian CAM supervision
- Spatially-aware hard negative mining
- Efficient inference for autonomous driving applications





---

# Repository Structure

```text
LAHN-DriveCLIP/
│
├── assets/
│   ├── pipeline.png
│   ├── LoRA.jpg
│   ├── Talk2Car.jpg
│   ├── BDD100k.jpg
│   ├── average.png
│   ├── iou_barplot.png
│   ├── Localization_Acc_IoU0.5_ViT-L14.png
│   └── Localization_Acc_IoU0.5_RN50.png
│
├── configs/
│
├── datasets/
│
├── losses/
│
├── models/
│
├── notebooks/
│
├── outputs/
│
├── logs/
│
├── scripts/
│
├── utils/
│
├── README.md
├── INSTALL.md
├── DATASETS.md
├── RUN.md
├── requirements.txt
├── LICENSE
└── .gitignore
```

---

# Installation

Please follow the installation instructions provided in **INSTALL.md**.

The installation guide includes

- Environment setup
- Python dependencies
- CUDA compatibility
- GPU requirements
- HPC installation
- Google Colab inference
- Troubleshooting

---

# Data Preparation

The complete dataset preparation procedure is described in **DATASETS.md**.

The repository supports the following autonomous driving datasets

| Dataset | Purpose |
|----------|----------|
| Talk2Car | Training & Evaluation |
| BDD-X | Cross-domain Evaluation |
| KITTI | Cross-domain Evaluation |
| Udacity Self-Driving Car | Cross-domain Evaluation |

The data preparation guide includes

- Download links
- Folder organization
- Annotation format
- Image preprocessing
- Train / Validation split
- Evaluation protocol

---

# Training and Evaluation

Detailed instructions for training and evaluation are provided in **RUN.md**.

Supported workflows include

- Model training
- Evaluation
- Cross-modal retrieval
- Weakly-supervised localization
- Visualization using gScoreCAM
- Inference on custom images
- Benchmark reproduction



---

# LoRA Adaptation

<p align="center">
<img src="assets/LoRA.jpg" width="45%">
</p>

LAHN-DriveCLIP employs parameter-efficient Low-Rank Adaptation (LoRA) to adapt both the vision encoder and the text encoder of CLIP.

Instead of updating hundreds of millions of pretrained parameters, only lightweight low-rank matrices are optimized, resulting in

- lower GPU memory usage
- faster convergence
- improved generalization
- negligible computational overhead

This design makes LAHN-DriveCLIP particularly suitable for large-scale autonomous driving applications.

---



# Experimental Results

The proposed **LAHN-DriveCLIP** is extensively evaluated on four autonomous driving benchmarks under both in-domain and cross-domain settings. Performance is assessed using two complementary tasks:

- Cross-modal Retrieval
- Weakly Supervised Localization

Compared with pretrained CLIP and CLIP Surgery, LAHN-DriveCLIP consistently achieves superior localization accuracy while preserving efficient inference and parameter efficiency.

---

# Cross-modal Retrieval

### Talk2Car (In-domain)

<p align="center">
<img src="assets/Talk2Car.jpg" width="95%">
</p>

The proposed LAHN-DriveCLIP achieves the highest retrieval accuracy on the Talk2Car benchmark for both image-to-text and text-to-image retrieval. By combining LoRA adaptation, Gaussian CAM Alignment, and CAM-guided Hard Negative Mining, the model significantly improves semantic alignment between visual scenes and natural language descriptions.

---

### BDD-X (Cross-domain)

<p align="center">
<img src="assets/BDD100k.jpg" width="95%">
</p>

To evaluate cross-domain generalization, the model is trained on Talk2Car and evaluated on the BDD-X dataset. LAHN-DriveCLIP consistently maintains strong retrieval performance under domain shift, demonstrating robust generalization across different autonomous driving environments.

---

# Weakly Supervised Localization

<p align="center">
<img src="assets/average.png" width="90%">
</p>

Average localization performance across all driving datasets.

The proposed framework consistently achieves the highest performance in

- EBPG
- IoU@0.5
- Point Accuracy

while introducing only negligible computational overhead.

---

<p align="center">
<img src="assets/iou_barplot.png" width="85%">
</p>

Comparison of localization performance using IoU@0.5.

The proposed localization-aware adaptation strategy significantly improves spatial grounding over both pretrained CLIP and CLIP Surgery.

---

# LoRA Ablation Study

The influence of LoRA configuration is analyzed using two different CLIP backbones.

---

### ViT-L/14 Backbone

<p align="center">
<img src="assets/Localization_Acc_IoU0.5_ViT-L14.png" width="70%">
</p>

Higher-capacity LoRA configurations achieve faster convergence and higher localization accuracy throughout training. The configuration **r = 16** and **α = 32** provides the best balance between localization accuracy and computational efficiency.

---

### RN50 Backbone

<p align="center">
<img src="assets/Localization_Acc_IoU0.5_RN50.png" width="70%">
</p>

The RN50 backbone exhibits similar trends, confirming that moderate LoRA ranks provide the most effective trade-off between efficiency and localization performance.

---

# Qualitative Results

The qualitative examples presented in the paper demonstrate progressively improved activation maps from

- CLIP
- CLIP Surgery
- CLIP + LoRA
- CLIP + LoRA + CAM
- **LAHN-DriveCLIP**

The proposed framework produces more accurate and object-focused localization maps while suppressing irrelevant background regions, leading to better spatial grounding in complex autonomous driving scenes.

---

# Future Work

Future work will extend LAHN-DriveCLIP from weakly supervised object localization toward weakly supervised semantic segmentation by leveraging Segment Anything Model (SAM) to generate pseudo masks from the refined CAM localization maps.

---

# Status

🚧 **The manuscript is currently under review at Robotics and Autonomous Systems (RAS).**

---

# Citation

If you find this work useful, please consider citing

```bibtex
@article{iraei2026lahndriveclip,
  title={LAHN-DriveCLIP: Localization-Aware CLIP Adaptation with CAM-Guided Hard Negative Mining for Autonomous Driving},
  author={Iraei, Iman and Ahmad, M. Omair and Swamy, M. N. S.},
  journal={Submitted to Robotics and Autonomous Systems},
  year={2026}
}
```

---

# License

This project is released under the **MIT License**.


