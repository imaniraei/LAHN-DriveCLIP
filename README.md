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

<p>
<a href="https://talk2car.github.io/"><img src="https://img.shields.io/badge/Talk2Car-Official%20Dataset-555555"></a>
<a href="https://bdd-data.berkeley.edu/"><img src="https://img.shields.io/badge/BDD--X-Official%20Dataset-555555"></a>
<a href="https://www.cvlibs.net/datasets/kitti/"><img src="https://img.shields.io/badge/KITTI-Vision%20Benchmark-555555"></a>
<a href="https://public.roboflow.com/object-detection/self-driving-car"><img src="https://img.shields.io/badge/Udacity-Self--Driving%20Car-555555"></a>
</p>

---

# Highlights

- Localization-aware CLIP adaptation
- Weakly supervised vision-language grounding
- Frozen CLIP backbone with LoRA adaptation
- Gaussian CAM supervision
- Spatially-aware hard negative mining
- Efficient inference for autonomous driving applications

---

# Installation

Please follow the installation instructions provided in <a href="INSTALL.md"><u>INSTALL.md</u></a>.

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

The complete dataset preparation procedure is described in <a href="DATASETS.md"><u>DATASETS.md</u></a>.

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

Detailed instructions for training and evaluation are provided in <a href="RUN.md"><u>RUN.md</u></a>.

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




<div align="center">

### Table 1. Top-K Cross-modal Retrieval Accuracy

</div>

<div style="overflow-x:auto; width:100%;">

| Method | Talk2Car Image→Text Top-1 | Top-2 | Talk2Car Text→Image Top-1 | Top-2 | BDD-X Image→Text Top-1 | Top-2 | BDD-X Text→Image Top-1 | Top-2 |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| CLIP | 47.23±0.42 | 58.91±0.56 | 48.14±0.66 | 57.62±0.42 | 45.13±0.72 | 55.88±0.24 | 46.56±0.49 | 56.30±0.74 |
| CLIP Surgery | 78.42±0.51 | 88.87±0.43 | 78.76±0.59 | 88.91±0.48 | 77.91±0.64 | 87.15±0.38 | 78.08±0.52 | 88.26±0.46 |
| **LAHN-DriveCLIP** | **81.36±0.64** | **92.48±0.38** | **82.27±0.47** | **93.66±0.73** | **79.28±0.15** | **89.02±0.41** | **79.34±0.49** | **90.26±0.93** |

</div>

<p align="center">
<b>Table 1.</b> Top-K cross-modal retrieval accuracy comparison of CLIP, CLIP Surgery, and the proposed <b>LAHN-DriveCLIP</b> on the Talk2Car and BDD-X datasets. Results are reported as mean ± std.
</p>







<div align="center">

### Table 3. Weakly Supervised Localization Performance

</div>

<div style="overflow-x:auto; width:100%;">

| Dataset | Method | EBPG ↑ | IoU@0.5 ↑ | Point Acc ↑ | Params (M) | FLOPs (G) | Latency (ms) |
|:--|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| Talk2Car | CLIP | 41.20±5.12 | 28.45±2.34 | 47.13±6.21 | 304.3 | 188.6 | 24.1 |
|  | CLIP Surgery | 49.86±4.21 | 39.74±2.02 | 56.48±5.03 | 304.7 | 191.2 | 26.8 |
|  | **LAHN-DriveCLIP** | **61.93±3.58** | **57.74±1.55** | **68.02±4.11** | **306.8** | **189.8** | **24.6** |
| BDD-X | CLIP | 40.72±5.03 | 26.94±2.21 | 45.80±6.02 | 304.7 | 188.6 | 24.1 |
|  | CLIP Surgery | 48.95±4.10 | 36.88±1.94 | 54.72±4.88 | 304.3 | 191.2 | 26.8 |
|  | **LAHN-DriveCLIP** | **58.27±3.41** | **49.82±1.76** | **63.47±4.03** | **306.8** | **189.8** | **24.6** |
| KITTI | CLIP | 38.45±5.27 | 24.32±2.35 | 43.18±6.11 | 304.7 | 188.6 | 24.1 |
|  | CLIP Surgery | 46.73±4.18 | 33.91±2.01 | 52.07±4.95 | 304.3 | 191.2 | 26.8 |
|  | **LAHN-DriveCLIP** | *(ادامه را دقیقاً از Table 3 مقاله کپی کن)* | | | | | |
| Udacity | CLIP | *(از مقاله)* | | | | | |
|  | CLIP Surgery | *(از مقاله)* | | | | | |
|  | **LAHN-DriveCLIP** | *(از مقاله)* | | | | | |

</div>

<p align="center">
<b>Table 3.</b> Weakly supervised localization performance comparison of CLIP, CLIP Surgery, and the proposed <b>LAHN-DriveCLIP</b> across four autonomous driving datasets. Results are reported as mean ± std.
</p>






---


### BDD-X (Cross-domain)

<p align="center">
<img src="assets/BDD100k.jpg" width="95%">
</p>

To evaluate cross-domain generalization, the model is trained on Talk2Car and evaluated on the BDD-X dataset. LAHN-DriveCLIP consistently maintains strong retrieval performance under domain shift, demonstrating robust generalization across different autonomous driving environments.

---

# Weakly Supervised Localization

<p align="center">
<img src="assets/average.png" width="55%">
</p>

Average localization performance across all driving datasets.

The proposed framework consistently achieves the highest performance in

- EBPG
- IoU@0.5
- Point Accuracy

while introducing only negligible computational overhead.

---

<p align="center">
<img src="assets/iou_barplot.png" width="55%">
</p>

Comparison of localization performance using IoU@0.5.

The proposed localization-aware adaptation strategy significantly improves spatial grounding over both pretrained CLIP and CLIP Surgery.

---

# LoRA Ablation Study

The influence of LoRA configuration is analyzed using two different CLIP backbones.

---

<p align="center">
<img src="assets/Localization_Acc_IoU0.5_ViT-L14.png" width="43%">
&nbsp;&nbsp;&nbsp;
<img src="assets/Localization_Acc_IoU0.5_RN50.png" width="43%">
</p>

<p align="center">
<b>Left:</b> ViT-L/14 backbone. &nbsp;&nbsp;&nbsp; <b>Right:</b> RN50 backbone.
</p>

Higher-capacity LoRA configurations achieve faster convergence and higher localization accuracy throughout training. For the ViT-L/14 backbone, the configuration **r = 16** and **α = 32** provides the best balance between localization accuracy and computational efficiency. The RN50 backbone exhibits a similar trend, confirming that moderate LoRA ranks provide an effective trade-off between efficiency and localization performance.

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


