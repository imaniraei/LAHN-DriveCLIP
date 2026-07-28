# Installation

This document describes how to install LAHN-DriveCLIP and verify that the environment is correctly configured.

---

## Clone the Repository

```bash
git clone https://github.com/imaniraei/LAHN-DriveCLIP.git
cd LAHN-DriveCLIP
```

---

## Create the Conda Environment

```bash
conda env create -f environment.yml
conda activate lahn-driveclip
```

---

## Install gScoreCAM

```bash
bash scripts/setup_gscorecam.sh
```

---

## Verify the Installation

```bash
pytest -q
```

Successful execution indicates that the environment has been correctly configured.

---

## Inspect LoRA Targets

```bash
python tools/inspect_model.py --model-name ViT-L-14 --pretrained openai
```

The default implementation inserts LoRA modules into both the vision and text transformer encoders while keeping the original CLIP backbone frozen.

---

## Project Structure

```
LAHN-DriveCLIP
│
├── configs/
├── lahn_driveclip/
├── notebooks/
├── scripts/
├── tests/
├── tools/
├── train.py
├── evaluate_retrieval.py
├── evaluate_localization.py
└── demo.py
```
