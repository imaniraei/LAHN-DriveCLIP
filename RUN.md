# Training and Evaluation

This document summarizes the training and evaluation pipeline of LAHN-DriveCLIP.

---

# Training

Modify the dataset paths inside

```
configs/talk2car_lora.yaml
```

Then launch training

```bash
python train.py \
    --config configs/talk2car_lora.yaml
```

---

# HPC Training

```bash
sbatch scripts/slurm/train_lora.sbatch
```

---

# Retrieval Evaluation

Talk2Car

```bash
python evaluate_retrieval.py \
    --config configs/talk2car_lora.yaml \
    --checkpoint outputs/talk2car_lora/last.pt \
    --output outputs/talk2car_retrieval.json
```

BDD-X

```bash
python evaluate_retrieval.py \
    --config configs/bddx_cross_domain.yaml \
    --checkpoint outputs/talk2car_lora/last.pt \
    --output outputs/bddx_cross_domain_retrieval.json
```

---

# Localization Evaluation

```bash
python evaluate_localization.py \
    --config configs/kitti_localization.yaml \
    --checkpoint outputs/talk2car_lora/last.pt \
    --method gscorecam \
    --gscorecam-path gScoreCAM \
    --output outputs/kitti_localization.json
```

---

# Output Directory

```
outputs/
│
├── checkpoints/
├── retrieval/
└── localization/
```

---

# Notes

- Only the LoRA parameters are updated during training.
- The original CLIP backbone remains frozen.
- Both the vision encoder and text encoder are adapted.
- The default implementation uses CLIP **ViT-L/14**.
