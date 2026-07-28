# Dataset Preparation

LAHN-DriveCLIP is evaluated on four public autonomous driving datasets.

| Dataset | Purpose |
|----------|---------|
| Talk2Car | In-domain retrieval and localization |
| BDD-X | Cross-domain retrieval and localization |
| KITTI | Cross-domain localization |
| Udacity Self-Driving Car | Cross-domain localization |

---

# Manifest Format

Each training sample is stored as one JSON object.

```json
{
  "sample_id":"001",
  "image_path":"/data/a.jpg",
  "text":"the car on the left",
  "bbox_xyxy":[10,20,100,150],
  "split":"test",
  "dataset":"talk2car",
  "metadata":{}
}
```

---

# Build Talk2Car Manifest

```bash
python tools/build_manifest.py talk2car \
    --json /data/talk2car/commands_train.json \
    --image-root /data/talk2car/images \
    --output /data/manifests/talk2car_train.jsonl \
    --split train
```

---

# Build BDD-X Manifest

```bash
python tools/build_manifest.py bddx \
    --csv /data/bddx/BDD-X-Annotations_v1.csv \
    --split-file /data/bddx/train.txt \
    --image-root /data/bddx/representative_frames \
    --frame-template '{video_id}.jpg' \
    --output /data/manifests/bddx_train.jsonl \
    --split train
```

BDD-X annotations are provided at the video level. A representative frame should be selected for every annotated driving instruction.

---

# Build KITTI / Udacity Localization Manifests

Prepare a CSV file

```text
image_path,text,x1,y1,x2,y2,sample_id
```

then execute

```bash
python tools/build_manifest.py localization_csv \
    --csv /data/kitti/prompts_boxes.csv \
    --dataset kitti \
    --split test \
    --output /data/manifests/kitti_localization.jsonl
```

---

# Expected Directory Structure

```
datasets/
│
├── Talk2Car/
├── BDD-X/
├── KITTI/
└── Udacity/
```
