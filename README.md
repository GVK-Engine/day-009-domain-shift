# Day 9 - Domain Shift: KITTI Germany vs nuScenes Singapore

> MS Robotics & Autonomous Systems Engineering - Arizona State University - Dec 2026

---

## The Question

A detector tuned on German roads.
Deployed in Singapore.
How much does it degrade - and why?

This is the problem every AV company faces when expanding to a new city.
Waymo trained in San Francisco. Deployed in Phoenix. The model failed.
I measured it myself on two real datasets.

---

## Live Demo

![Domain Shift Demo](https://drive.google.com/uc?id=1CXS5oFHJOl4ITEiQPBkO5QiIMo_RQ-K7)

*Left: KITTI Germany (30 detections). Right: nuScenes Singapore (12 detections). Same detector. Same parameters.*

---

## 4-Panel Comparison - Camera + BEV

![Comparison Frame](https://drive.google.com/uc?id=1rRRI6jn6cQEVn5Ao-DHepBVVG_X2eWvx)

Real road scenes from both cities with detection dots and distances labeled.
KITTI Germany road on the left. nuScenes Singapore street on the right.
BEV point clouds below - the density difference is immediately visible.

---

## Key Finding

![Domain Shift Analysis](https://drive.google.com/uc?id=14EY5xW_nlQv7Iyj0fk6FEGJm4G4qAWmK)

![Evaluation Chart]()

| Metric | Value |
|--------|-------|
| KITTI avg detections | 30.4 / frame |
| nuScenes avg detections | 12.6 / frame |
| Detection drop | 58.4% |
| KITTI consistency | 90.5% |
| nuScenes consistency | 70.6% |
| KITTI points/scan | 121,855 |
| nuScenes points/scan | 34,722 |
| Point drop | 71.5% |
| Root cause | Sensor not scene |

---

## The Finding Nobody States Clearly

Most papers assume domain shift comes from scene differences.
Different roads. Different countries. Different objects.

**My data proves otherwise.**

```
Detection drop : 58.4%
Point drop     : 71.5%
```

The drops are proportional. The algorithm never changed.
The scene did not cause the failure. **The sensor did.**

```
KITTI sensor    : Velodyne HDL-64E   64 beams   121,855 pts/scan
nuScenes sensor : Velodyne HDL-32E   32 beams    34,722 pts/scan
```

Half the beams means half the points per object.
Fewer points per object means harder clustering.
Harder clustering means fewer detections.

The DBSCAN parameters were tuned on HDL-64E density.
They do not transfer to HDL-32E without retuning.
This is sensor-driven domain shift, not scene-driven.

---

## Why Consistency Matters More Than Mean Accuracy

```
KITTI consistency    90.5%   range 25-37 objects
nuScenes consistency 70.6%   range 7-23 objects
```

The nuScenes detector is not just less accurate.
It is **unpredictable**.

A detector that finds 30 objects reliably is safer
than one that finds 23 objects sometimes and 7 sometimes.
You cannot build a safety system around unpredictable behavior.

This is why sensor specification matters in the ODD.
Waymo publishes their sensor requirements.
A system certified for HDL-64E cannot simply run on HDL-32E
without re-evaluation. My numbers show exactly why.

---

## BEV Point Cloud Comparison

![BEV Comparison](https://drive.google.com/uc?id=10-D2QbnBb8yaBKPKuX65ZZ8v5OZvXeA4)

The density difference is visible. KITTI fills the entire BEV frame.
nuScenes is sparse with large gaps between scan lines.
Those gaps are where objects disappear.

---

## What This Connects To

```
Day 1: RANSAC + DBSCAN detector built on KITTI
       Parameters tuned for HDL-64E density

Day 9: Same detector on nuScenes HDL-32E
       58.4% detection drop
       Root cause: sensor not scene

Next:  Adapt parameters for HDL-32E density
       Measure how much retuning recovers
       This is domain adaptation
```

The detector from Day 1 was built for a specific sensor.
Deploying it on a different sensor without retuning
is exactly the mistake AV companies make in practice.
The fix is either retuning parameters or
training a sensor-agnostic representation.
Both are active research areas. My data motivates why.

---

## Run It Yourself

```bash
git clone https://github.com/GVK-Engine/day-009-domain-shift
cd day-009-domain-shift
pip install -r requirements.txt
```

Update `KITTI_DIR` and `NUSCENES_BASE` to your local paths.

```bash
# test detector on both datasets
py -3.11 detector.py

# run full domain shift analysis
py -3.11 domain_shift.py

# evaluate consistency and statistics
py -3.11 evaluate.py

# generate 4-panel video and GIF
py -3.11 visualize.py
```

Datasets:
- KITTI: https://www.cvlibs.net/datasets/kitti/raw_data.php
- nuScenes: https://www.nuscenes.org/nuscenes

---

## Project Structure

```
day-009-domain-shift/
├── detector.py         RANSAC+DBSCAN pipeline for KITTI and nuScenes
├── domain_shift.py     Cross-dataset detection comparison
├── evaluate.py         Consistency and variance analysis
├── visualize.py        4-panel camera+BEV demo video
├── projection.py       LiDAR-camera projection (from Day 8)
├── requirements.txt
└── results/
    ├── domain_shift_analysis.png
    ├── comparison_frame.png
    ├── evaluation_chart.png
    ├── domain_shift_demo.gif
    └── bev_comparison.png
```

---

## Stack

`Python 3.11` `NumPy` `OpenCV` `SciPy` `Matplotlib` `imageio` `KITTI` `nuScenes`

---

## Series 1 Progress

| # | Project | Finding | Status |
|---|---------|---------|--------|
| P1.1 | LiDAR Obstacle Detection | 0.4m voxel creates ghost detections | ✅ |
| P1.2 | Stereo Camera Depth Safety | Camera unsafe beyond 10m | ✅ |
| P1.3 | PointPillars 3D Detector | 98.9% loss reduction from scratch | ✅ |
| P1.4 | Multi-Camera BEV Perception | 178 objects from 6 cameras | ✅ |
| P1.5 | Multi-Object Tracking SORT | Detector is bottleneck - not tracker | ✅ |
| P1.6 | Semantic Segmentation ROS2 | 52.6 FPS - warmup cost measured | ✅ |
| P1.7 | Adverse Weather Analysis | Fog unsafe below 75m visibility | ✅ |
| P1.8 | LiDAR-Camera Depth Completion | 44x MAE improvement at 0-10m | ✅ |
| P1.9 | Domain Shift Analysis | 58.4% drop - sensor not scene | ✅ |
