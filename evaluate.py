# evaluate.py
# compares detection statistics across both datasets
# measures consistency and variance of domain shift
# generates summary statistics for README
#
# Nani — MS Robotics ASU

import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from detector import detect

KITTI_DIR = (
    r"C:\Users\vamsh\Downloads\kitti"
    r"\2011_09_26_drive_0001_sync"
    r"\2011_09_26"
    r"\2011_09_26_drive_0001_sync"
    r"\velodyne_points\data"
)
NUSCENES_LIDAR = (
    r"D:\day-004-bev-perception\samples\LIDAR_TOP"
)
RESULTS_DIR = "results"


def detection_stats(det_counts):
    # basic statistics for a list of detection counts
    arr = np.array(det_counts)
    return {
        'mean':   float(arr.mean()),
        'std':    float(arr.std()),
        'min':    int(arr.min()),
        'max':    int(arr.max()),
        'median': float(np.median(arr)),
    }


def run_evaluation(n_frames=30):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    kitti_files = sorted([
        os.path.join(KITTI_DIR, f)
        for f in os.listdir(KITTI_DIR)
        if f.endswith('.bin')
    ])[:n_frames]

    nu_files = sorted([
        os.path.join(NUSCENES_LIDAR, f)
        for f in os.listdir(NUSCENES_LIDAR)
        if f.endswith('.pcd.bin')
    ])[:n_frames]

    print(f"evaluating {len(kitti_files)} kitti frames")
    print(f"evaluating {len(nu_files)} nuscenes frames")

    kitti_dets = []
    for f in kitti_files:
        c, _ = detect(f, 'kitti')
        kitti_dets.append(len(c))

    nu_dets = []
    for f in nu_files:
        c, _ = detect(f, 'nuscenes')
        nu_dets.append(len(c))

    ks = detection_stats(kitti_dets)
    ns = detection_stats(nu_dets)

    drop      = (ks['mean'] - ns['mean']) / ks['mean'] * 100
    # consistency: how stable are detections frame to frame
    k_consist = (1 - ks['std'] / ks['mean']) * 100
    n_consist = (1 - ns['std'] / ns['mean']) * 100

    print(f"\nkitti    mean:{ks['mean']:.1f}  "
          f"std:{ks['std']:.1f}  "
          f"range:{ks['min']}-{ks['max']}")
    print(f"nuscenes mean:{ns['mean']:.1f}  "
          f"std:{ns['std']:.1f}  "
          f"range:{ns['min']}-{ns['max']}")
    print(f"\ndrop         {drop:.1f}%")
    print(f"kitti consistency   {k_consist:.1f}%")
    print(f"nuscenes consistency {n_consist:.1f}%")

    return {
        'kitti_dets':  kitti_dets,
        'nu_dets':     nu_dets,
        'kitti_stats': ks,
        'nu_stats':    ns,
        'drop':        drop,
        'k_consist':   k_consist,
        'n_consist':   n_consist,
    }


def plot_evaluation(r):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor('#1a1a1a')
    fig.suptitle(
        'Detection Consistency Analysis\n'
        'Vamshikrishna Gadde  |  MS Robotics ASU',
        color='white', fontsize=12
    )

    def style(ax):
        ax.set_facecolor('#1a1a1a')
        ax.tick_params(colors='white')
        for sp in ax.spines.values():
            sp.set_edgecolor('#444')

    # frame by frame detection counts
    ax = axes[0]
    style(ax)
    ax.plot(r['kitti_dets'], 'o-',
            color='#00C8FF', linewidth=1.5,
            markersize=4, label='KITTI Germany')
    ax.plot(r['nu_dets'], 's-',
            color='#FF6B35', linewidth=1.5,
            markersize=4, label='nuScenes Singapore')
    ax.axhline(r['kitti_stats']['mean'],
               color='#00C8FF', linestyle='--',
               linewidth=1, alpha=0.5)
    ax.axhline(r['nu_stats']['mean'],
               color='#FF6B35', linestyle='--',
               linewidth=1, alpha=0.5)
    ax.set_title('Detections per Frame',
                 color='white')
    ax.set_xlabel('Frame', color='white')
    ax.set_ylabel('Detections', color='white')
    ax.legend(facecolor='#1a1a1a',
              labelcolor='white', fontsize=9)

    # mean and std bars
    ax = axes[1]
    style(ax)
    datasets = ['KITTI\nGermany', 'nuScenes\nSingapore']
    means    = [r['kitti_stats']['mean'],
                r['nu_stats']['mean']]
    stds     = [r['kitti_stats']['std'],
                r['nu_stats']['std']]
    colors   = ['#00C8FF', '#FF6B35']
    bars     = ax.bar(datasets, means,
                      color=colors, edgecolor='#333',
                      width=0.5,
                      yerr=stds, capsize=6,
                      error_kw={'color': 'white',
                                'linewidth': 1.5})
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + s + 0.5,
                f"{m:.1f} +/- {s:.1f}",
                ha='center', color='white',
                fontsize=10, fontweight='bold')
    ax.set_title('Mean Detections +/- Std Dev',
                 color='white')
    ax.set_ylabel('Detections', color='white')

    # consistency comparison
    ax = axes[2]
    style(ax)
    metrics  = ['Detection\nDrop', 'KITTI\nConsistency',
                 'nuScenes\nConsistency']
    values   = [r['drop'], r['k_consist'],
                r['n_consist']]
    m_colors = ['#FF4444', '#00FF88', '#FFD700']
    bars     = ax.bar(metrics, values,
                      color=m_colors, edgecolor='#333',
                      width=0.5)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.5,
                f"{v:.1f}%",
                ha='center', color='white',
                fontsize=11, fontweight='bold')
    ax.set_title('Key Metrics (%)', color='white')
    ax.set_ylabel('Percentage', color='white')
    ax.set_ylim(0, 110)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR,
                        'evaluation_chart.png')
    plt.savefig(path, dpi=130, bbox_inches='tight',
                facecolor='#1a1a1a')
    plt.close()
    print(f"\nsaved {path}")
    return path


if __name__ == "__main__":
    results = run_evaluation(n_frames=30)
    plot_evaluation(results)