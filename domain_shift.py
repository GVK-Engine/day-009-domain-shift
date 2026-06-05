# domain_shift.py
# KITTI vs nuScenes detection comparison
# measures how much a KITTI-tuned detector degrades
# when run on Singapore data without retuning
#
# finding: drop is sensor-driven not scene-driven
# HDL-64E (64 beams) vs HDL-32E (32 beams)
#
# Nani — MS Robotics ASU

import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from detector import detect, load_kitti_lidar, load_nuscenes_lidar

KITTI_DIR = (
    r"C:\Users\vamsh\Downloads\kitti"
    r"\2011_09_26_drive_0001_sync"
    r"\2011_09_26"
    r"\2011_09_26_drive_0001_sync"
    r"\velodyne_points\data"
)
NUSCENES_BASE = r"D:\day-004-bev-perception"
RESULTS_DIR   = "results"


def scan_properties(lidar_path, dataset='kitti'):
    # basic stats about one LiDAR scan
    pts    = load_kitti_lidar(lidar_path) \
             if dataset == 'kitti' \
             else load_nuscenes_lidar(lidar_path)
    ranges = np.sqrt((pts[:, :3]**2).sum(axis=1))
    return {
        'n_points':   len(pts),
        'mean_range': float(ranges.mean()),
        'max_range':  float(ranges.max()),
    }


def run_analysis(n_frames=30):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    kitti_files = sorted([
        os.path.join(KITTI_DIR, f)
        for f in os.listdir(KITTI_DIR)
        if f.endswith('.bin')
    ])[:n_frames]

    nu_dir   = os.path.join(NUSCENES_BASE, "samples", "LIDAR_TOP")
    nu_files = sorted([
        os.path.join(nu_dir, f)
        for f in os.listdir(nu_dir)
        if f.endswith('.pcd.bin')
    ])[:n_frames]

    print(f"kitti   : {len(kitti_files)} frames")
    print(f"nuscenes: {len(nu_files)} frames")

    kitti_dets, kitti_props = [], []
    for f in kitti_files:
        clusters, _ = detect(f, dataset='kitti')
        kitti_dets.append(len(clusters))
        kitti_props.append(scan_properties(f, 'kitti'))

    nu_dets, nu_props = [], []
    for f in nu_files:
        clusters, _ = detect(f, dataset='nuscenes')
        nu_dets.append(len(clusters))
        nu_props.append(scan_properties(f, 'nuscenes'))

    km   = np.mean(kitti_dets)
    nm   = np.mean(nu_dets)
    drop = (km - nm) / km * 100
    kpts = np.mean([p['n_points'] for p in kitti_props])
    npts = np.mean([p['n_points'] for p in nu_props])
    pt_drop = (kpts - npts) / kpts * 100

    print(f"\nkitti    {km:.1f} det/frame  {kpts:.0f} pts/scan  "
          f"{np.mean([p['mean_range'] for p in kitti_props]):.1f}m avg range")
    print(f"nuscenes {nm:.1f} det/frame  {npts:.0f} pts/scan  "
          f"{np.mean([p['mean_range'] for p in nu_props]):.1f}m avg range")
    print(f"\ndetection drop : {drop:.1f}%")
    print(f"point drop     : {pt_drop:.1f}%")
    print(f"sensor kitti   : Velodyne HDL-64E (64 beams)")
    print(f"sensor nuscenes: Velodyne HDL-32E (32 beams)")

    return {
        'kitti_dets':  kitti_dets,
        'nu_dets':     nu_dets,
        'kitti_mean':  km,
        'nu_mean':     nm,
        'drop_pct':    drop,
        'kitti_pts':   kpts,
        'nu_pts':      npts,
        'pt_drop':     pt_drop,
        'kitti_props': kitti_props,
        'nu_props':    nu_props,
    }


def plot_results(r):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.patch.set_facecolor('#1a1a1a')
    fig.suptitle(
        'Domain Shift: KITTI Germany vs nuScenes Singapore\n'
        'Vamshikrishna Gadde  |  MS Robotics ASU',
        color='white', fontsize=13
    )

    def style(ax):
        ax.set_facecolor('#1a1a1a')
        ax.tick_params(colors='white')
        for sp in ax.spines.values():
            sp.set_edgecolor('#444')

    # detection count distributions
    ax = axes[0, 0]
    style(ax)
    ax.hist(r['kitti_dets'], bins=10, alpha=0.7,
            color='#00C8FF',
            label=f"KITTI Germany (avg {r['kitti_mean']:.1f})",
            edgecolor='#222')
    ax.hist(r['nu_dets'], bins=10, alpha=0.7,
            color='#FF6B35',
            label=f"nuScenes Singapore (avg {r['nu_mean']:.1f})",
            edgecolor='#222')
    ax.set_title('Detections per Frame', color='white')
    ax.set_xlabel('Objects Detected', color='white')
    ax.set_ylabel('Frames', color='white')
    ax.legend(facecolor='#1a1a1a', labelcolor='white',
              fontsize=9)

    # point cloud size comparison
    ax = axes[0, 1]
    style(ax)
    labels = ['KITTI\nGermany\nHDL-64E',
              'nuScenes\nSingapore\nHDL-32E']
    vals   = [r['kitti_pts'], r['nu_pts']]
    bars   = ax.bar(labels, vals,
                    color=['#00C8FF', '#FF6B35'],
                    edgecolor='#222', width=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 800,
                f"{v:,.0f}", ha='center',
                color='white', fontsize=11,
                fontweight='bold')
    ax.set_title('Avg LiDAR Points per Scan', color='white')
    ax.set_ylabel('Points', color='white')

    # range distributions
    ax = axes[1, 0]
    style(ax)
    ax.hist([p['mean_range'] for p in r['kitti_props']],
            bins=10, alpha=0.7, color='#00C8FF',
            label='KITTI Germany', edgecolor='#222')
    ax.hist([p['mean_range'] for p in r['nu_props']],
            bins=10, alpha=0.7, color='#FF6B35',
            label='nuScenes Singapore', edgecolor='#222')
    ax.set_title('Mean Scan Range Distribution', color='white')
    ax.set_xlabel('Mean Range (m)', color='white')
    ax.set_ylabel('Frames', color='white')
    ax.legend(facecolor='#1a1a1a', labelcolor='white',
              fontsize=9)

    # summary panel
    ax = axes[1, 1]
    style(ax)
    ax.axis('off')
    ax.set_title('Summary', color='white')

    rows = [
        ('KITTI detections',    f"{r['kitti_mean']:.1f}/frame",  '#00C8FF'),
        ('nuScenes detections', f"{r['nu_mean']:.1f}/frame",     '#FF6B35'),
        ('Detection drop',      f"{r['drop_pct']:.1f}%",         '#FF4444'),
        ('KITTI points',        f"{r['kitti_pts']:,.0f}/scan",   '#00C8FF'),
        ('nuScenes points',     f"{r['nu_pts']:,.0f}/scan",      '#FF6B35'),
        ('Point drop',          f"{r['pt_drop']:.1f}%",          '#FF4444'),
        ('KITTI sensor',        'HDL-64E  64 beams',             '#AAAAAA'),
        ('nuScenes sensor',     'HDL-32E  32 beams',             '#AAAAAA'),
        ('Root cause',          'Sensor not scene',              '#FFD700'),
    ]

    for i, (label, val, color) in enumerate(rows):
        ax.text(0.04, 0.94 - i * 0.10, label,
                transform=ax.transAxes,
                color='#888888', fontsize=10)
        ax.text(0.52, 0.94 - i * 0.10, val,
                transform=ax.transAxes,
                color=color, fontsize=10,
                fontweight='bold')

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'domain_shift_analysis.png')
    plt.savefig(path, dpi=130, bbox_inches='tight',
                facecolor='#1a1a1a')
    plt.close()
    print(f"\nsaved {path}")
    return path


if __name__ == "__main__":
    results = run_analysis(n_frames=30)
    plot_results(results)