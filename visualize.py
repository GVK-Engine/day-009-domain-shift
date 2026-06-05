# visualize.py
# 4-panel comparison: camera + BEV for both datasets
# shows real road scenes with detection boxes
# KITTI Germany vs nuScenes Singapore
#
# panel layout:
#   top-left:     KITTI camera with detection dots
#   top-right:    nuScenes camera with detection dots
#   bottom-left:  KITTI BEV point cloud with boxes
#   bottom-right: nuScenes BEV point cloud with boxes
#
# Nani — MS Robotics ASU

import numpy as np
import os
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from detector import (detect, load_kitti_lidar,
                      load_nuscenes_lidar)
from projection import (load_calibration, CALIB_DIR)

KITTI_DIR = (
    r"C:\Users\vamsh\Downloads\kitti"
    r"\2011_09_26_drive_0001_sync"
    r"\2011_09_26"
    r"\2011_09_26_drive_0001_sync"
    r"\velodyne_points\data"
)
KITTI_IMG_DIR = (
    r"C:\Users\vamsh\Downloads\kitti"
    r"\2011_09_26_drive_0001_sync"
    r"\2011_09_26"
    r"\2011_09_26_drive_0001_sync"
    r"\image_02\data"
)
NUSCENES_BASE  = r"D:\day-004-bev-perception"

# nuScenes has CAM_FRONT_LEFT not CAM_FRONT
NUSCENES_CAM   = os.path.join(
    NUSCENES_BASE, "samples", "CAM_FRONT_LEFT"
)
NUSCENES_LIDAR = os.path.join(
    NUSCENES_BASE, "samples", "LIDAR_TOP"
)
RESULTS_DIR = "results"
MAX_RANGE   = 50.0


def load_nuscenes_pairs():
    # nuScenes files named by timestamp
    # sorting gives approximately synchronized pairs
    cam_files = sorted([
        f for f in os.listdir(NUSCENES_CAM)
        if f.endswith('.jpg') or f.endswith('.png')
    ])
    lidar_files = sorted([
        f for f in os.listdir(NUSCENES_LIDAR)
        if f.endswith('.pcd.bin')
    ])
    n = min(len(cam_files), len(lidar_files))
    return (
        [os.path.join(NUSCENES_CAM,   f)
         for f in cam_files[:n]],
        [os.path.join(NUSCENES_LIDAR, f)
         for f in lidar_files[:n]]
    )


def draw_detections_on_image(img, clusters,
                              P2=None,
                              T_lidar_to_cam=None,
                              color=(0, 255, 100)):
    # project each cluster centroid onto camera image
    # draw a colored dot and distance label
    result = img.copy()
    H, W   = img.shape[:2]

    for cluster in clusters:
        if len(cluster) < 5:
            continue

        cx   = float(cluster[:, 0].mean())
        cy   = float(cluster[:, 1].mean())
        cz   = float(cluster[:, 2].mean())
        dist = np.sqrt(cx**2 + cy**2)

        if dist > 40 or cx < 0.5:
            continue

        if P2 is not None and T_lidar_to_cam is not None:
            # proper calibrated projection for KITTI
            pt   = np.array([cx, cy, cz, 1.0],
                             dtype=np.float32)
            pt_c = T_lidar_to_cam @ pt
            if pt_c[2] < 0.5:
                continue
            pt_i = P2 @ pt_c[:4]
            u    = int(pt_i[0] / pt_i[2])
            v    = int(pt_i[1] / pt_i[2])
        else:
            # approximate projection for nuScenes
            # camera is roughly forward-facing
            u = int(W/2 + cy * (-W/2.5) / max(cx, 1))
            v = int(H * 0.55 + cz * (-H/3) / max(cx, 1))

        if not (10 < u < W-10 and 10 < v < H-10):
            continue

        cv2.circle(result, (u, v), 7, color, -1)
        cv2.putText(result, f"{dist:.0f}m",
                    (u - 14, v - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, color, 1, cv2.LINE_AA)

    return result


def plot_bev(ax, points, clusters, title,
             pt_color, box_color):
    ax.set_facecolor('#0d0d0d')

    mask = np.sqrt(
        (points[:, :2]**2).sum(axis=1)
    ) < MAX_RANGE
    pts  = points[mask]
    ax.scatter(pts[:, 0], pts[:, 1],
               s=0.15, c=pt_color, alpha=0.3)

    for cluster in clusters:
        if len(cluster) < 5:
            continue
        xmin = cluster[:, 0].min()
        xmax = cluster[:, 0].max()
        ymin = cluster[:, 1].min()
        ymax = cluster[:, 1].max()
        cx   = (xmin + xmax) / 2
        cy   = (ymin + ymax) / 2
        dist = np.sqrt(cx**2 + cy**2)

        if dist > MAX_RANGE:
            continue

        ax.add_patch(plt.Rectangle(
            (xmin, ymin), xmax-xmin, ymax-ymin,
            fill=False, edgecolor=box_color,
            linewidth=0.9, alpha=0.9
        ))
        ax.text(cx, ymax + 0.3, f"{dist:.0f}m",
                ha='center', va='bottom',
                color=box_color, fontsize=5,
                fontweight='bold')

    for r in [10, 20, 30, 40, 50]:
        ax.add_patch(plt.Circle(
            (0, 0), r, fill=False,
            color='#2a2a2a', linewidth=0.5
        ))

    ax.set_xlim(-MAX_RANGE, MAX_RANGE)
    ax.set_ylim(-MAX_RANGE, MAX_RANGE)
    ax.set_aspect('equal')
    ax.set_title(title, color='white',
                 fontsize=8, pad=3)
    ax.tick_params(colors='#444', labelsize=6)
    for sp in ax.spines.values():
        sp.set_edgecolor('#333')


def make_frame(kitti_lidar, kitti_img,
               nu_lidar, nu_img,
               P2, T_lidar_to_cam,
               idx, total):

    k_pts = load_kitti_lidar(kitti_lidar)
    n_pts = load_nuscenes_lidar(nu_lidar)
    k_img = cv2.imread(kitti_img)
    n_img = cv2.imread(nu_img)

    if k_img is None or n_img is None:
        return None

    k_clusters, _ = detect(kitti_lidar, 'kitti')
    n_clusters, _ = detect(nu_lidar, 'nuscenes')

    # draw detections on camera images
    k_drawn = draw_detections_on_image(
        k_img, k_clusters,
        P2=P2, T_lidar_to_cam=T_lidar_to_cam,
        color=(0, 255, 100)
    )
    n_drawn = draw_detections_on_image(
        n_img, n_clusters,
        color=(0, 200, 255)
    )

    # BGR to RGB for matplotlib
    k_rgb = cv2.cvtColor(k_drawn, cv2.COLOR_BGR2RGB)
    n_rgb = cv2.cvtColor(n_drawn, cv2.COLOR_BGR2RGB)

    fig   = plt.figure(figsize=(16, 8))
    fig.patch.set_facecolor('#0d0d0d')
    ax1   = fig.add_subplot(2, 2, 1)
    ax2   = fig.add_subplot(2, 2, 2)
    ax3   = fig.add_subplot(2, 2, 3)
    ax4   = fig.add_subplot(2, 2, 4)

    ax1.imshow(k_rgb)
    ax1.set_title(
        f"KITTI Germany — {len(k_clusters)} detections",
        color='white', fontsize=9
    )
    ax1.axis('off')

    ax2.imshow(n_rgb)
    ax2.set_title(
        f"nuScenes Singapore — {len(n_clusters)} detections",
        color='white', fontsize=9
    )
    ax2.axis('off')

    plot_bev(
        ax3, k_pts, k_clusters,
        f"KITTI BEV  HDL-64E 64 beams  {len(k_pts):,} pts",
        '#00C8FF', '#00FF88'
    )
    plot_bev(
        ax4, n_pts, n_clusters,
        f"nuScenes BEV  HDL-32E 32 beams  {len(n_pts):,} pts",
        '#FF6B35', '#FFD700'
    )

    fig.suptitle(
        f"Domain Shift: KITTI Germany vs nuScenes Singapore"
        f"   frame {idx+1}/{total}"
        f"   Vamshikrishna Gadde  |  MS Robotics ASU",
        color='#888', fontsize=8
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    tmp = os.path.join(RESULTS_DIR, f"_tmp_{idx}.png")
    plt.savefig(tmp, dpi=100, bbox_inches='tight',
                facecolor='#0d0d0d')
    plt.close()
    return tmp


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    P2, T_lidar_to_cam = load_calibration(CALIB_DIR)

    kitti_lidar = sorted([
        os.path.join(KITTI_DIR, f)
        for f in os.listdir(KITTI_DIR)
        if f.endswith('.bin')
    ])
    kitti_imgs = sorted([
        os.path.join(KITTI_IMG_DIR, f)
        for f in os.listdir(KITTI_IMG_DIR)
        if f.endswith('.png')
    ])
    nu_cams, nu_lidars = load_nuscenes_pairs()

    n = min(15, len(kitti_lidar),
            len(kitti_imgs), len(nu_cams))

    print(f"building {n} frames")

    # save one high quality frame
    tmp = make_frame(
        kitti_lidar[5], kitti_imgs[5],
        nu_lidars[5],   nu_cams[5],
        P2, T_lidar_to_cam, 5, n
    )
    if tmp:
        cv2.imwrite(
            os.path.join(RESULTS_DIR,
                         'comparison_frame.png'),
            cv2.imread(tmp)
        )
        print("saved results/comparison_frame.png")

    # build animated GIF
    gif_frames = []
    tmp_files  = []

    for i in range(n):
        t = make_frame(
            kitti_lidar[i], kitti_imgs[i],
            nu_lidars[i],   nu_cams[i],
            P2, T_lidar_to_cam, i, n
        )
        if t:
            gif_frames.append(imageio.imread(t))
            tmp_files.append(t)
        print(f"  frame {i+1}/{n}")

    gif_path = os.path.join(RESULTS_DIR,
                             'domain_shift_demo.gif')
    imageio.mimsave(gif_path, gif_frames,
                    duration=0.8, loop=0)

    for f in tmp_files:
        if os.path.exists(f):
            os.remove(f)

    print(f"saved {gif_path}")