# detector.py
# LiDAR object detector that works on both
# KITTI and nuScenes point cloud formats
#
# This is the same core algorithm from Day 1.
# Adapted to handle both dataset formats.
# Used as the baseline for domain shift analysis.
#
# Nani | Day 9 of 90 | MS Robotics ASU

import numpy as np
import os


# detection parameters
# tuned on KITTI — this is the key point
# these were never tuned for Singapore data
VOXEL_SIZE     = 0.2
RANSAC_DIST    = 0.3
RANSAC_ITER    = 100
DBSCAN_EPS     = 0.6
DBSCAN_MIN_PTS = 8
MIN_CLUSTER    = 15
MAX_CLUSTER    = 3000

# height filter — ground to rooftop
MIN_HEIGHT     = -1.5
MAX_HEIGHT     =  3.5


def load_kitti_lidar(filepath):
    # KITTI stores points as float32 (x, y, z, intensity)
    pts = np.fromfile(filepath, dtype=np.float32)
    return pts.reshape(-1, 4)


def load_nuscenes_lidar(filepath):
    # nuScenes stores points as float32
    # format: (x, y, z, intensity, ring_index)
    pts = np.fromfile(filepath, dtype=np.float32)
    pts = pts.reshape(-1, 5)
    # return only x, y, z, intensity to match KITTI format
    return pts[:, :4]


def voxel_downsample(points, voxel_size=VOXEL_SIZE):
    # keep one point per voxel cell
    coords = np.floor(points[:, :3] / voxel_size).astype(int)
    _, idx = np.unique(coords, axis=0, return_index=True)
    return points[idx]


def remove_ground(points, dist_thresh=RANSAC_DIST,
                  n_iter=RANSAC_ITER):
    # RANSAC ground plane removal
    # same algorithm as Day 1
    best_mask  = np.zeros(len(points), dtype=bool)
    best_count = 0

    for _ in range(n_iter):
        idx     = np.random.choice(len(points), 3,
                                    replace=False)
        p1, p2, p3 = points[idx, :3]
        n       = np.cross(p2 - p1, p3 - p1)
        if np.linalg.norm(n) < 1e-6:
            continue
        n     = n / np.linalg.norm(n)
        dists = np.abs(points[:, :3] @ n - np.dot(n, p1))
        mask  = dists < dist_thresh
        if mask.sum() > best_count:
            best_count = mask.sum()
            best_mask  = mask

    return points[~best_mask]


def cluster_objects(points, eps=DBSCAN_EPS,
                    min_pts=DBSCAN_MIN_PTS):
    # DBSCAN clustering using scipy for speed
    from scipy.spatial import cKDTree

    if len(points) < min_pts:
        return []

    tree   = cKDTree(points[:, :3])
    labels = -np.ones(len(points), dtype=int)
    cid    = 0

    for i in range(len(points)):
        if labels[i] != -1:
            continue
        nbrs = tree.query_ball_point(
            points[i, :3], eps
        )
        if len(nbrs) < min_pts:
            continue
        labels[i] = cid
        queue = list(nbrs)
        while queue:
            j = queue.pop(0)
            if labels[j] == -1:
                labels[j] = cid
                nbrs_j = tree.query_ball_point(
                    points[j, :3], eps
                )
                if len(nbrs_j) >= min_pts:
                    queue.extend(nbrs_j)
        cid += 1

    return [
        points[labels == c]
        for c in range(cid)
        if MIN_CLUSTER < (labels == c).sum() < MAX_CLUSTER
    ]


def detect(lidar_path, dataset='kitti'):
    """
    Full detection pipeline.
    Works on both KITTI and nuScenes LiDAR files.

    dataset: 'kitti' or 'nuscenes'
    Returns list of detected cluster arrays.
    """
    # load based on dataset format
    if dataset == 'kitti':
        pts = load_kitti_lidar(lidar_path)
    else:
        pts = load_nuscenes_lidar(lidar_path)

    # keep only forward-facing points
    pts = pts[pts[:, 0] > 0]

    if len(pts) < 100:
        return [], {}

    # pipeline: downsample → remove ground → cluster
    ds    = voxel_downsample(pts)
    above = remove_ground(ds)

    # height filter
    above = above[
        (above[:, 2] > MIN_HEIGHT) &
        (above[:, 2] < MAX_HEIGHT)
    ]

    clusters = cluster_objects(above)

    stats = {
        'total_points':   len(pts),
        'after_ground':   len(above),
        'n_detections':   len(clusters),
        'dataset':        dataset
    }

    return clusters, stats


if __name__ == "__main__":

    print("\n" + "="*50)
    print("  Detector Test — Day 9")
    print("="*50)

    # test on KITTI
    KITTI_DIR = (
        r"C:\Users\vamsh\Downloads\kitti"
        r"\2011_09_26_drive_0001_sync"
        r"\2011_09_26"
        r"\2011_09_26_drive_0001_sync"
        r"\velodyne_points\data"
    )

    kitti_files = sorted(os.listdir(KITTI_DIR))[:5]
    print(f"\n  KITTI test ({len(kitti_files)} frames):")

    kitti_counts = []
    for f in kitti_files:
        path     = os.path.join(KITTI_DIR, f)
        clusters, stats = detect(path, dataset='kitti')
        kitti_counts.append(len(clusters))
        print(f"    {f[:14]}: {len(clusters)} objects detected")

    print(f"  Avg KITTI detections: "
          f"{np.mean(kitti_counts):.1f}")

    # test on nuScenes
    import json
    NUSCENES_BASE = r"D:\day-004-bev-perception"
    lidar_dir     = os.path.join(
        NUSCENES_BASE, "samples", "LIDAR_TOP"
    )

    if os.path.exists(lidar_dir):
        nu_files = sorted(os.listdir(lidar_dir))[:5]
        print(f"\n  nuScenes test ({len(nu_files)} frames):")

        nu_counts = []
        for f in nu_files:
            path = os.path.join(lidar_dir, f)
            clusters, stats = detect(path,
                                      dataset='nuscenes')
            nu_counts.append(len(clusters))
            print(f"    {f[:14]}: "
                  f"{len(clusters)} objects detected")

        print(f"  Avg nuScenes detections: "
              f"{np.mean(nu_counts):.1f}")
    else:
        print(f"\n  nuScenes LIDAR not found at {lidar_dir}")

    print("="*50)