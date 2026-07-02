import numpy as np
from scipy.spatial import cKDTree


def _camera_positions(center, radius, num_views):
    # Fibonacci sphere for near-uniform camera placement.
    i = np.arange(num_views, dtype=np.float64)
    phi = (1 + np.sqrt(5)) / 2
    theta = 2.0 * np.pi * i / phi
    z = 1.0 - (2.0 * i + 1.0) / num_views
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    dirs = np.stack([x, y, z], axis=1)
    return center[None, :] + radius * dirs


def _visible_from_view(points, cam, azimuth_bins, elevation_bins):
    vec = points - cam[None, :]
    dist = np.linalg.norm(vec, axis=1)
    valid = dist > 1e-12
    if not np.any(valid):
        return np.zeros((0,), dtype=np.int64)

    idx = np.nonzero(valid)[0]
    vec = vec[valid]
    dist = dist[valid]

    az = np.arctan2(vec[:, 1], vec[:, 0])  # [-pi, pi]
    el = np.arcsin(np.clip(vec[:, 2] / dist, -1.0, 1.0))  # [-pi/2, pi/2]

    az_bin = np.floor((az + np.pi) / (2.0 * np.pi) * azimuth_bins).astype(np.int64)
    el_bin = np.floor((el + np.pi / 2.0) / np.pi * elevation_bins).astype(np.int64)
    az_bin = np.clip(az_bin, 0, azimuth_bins - 1)
    el_bin = np.clip(el_bin, 0, elevation_bins - 1)
    key = el_bin * azimuth_bins + az_bin

    # Sort by (key, dist) and keep first per key => closest visible point in each ray bin.
    order = np.lexsort((dist, key))
    key_sorted = key[order]
    first = np.ones_like(key_sorted, dtype=bool)
    first[1:] = key_sorted[1:] != key_sorted[:-1]
    picked_local = order[first]
    return idx[picked_local]


def filter_interior_points(
    points,
    num_views=24,
    azimuth_bins=192,
    elevation_bins=96,
    view_radius_scale=2.5,
    max_points_for_filter=1200000,
):
    # Optional pre-downsample for speed/memory on very large clouds.
    if len(points) > max_points_for_filter:
        sample_idx = np.random.choice(len(points), max_points_for_filter, replace=False)
        work_points = points[sample_idx]
    else:
        sample_idx = None
        work_points = points

    center = work_points.mean(axis=0)
    extent = work_points.max(axis=0) - work_points.min(axis=0)
    radius = np.linalg.norm(extent) * view_radius_scale
    cams = _camera_positions(center, radius, num_views)

    visible = np.zeros((len(work_points),), dtype=bool)
    for cam in cams:
        idx = _visible_from_view(work_points, cam, azimuth_bins, elevation_bins)
        visible[idx] = True

    filtered = work_points[visible]

    # Thicken the exterior shell by keeping points close to visible points.
    target_keep = min(len(work_points), max(20000, filtered.shape[0] * 8))
    if filtered.shape[0] < target_keep and filtered.shape[0] > 0:
        tree = cKDTree(filtered)
        d, _ = tree.query(work_points, k=1)
        keep_idx = np.argpartition(d, target_keep - 1)[:target_keep]
        filtered = work_points[keep_idx]

    # Ensure we never return empty or tiny sets in degenerate cases.
    if len(filtered) < 2048:
        filtered = work_points

    if sample_idx is None:
        return filtered
    return filtered
