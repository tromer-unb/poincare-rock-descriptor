import os
import glob
import math
import warnings
import numpy as np
import pandas as pd

from PIL import Image
from scipy import ndimage as ndi

try:
    from skimage.morphology import skeletonize
except ImportError:
    raise ImportError(
        "Install scikit-image before running:\n"
        "pip install scikit-image"
    )


# ============================================================
# Main settings
# ============================================================

INPUT_FOLDER = "../poros/poros/"  # folder containing the rock images
OUTPUT_CSV = "poincar_descriptors.csv"

IMAGE_EXTENSIONS = ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp"]

RADIAL_BINS = 16
ANGULAR_BINS = 16
COMPONENT_DISTANCE_BINS = 16
BRANCH_LENGTH_BINS = 16

MAX_COMPONENT_PAIRS = 20000

GREEN_MIN = 80
GREEN_DELTA = 30

EPS = 1e-9


# ============================================================
# Helper functions
# ============================================================

def safe_mean(x):
    x = np.asarray(x)
    if x.size == 0:
        return 0.0
    return float(np.mean(x))


def safe_std(x):
    x = np.asarray(x)
    if x.size == 0:
        return 0.0
    return float(np.std(x))


def safe_min(x):
    x = np.asarray(x)
    if x.size == 0:
        return 0.0
    return float(np.min(x))


def safe_max(x):
    x = np.asarray(x)
    if x.size == 0:
        return 0.0
    return float(np.max(x))


def safe_median(x):
    x = np.asarray(x)
    if x.size == 0:
        return 0.0
    return float(np.median(x))


def normalized_entropy(values, bins, value_range=None):
    values = np.asarray(values)

    if values.size == 0:
        return 0.0, np.zeros(bins, dtype=float)

    hist, _ = np.histogram(values, bins=bins, range=value_range)
    total = hist.sum()

    if total == 0:
        return 0.0, np.zeros(bins, dtype=float)

    p = hist.astype(float) / total
    p_nonzero = p[p > 0]

    entropy = -np.sum(p_nonzero * np.log(p_nonzero)) / np.log(bins)

    return float(entropy), p


def add_histogram_features(features, prefix, hist):
    for i, v in enumerate(hist):
        features[f"{prefix}_bin_{i:02d}"] = float(v)


def extract_pore_mask(image_path):
    """
    Extracts the boolean pore mask.

    Returns:
        pore_mask: True = pore/void; False = rock matrix
    """

    img = Image.open(image_path).convert("RGB")
    arr = np.asarray(img)

    R = arr[:, :, 0].astype(np.int16)
    G = arr[:, :, 1].astype(np.int16)
    B = arr[:, :, 2].astype(np.int16)

    pore = (
        (G >= GREEN_MIN) &
        (G >= R + GREEN_DELTA) &
        (G >= B + GREEN_DELTA)
    )

    return pore


def image_coordinates_to_poincare(xs, ys, width, height):
    """
    Maps image coordinates to the Poincaré disk.

    The image center becomes the disk center.
    The corners remain within radius 0.99.
    """

    scale = 0.99 / np.sqrt(2.0)

    x = scale * (2.0 * (xs + 0.5) / width - 1.0)
    y = scale * (2.0 * (ys + 0.5) / height - 1.0)

    r = np.sqrt(x**2 + y**2)
    r = np.clip(r, 0.0, 0.999999)

    theta = np.arctan2(y, x)
    rho = 2.0 * np.arctanh(r)

    return x, y, r, theta, rho


def poincare_distance(u, v):
    """
    Hyperbolic distance in the Poincaré disk.

    u and v must have shape (..., 2)
    """

    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)

    diff2 = np.sum((u - v) ** 2, axis=-1)
    norm_u2 = np.sum(u ** 2, axis=-1)
    norm_v2 = np.sum(v ** 2, axis=-1)

    denom = (1.0 - norm_u2) * (1.0 - norm_v2)
    denom = np.maximum(denom, EPS)

    arg = 1.0 + 2.0 * diff2 / denom
    arg = np.maximum(arg, 1.0)

    return np.arccosh(arg)


# ============================================================
# Level 1: global hyperbolic descriptor
# ============================================================

def global_poincare_descriptor(pore):
    features = {}

    height, width = pore.shape
    total_pixels = height * width

    ys, xs = np.nonzero(pore)
    n_pores = len(xs)

    features["image_height"] = int(height)
    features["image_width"] = int(width)
    features["n_pixels"] = int(total_pixels)
    features["n_pore_pixels"] = int(n_pores)
    features["porosity_2d"] = float(n_pores / total_pixels)

    if n_pores == 0:
        empty_radial = np.zeros(RADIAL_BINS)
        empty_angular = np.zeros(ANGULAR_BINS)

        features.update({
            "poincare_r_mean": 0.0,
            "poincare_r_std": 0.0,
            "poincare_r_min": 0.0,
            "poincare_r_max": 0.0,
            "poincare_r_median": 0.0,
            "poincare_rho_mean": 0.0,
            "poincare_rho_std": 0.0,
            "poincare_rho_min": 0.0,
            "poincare_rho_max": 0.0,
            "poincare_rho_median": 0.0,
            "radial_entropy": 0.0,
            "angular_entropy": 0.0,
            "angular_anisotropy_A1": 0.0,
            "angular_anisotropy_A2": 0.0,
            "angular_anisotropy_A4": 0.0,
        })

        add_histogram_features(features, "global_radial_hist", empty_radial)
        add_histogram_features(features, "global_angular_hist", empty_angular)

        return features

    x, y, r, theta, rho = image_coordinates_to_poincare(xs, ys, width, height)

    rho_max = 2.0 * np.arctanh(0.99)

    radial_entropy, radial_hist = normalized_entropy(
        rho,
        bins=RADIAL_BINS,
        value_range=(0.0, rho_max)
    )

    angular_entropy, angular_hist = normalized_entropy(
        theta,
        bins=ANGULAR_BINS,
        value_range=(-np.pi, np.pi)
    )

    A1 = np.abs(np.mean(np.exp(1j * theta)))
    A2 = np.abs(np.mean(np.exp(2j * theta)))
    A4 = np.abs(np.mean(np.exp(4j * theta)))

    features.update({
        "poincare_r_mean": safe_mean(r),
        "poincare_r_std": safe_std(r),
        "poincare_r_min": safe_min(r),
        "poincare_r_max": safe_max(r),
        "poincare_r_median": safe_median(r),

        "poincare_rho_mean": safe_mean(rho),
        "poincare_rho_std": safe_std(rho),
        "poincare_rho_min": safe_min(rho),
        "poincare_rho_max": safe_max(rho),
        "poincare_rho_median": safe_median(rho),

        "radial_entropy": radial_entropy,
        "angular_entropy": angular_entropy,

        "angular_anisotropy_A1": float(A1),
        "angular_anisotropy_A2": float(A2),
        "angular_anisotropy_A4": float(A4),
    })

    add_histogram_features(features, "global_radial_hist", radial_hist)
    add_histogram_features(features, "global_angular_hist", angular_hist)

    return features


# ============================================================
# Level 2: connected-component descriptor
# ============================================================

def component_poincare_descriptor(pore):
    features = {}

    height, width = pore.shape

    labels, n_components = ndi.label(pore, structure=np.ones((3, 3)))
    areas = np.bincount(labels.ravel())[1:]

    features["component_count"] = int(n_components)

    if n_components == 0:
        features.update({
            "component_area_mean": 0.0,
            "component_area_std": 0.0,
            "component_area_min": 0.0,
            "component_area_max": 0.0,
            "component_area_median": 0.0,
            "largest_component_area": 0.0,
            "largest_component_fraction_of_pores": 0.0,
            "component_area_entropy": 0.0,

            "component_centroid_rho_mean": 0.0,
            "component_centroid_rho_std": 0.0,
            "component_centroid_angular_entropy": 0.0,
            "component_centroid_radial_entropy": 0.0,

            "component_pair_distance_mean": 0.0,
            "component_pair_distance_std": 0.0,
            "component_pair_distance_min": 0.0,
            "component_pair_distance_max": 0.0,
            "component_pair_distance_median": 0.0,
            "component_pair_distance_entropy": 0.0,
        })

        add_histogram_features(
            features,
            "component_pair_distance_hist",
            np.zeros(COMPONENT_DISTANCE_BINS)
        )

        return features

    total_pore_area = areas.sum()
    largest_area = areas.max()

    area_entropy, _ = normalized_entropy(
        areas,
        bins=RADIAL_BINS,
        value_range=(1.0, max(2.0, float(largest_area)))
    )

    component_indices = np.arange(1, n_components + 1)

    centroids_yx = ndi.center_of_mass(
        pore.astype(np.uint8),
        labels,
        component_indices
    )

    centroids_y = np.array([c[0] for c in centroids_yx], dtype=float)
    centroids_x = np.array([c[1] for c in centroids_yx], dtype=float)

    cx, cy, cr, ctheta, crho = image_coordinates_to_poincare(
        centroids_x,
        centroids_y,
        width,
        height
    )

    centroid_radial_entropy, _ = normalized_entropy(
        crho,
        bins=RADIAL_BINS,
        value_range=(0.0, 2.0 * np.arctanh(0.99))
    )

    centroid_angular_entropy, _ = normalized_entropy(
        ctheta,
        bins=ANGULAR_BINS,
        value_range=(-np.pi, np.pi)
    )

    features.update({
        "component_area_mean": safe_mean(areas),
        "component_area_std": safe_std(areas),
        "component_area_min": safe_min(areas),
        "component_area_max": safe_max(areas),
        "component_area_median": safe_median(areas),
        "largest_component_area": float(largest_area),
        "largest_component_fraction_of_pores": float(largest_area / total_pore_area),
        "component_area_entropy": area_entropy,

        "component_centroid_rho_mean": safe_mean(crho),
        "component_centroid_rho_std": safe_std(crho),
        "component_centroid_angular_entropy": centroid_angular_entropy,
        "component_centroid_radial_entropy": centroid_radial_entropy,
    })

    if n_components < 2:
        pair_distances = np.array([])
    else:
        points = np.column_stack([cx, cy])

        all_pairs_count = n_components * (n_components - 1) // 2

        if all_pairs_count <= MAX_COMPONENT_PAIRS:
            i_idx, j_idx = np.triu_indices(n_components, k=1)
        else:
            rng = np.random.default_rng(12345)
            i_idx = rng.integers(0, n_components, size=MAX_COMPONENT_PAIRS)
            j_idx = rng.integers(0, n_components, size=MAX_COMPONENT_PAIRS)

            valid = i_idx != j_idx
            i_idx = i_idx[valid]
            j_idx = j_idx[valid]

        pair_distances = poincare_distance(points[i_idx], points[j_idx])

    pair_entropy, pair_hist = normalized_entropy(
        pair_distances,
        bins=COMPONENT_DISTANCE_BINS,
        value_range=None
    )

    features.update({
        "component_pair_distance_mean": safe_mean(pair_distances),
        "component_pair_distance_std": safe_std(pair_distances),
        "component_pair_distance_min": safe_min(pair_distances),
        "component_pair_distance_max": safe_max(pair_distances),
        "component_pair_distance_median": safe_median(pair_distances),
        "component_pair_distance_entropy": pair_entropy,
    })

    add_histogram_features(
        features,
        "component_pair_distance_hist",
        pair_hist
    )

    return features


# ============================================================
# Level 3: skeleton network descriptor
# ============================================================

def skeleton_neighbor_count(skel):
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0
    return ndi.convolve(skel.astype(np.uint8), kernel, mode="constant", cval=0)


def trace_skeleton_branches(skel):
    """
    Extracts branches from the 8-connected skeleton.

    Returns a list of branches.
    Each branch is a list of coordinates [(y, x), ...].
    """

    skel = skel.astype(bool)
    degree = skeleton_neighbor_count(skel)

    node_mask = skel & (degree != 2)

    neighbors = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]

    visited_edges = set()
    branches = []

    node_positions = np.argwhere(node_mask)

    for y0, x0 in node_positions:
        y0 = int(y0)
        x0 = int(x0)

        for dy, dx in neighbors:
            y1 = y0 + dy
            x1 = x0 + dx

            if (
                y1 < 0 or y1 >= skel.shape[0] or
                x1 < 0 or x1 >= skel.shape[1] or
                not skel[y1, x1]
            ):
                continue

            edge_key = tuple(sorted(((y0, x0), (y1, x1))))

            if edge_key in visited_edges:
                continue

            branch = [(y0, x0)]
            prev = (y0, x0)
            curr = (y1, x1)

            visited_edges.add(edge_key)

            while True:
                branch.append(curr)
                cy, cx = curr

                if node_mask[cy, cx] and curr != (y0, x0):
                    break

                next_candidates = []

                for ndy, ndx in neighbors:
                    ny = cy + ndy
                    nx = cx + ndx

                    if (
                        ny < 0 or ny >= skel.shape[0] or
                        nx < 0 or nx >= skel.shape[1]
                    ):
                        continue

                    if not skel[ny, nx]:
                        continue

                    if (ny, nx) == prev:
                        continue

                    next_candidates.append((ny, nx))

                if len(next_candidates) == 0:
                    break

                next_pixel = next_candidates[0]

                edge_key = tuple(sorted((curr, next_pixel)))

                if edge_key in visited_edges:
                    break

                visited_edges.add(edge_key)

                prev = curr
                curr = next_pixel

            if len(branch) >= 2:
                branches.append(branch)

    return branches


def branch_euclidean_length(branch):
    if len(branch) < 2:
        return 0.0

    length = 0.0

    for i in range(len(branch) - 1):
        y0, x0 = branch[i]
        y1, x1 = branch[i + 1]

        dy = y1 - y0
        dx = x1 - x0

        length += math.sqrt(dx * dx + dy * dy)

    return float(length)


def branch_hyperbolic_length(branch, width, height):
    if len(branch) < 2:
        return 0.0

    coords = np.array(branch, dtype=float)
    ys = coords[:, 0]
    xs = coords[:, 1]

    px, py, pr, ptheta, prho = image_coordinates_to_poincare(
        xs,
        ys,
        width,
        height
    )

    points = np.column_stack([px, py])

    d = poincare_distance(points[:-1], points[1:])

    return float(np.sum(d))


def branch_hyperbolic_chord(branch, width, height):
    if len(branch) < 2:
        return 0.0

    y0, x0 = branch[0]
    y1, x1 = branch[-1]

    px, py, _, _, _ = image_coordinates_to_poincare(
        np.array([x0, x1], dtype=float),
        np.array([y0, y1], dtype=float),
        width,
        height
    )

    points = np.column_stack([px, py])

    return float(poincare_distance(points[0], points[1]))


def network_poincare_descriptor(pore):
    features = {}

    height, width = pore.shape

    if pore.sum() == 0:
        features.update({
            "skeleton_pixel_count": 0,
            "network_endpoint_count": 0,
            "network_junction_count": 0,
            "network_node_count": 0,
            "network_branch_count": 0,

            "branch_euclidean_length_mean": 0.0,
            "branch_euclidean_length_std": 0.0,
            "branch_euclidean_length_min": 0.0,
            "branch_euclidean_length_max": 0.0,
            "branch_euclidean_length_median": 0.0,

            "branch_hyperbolic_length_mean": 0.0,
            "branch_hyperbolic_length_std": 0.0,
            "branch_hyperbolic_length_min": 0.0,
            "branch_hyperbolic_length_max": 0.0,
            "branch_hyperbolic_length_median": 0.0,

            "branch_hyperbolic_tortuosity_mean": 0.0,
            "branch_hyperbolic_tortuosity_std": 0.0,
            "branch_hyperbolic_tortuosity_median": 0.0,

            "network_mean_degree": 0.0,
            "network_max_degree": 0.0,
            "network_branch_density": 0.0,
        })

        add_histogram_features(
            features,
            "branch_hyperbolic_length_hist",
            np.zeros(BRANCH_LENGTH_BINS)
        )

        return features

    skel = skeletonize(pore).astype(bool)
    degree = skeleton_neighbor_count(skel)

    endpoints = skel & (degree == 1)
    junctions = skel & (degree >= 3)
    nodes = skel & (degree != 2)

    skeleton_pixel_count = int(skel.sum())
    endpoint_count = int(endpoints.sum())
    junction_count = int(junctions.sum())
    node_count = int(nodes.sum())

    branches = trace_skeleton_branches(skel)
    branch_count = len(branches)

    euclidean_lengths = []
    hyperbolic_lengths = []
    hyperbolic_chords = []
    hyperbolic_tortuosities = []

    for branch in branches:
        e_len = branch_euclidean_length(branch)
        h_len = branch_hyperbolic_length(branch, width, height)
        h_chord = branch_hyperbolic_chord(branch, width, height)

        if h_chord > EPS:
            tort = h_len / h_chord
        else:
            tort = 0.0

        euclidean_lengths.append(e_len)
        hyperbolic_lengths.append(h_len)
        hyperbolic_chords.append(h_chord)
        hyperbolic_tortuosities.append(tort)

    euclidean_lengths = np.asarray(euclidean_lengths)
    hyperbolic_lengths = np.asarray(hyperbolic_lengths)
    hyperbolic_tortuosities = np.asarray(hyperbolic_tortuosities)

    branch_entropy, branch_hist = normalized_entropy(
        hyperbolic_lengths,
        bins=BRANCH_LENGTH_BINS,
        value_range=None
    )

    node_degrees = degree[nodes]

    features.update({
        "skeleton_pixel_count": skeleton_pixel_count,
        "network_endpoint_count": endpoint_count,
        "network_junction_count": junction_count,
        "network_node_count": node_count,
        "network_branch_count": branch_count,

        "branch_euclidean_length_mean": safe_mean(euclidean_lengths),
        "branch_euclidean_length_std": safe_std(euclidean_lengths),
        "branch_euclidean_length_min": safe_min(euclidean_lengths),
        "branch_euclidean_length_max": safe_max(euclidean_lengths),
        "branch_euclidean_length_median": safe_median(euclidean_lengths),

        "branch_hyperbolic_length_mean": safe_mean(hyperbolic_lengths),
        "branch_hyperbolic_length_std": safe_std(hyperbolic_lengths),
        "branch_hyperbolic_length_min": safe_min(hyperbolic_lengths),
        "branch_hyperbolic_length_max": safe_max(hyperbolic_lengths),
        "branch_hyperbolic_length_median": safe_median(hyperbolic_lengths),
        "branch_hyperbolic_length_entropy": branch_entropy,

        "branch_hyperbolic_tortuosity_mean": safe_mean(hyperbolic_tortuosities),
        "branch_hyperbolic_tortuosity_std": safe_std(hyperbolic_tortuosities),
        "branch_hyperbolic_tortuosity_median": safe_median(hyperbolic_tortuosities),

        "network_mean_degree": safe_mean(node_degrees),
        "network_max_degree": safe_max(node_degrees),
        "network_branch_density": float(branch_count / max(1, skeleton_pixel_count)),
    })

    add_histogram_features(
        features,
        "branch_hyperbolic_length_hist",
        branch_hist
    )

    return features


# ============================================================
# Complete descriptor
# ============================================================

def compute_complete_poincare_descriptor(image_path):
    pore = extract_pore_mask(image_path)

    features = {
        "filename": os.path.basename(image_path),
        "path": image_path,
    }

    level_1 = global_poincare_descriptor(pore)
    level_2 = component_poincare_descriptor(pore)
    level_3 = network_poincare_descriptor(pore)

    for k, v in level_1.items():
        features[f"L1_global_{k}"] = v

    for k, v in level_2.items():
        features[f"L2_components_{k}"] = v

    for k, v in level_3.items():
        features[f"L3_network_{k}"] = v

    return features


# ============================================================
# Batch execution
# ============================================================

def find_images(folder):
    paths = []

    for ext in IMAGE_EXTENSIONS:
        paths.extend(glob.glob(os.path.join(folder, ext)))

    paths = sorted(paths)

    return paths


def main():
    if not os.path.isdir(INPUT_FOLDER):
        raise FileNotFoundError(
            f"The folder '{INPUT_FOLDER}' was not found. "
            f"Place the images inside it or change INPUT_FOLDER in the script."
        )

    image_paths = find_images(INPUT_FOLDER)

    if len(image_paths) == 0:
        raise RuntimeError(
            f"No images were found in '{INPUT_FOLDER}'. "
            f"Accepted extensions: {IMAGE_EXTENSIONS}"
        )

    rows = []

    print(f"Found {len(image_paths)} images in '{INPUT_FOLDER}'.")

    for i, path in enumerate(image_paths, start=1):
        print(f"[{i}/{len(image_paths)}] Processing: {os.path.basename(path)}")

        try:
            desc = compute_complete_poincare_descriptor(path)
            rows.append(desc)

        except Exception as e:
            warnings.warn(f"Error while processing {path}: {e}")

            rows.append({
                "filename": os.path.basename(path),
                "path": path,
                "error": str(e),
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)

    print()
    print(f"Descriptors saved to: {OUTPUT_CSV}")
    print(f"Number of rocks analyzed: {len(df)}")
    print(f"Number of features: {df.shape[1]}")


if __name__ == "__main__":
    main()
