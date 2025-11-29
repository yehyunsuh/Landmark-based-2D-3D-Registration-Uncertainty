import os
import torch
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans


def save_predictions_csv(args, image_names, csv_dir, mc_preds, gt_coords, prefix="mc_predictions"):
    # print(count*args.n_simulations, "invisible landmarks in total")

    mean_mc = torch.nanmean(mc_preds, dim=0)      # [N, C, 2]e
    diff = mean_mc - gt_coords                    # [N, C, 2]
    dist = torch.norm(diff, dim=2)                # [N, C]

    # Mask: valid GT landmarks = not NaN
    mask = ~torch.isnan(gt_coords).any(dim=2)     # [N, C]
    dist[~mask] = float('nan')

    print(f"[MC Dropout] Mean Dist = {torch.nanmean(dist).item():.4f}")

    # -------------------------------------------------
    # Save MC predictions & GT coordinates as CSV
    # -------------------------------------------------
    S, N, C, _ = mc_preds.shape   # S = n_sim, N = n_images

    # MC predictions CSV: image_id is image, simulation is simulation
    mc_list = []
    for i in range(N):      # image index
        for s in range(S):          # simulation index
            for c in range(C):  # landmark index
                x, y = mc_preds[s, i, c].tolist()
                image_id = image_names[i]
                mc_list.append([image_id, s, c, x, y])

    mc_df = pd.DataFrame(mc_list, columns=["image_id", "simulation", "landmark", "x", "y"])
    mc_path = os.path.join(csv_dir, f"{prefix}_{args.dropout_rate}.csv")  # <--- changed

    mc_df.to_csv(mc_path, index=False)

    if prefix == "mc_predictions":
        return
    
    # Ground truth CSV: use gt_coords.shape[0] (N_gt), not S
    N_gt = gt_coords.shape[0]   # should be equal to N
    gt_list = []
    for i in range(N_gt):
        for c in range(C):
            x, y = gt_coords[i, c].tolist()
            image_id = image_names[i]
            gt_list.append([image_id, c, x, y])

    gt_df = pd.DataFrame(gt_list, columns=["image_id", "landmark", "x", "y"])
    gt_path = os.path.join(csv_dir, "ground_truth.csv")
    gt_df.to_csv(gt_path, index=False)

    print(f"[Saved] MC predictions → {mc_path}")
    print(f"[Saved] Ground truth   → {gt_path}")
    # -------------------------------------------------


def reconstruct_mc_preds(mc_df):
    # Preserve first-appearance order instead of sorting
    img_ids = list(dict.fromkeys(mc_df["image_id"]))
    sim_ids = sorted(mc_df["simulation"].unique())
    lm_ids  = sorted(mc_df["landmark"].unique())

    N = len(img_ids)
    S = len(sim_ids)
    C = len(lm_ids)

    img_map = {img: i for i, img in enumerate(img_ids)}
    sim_map = {sim: j for j, sim in enumerate(sim_ids)}

    mc_preds = torch.full((S, N, C, 2), float('nan'), dtype=torch.float32)

    for _, row in mc_df.iterrows():
        i = img_map[row.image_id]
        s = sim_map[int(row.simulation)]
        c = int(row.landmark)
        mc_preds[s, i, c] = torch.tensor([row.x, row.y], dtype=torch.float32)

    return mc_preds


def reconstruct_gt(gt_df):
    # Again, keep original order
    img_ids = list(dict.fromkeys(gt_df["image_id"]))
    lm_ids  = sorted(gt_df["landmark"].unique())

    N = len(img_ids)
    C = len(lm_ids)

    img_map = {img: i for i, img in enumerate(img_ids)}

    gt_coords = torch.full((N, C, 2), float('nan'), dtype=torch.float32)

    for _, row in gt_df.iterrows():
        i = img_map[row.image_id]
        c = int(row.landmark)
        gt_coords[i, c] = torch.tensor([row.x, row.y], dtype=torch.float32)

    return gt_coords


def compute_mc_distance_from_csv(csv_dir: str, dropout_rate: float):
    mc_df = pd.read_csv(os.path.join(csv_dir, f"mc_predictions_{dropout_rate}.csv"))
    gt_df = pd.read_csv(os.path.join(csv_dir, "ground_truth.csv"))

    mc_preds  = reconstruct_mc_preds(mc_df)     # [S, N, C, 2]
    gt_coords = reconstruct_gt(gt_df)           # [N, C, 2]

    mean_mc = torch.nanmean(mc_preds, dim=0)    # [N, C, 2]
    diff    = mean_mc - gt_coords               # [N, C, 2]
    dist    = torch.norm(diff, dim=2)           # [N, C]

    mask = ~torch.isnan(gt_coords).any(dim=2)
    dist[~mask] = float('nan')

    mean_dist = torch.nanmean(dist).item()

    return mean_dist, mc_preds, gt_coords, dist


def classify_gt_labels(gt_df, image_size=512):

    def classify(x, y):
        if np.isnan(x) or np.isnan(y):
            return "invisible"
        # if x <= 0 or x >= image_size - 1 or y <= 0 or y >= image_size - 1:
        #     return "edge"
        return "normal"

    gt_df["gt_label"] = gt_df.apply(lambda r: classify(r["x"], r["y"]), axis=1)
    return gt_df


def per_image_two_cluster_distance(mc_df, gt_df, n_clusters=2, merge_dist=None, gt_match_thresh=30):
    results = []

    # loop over landmarks
    for lm in sorted(mc_df.landmark.unique()):

        mc_lm = mc_df[mc_df.landmark == lm]

        # loop over images
        for img_id in mc_lm.image_id.unique():

            df_img = mc_lm[mc_lm.image_id == img_id].copy()

            pts = df_img[["x", "y"]].values  # shape: (S, 2), S = n_sim

            # Skip if too few points to run KMeans K=2
            if len(pts) < n_clusters:
                continue

            # ---------------------------------
            # (1) K-means with K=2
            # ---------------------------------
            km = KMeans(n_clusters=2, n_init="auto", random_state=0)
            labels = km.fit_predict(pts)
            centers = km.cluster_centers_  # shape (2, 2)

            # Distance between 2 centers (old metric)
            cluster_dist = np.linalg.norm(centers[0] - centers[1])

            # ---------------------------------
            # (2) Deviation metric (NEW)
            # ---------------------------------
            # Use RMS radial deviation around the mean of MC samples
            mean_pt = pts.mean(axis=0)                 # [2]
            diffs = pts - mean_pt                      # [S, 2]
            sq_dists = np.sum(diffs**2, axis=1)        # [S]
            deviation = float(np.sqrt(np.mean(sq_dists)))  # scalar

            results.append({
                "image_id": img_id,
                "landmark": lm,
                "cluster_distance": cluster_dist,
                "deviation": deviation
            })

    return pd.DataFrame(results)


def run_full_csv_pipeline(csv_dir, dropout_rate):

    # compute MC dropout metrics (unchanged)
    mean_dist, mc_preds, gt_coords, dist_map = \
        compute_mc_distance_from_csv(csv_dir, dropout_rate)

    print("[Recomputed MC Dropout] Mean Dist =", mean_dist)

    # load original CSVs
    mc_df = pd.read_csv(os.path.join(csv_dir, f"mc_predictions_{dropout_rate}.csv"))
    gt_df = pd.read_csv(os.path.join(csv_dir, "ground_truth.csv"))

    # classify GT (unchanged)
    gt_df = classify_gt_labels(gt_df)

    # ============================================
    # compute per-image, per-landmark 2-cluster distances + deviation
    # ============================================
    df_unc = per_image_two_cluster_distance(
        mc_df,
        gt_df,
        n_clusters=2
    )

    return df_unc


def load_or_compute_per_image_uncertainty(csv_dir, dropout_rate):
    """
    Load final per-image cluster distances + deviation if available,
    otherwise compute from MC CSVs.

    Returns:
        df_unc: DataFrame with ['image_id','landmark','cluster_distance','deviation']
        cluster_pivot: DataFrame pivoted to [image_id x landmark] -> deviation
    """
    cluster_csv_path = os.path.join(csv_dir, "final_clustered_results.csv")
    recompute = True

    if os.path.exists(cluster_csv_path):
        df_unc = pd.read_csv(cluster_csv_path)
        # If deviation column already there, we can reuse; otherwise recompute
        if "deviation" in df_unc.columns:
            recompute = False

    if recompute:
        df_unc = run_full_csv_pipeline(csv_dir, dropout_rate=dropout_rate)
        df_unc.to_csv(cluster_csv_path, index=False)
        print(f"[Per-image uncertainty] Saved {cluster_csv_path}")
    else:
        print(f"[Per-image uncertainty] Loaded {cluster_csv_path}")

    # Pivot to fast lookup: rows = image_id, cols = landmark, values = deviation
    cluster_pivot = df_unc.pivot(
        index="image_id",
        columns="landmark",
        values="deviation"      # <---- KEY LINE: use deviation, not cluster_distance
    )

    return df_unc, cluster_pivot
