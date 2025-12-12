import torch
import numpy as np
import nibabel as nib

from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

from src.test.perspective_projection import apply_transformation, project_point
from src.test.pose_estimation import pose_estimation, pose_estimation_weighted


def compute_geodesic_distance(pred_euler, gt_euler, seq='yxz'):
    """
    Computes the geodesic distance (angular error) between two sets of Euler angles.
    Args:
        pred_euler: (3,) array of predicted angles in degrees
        gt_euler: (3,) array of GT angles in degrees
        seq: Rotation sequence (e.g., 'yxz') matching your pose estimation
    Returns:
        float: Error in degrees [0, 180]
    """
    # Create rotation objects
    r_pred = R.from_euler(seq, pred_euler, degrees=True)
    r_gt = R.from_euler(seq, gt_euler, degrees=True)
    
    # Compute relative rotation: R_diff = R_pred * R_gt^-1
    r_diff = r_pred * r_gt.inv()
    
    # The magnitude of the rotation vector represents the geodesic distance
    # .magnitude() returns radians, so we convert to degrees
    error_deg = np.rad2deg(r_diff.magnitude())
    
    return float(error_deg)


def compute_euler_error_wrapped(pred_deg, gt_deg):
    """
    Computes the shortest difference between angles in degrees.
    Handles wrapping: 359 vs 1 becomes 2.
    Also handles unbounded inputs: 721 vs 0 becomes 1.
    """
    # Calculate raw difference
    diff = pred_deg - gt_deg
    
    # Wrap to [-180, 180]
    # (diff + 180) % 360 - 180 ensures the result is always the shortest path
    wrapped_diff = (diff + 180) % 360 - 180
    
    return np.abs(wrapped_diff)


def uncertainty_evaluation(args, model, test_loader, device, cluster_pivot):
    sdd = args.sdd
    svd = args.svd
    vdd = sdd - svd

    all_results = []

    # epsilon thresholds for comparisons
    eps_pred = 0.1
    eps_rot = 0.1
    eps_trans = 0.1

    with torch.no_grad():
        for idx, (image, specimen_id, image_name, landmarks, pose_params) in tqdm(enumerate(test_loader)):
            specimen_id = specimen_id[0]
            image_name = image_name[0]  # must match 'image_id' in CSVs

            # =====================================================
            # GT pose parameters
            # =====================================================
            rotation_gt = np.array([
                np.rad2deg(pose_params[0].cpu().numpy()[0]),
                np.rad2deg(pose_params[1].cpu().numpy()[0]),
                np.rad2deg(pose_params[2].cpu().numpy()[0])
            ])
            translation_gt = np.array([
                pose_params[3].cpu().numpy()[0],
                pose_params[4].cpu().numpy()[0],
                pose_params[5].cpu().numpy()[0]
            ])

            # =====================================================
            # Manual translation offsets
            # =====================================================
            manual_translations_list = torch.tensor([[0.0, svd, 0.0]])

            # =====================================================
            # Forward pass: get prob maps, max prob + coords
            # =====================================================
            image_device = image.to(device)
            outputs = model(image_device)       # [B, C, H, W]
            probs = torch.sigmoid(outputs)
            B, C, H, W = probs.shape

            probs_flat = probs.view(B, C, -1)
            max_vals, max_indices = probs_flat.max(dim=2)  # [B, C], [B, C]

            pred_coords = torch.zeros((B, C, 2), device=device)
            for b in range(B):
                for c in range(C):
                    index = max_indices[b, c].item()
                    y, x = divmod(index, W)
                    pred_coords[b, c] = torch.tensor([x, y], device=device)

            # prediction-based visibility: max prob >= threshold
            max_vals_np = max_vals[0].cpu().numpy()  # [C]
            pred_visible_mask = max_vals_np >= args.pred_visibility_thresh  # bool [C]

            # =====================================================
            # GT 2D landmarks + GT visibility
            # =====================================================
            landmarks = landmarks.squeeze(0)   # shape: (C, 2)
            L_Proj_gt = np.array([
                [np.nan, np.nan] if (coord[0].item() == 0 and coord[1].item() == 0)
                else [coord[0].item(), coord[1].item()]
                for coord in landmarks
            ], dtype=np.float32)

            # GT visibility
            gt_visible_mask = ~np.isnan(L_Proj_gt).any(axis=1)  # [C]

            # Copy and convert to projection coordinates
            L_Proj_gt_cp = L_Proj_gt.copy()
            L_Proj_gt_cp[gt_visible_mask, 1] -= args.image_resize / 2
            L_Proj_gt_cp[gt_visible_mask, 0] -= args.image_resize / 2
            L_Proj_gt_cp[gt_visible_mask, 1] *= -1

            # =====================================================
            # Base visibility mask (how we treat 'existence')
            # =====================================================
            if args.visibility_mode == "pred":
                base_mask = pred_visible_mask.copy()
            elif args.visibility_mode == "gt":
                base_mask = gt_visible_mask.copy()
            elif args.visibility_mode == "both":
                base_mask = pred_visible_mask & gt_visible_mask
            else:
                raise ValueError(f"Unknown visibility_mode: {args.visibility_mode}")

            # If too few base visible lms, skip
            if base_mask.sum() < 3:
                print(f"[Skipping] {image_name}: < 3 base visible landmarks (mode={args.visibility_mode})")
                continue

            # =====================================================
            # Per-image per-landmark uncertainty: cluster distance
            # =====================================================
            # Default: all NaN
            cluster_vals = np.full(args.n_landmarks, np.nan, dtype=np.float32)

            if image_name in cluster_pivot.index:
                # Reindex to ensure columns 0..C-1 exist
                row = cluster_pivot.loc[image_name].reindex(range(args.n_landmarks))
                cluster_vals = row.to_numpy(dtype=np.float32)

            # =====================================================
            # Construct landmark weights from deviation (uncertainty)
            # =====================================================

            # cluster_vals = deviation per landmark
            deviation = cluster_vals.copy()

            # If deviation is missing for a landmark, treat as medium uncertainty
            # Deviations should be >= 0, but NaNs come from invisibility
            eps = 1e-6
            deviation = np.nan_to_num(deviation, nan=np.nanmedian(deviation))

            # Inverse deviation → higher deviation = lower weight
            inverse = 1.0 / (deviation + eps)

            # Normalize so that best landmark has weight = 1.0
            inverse /= np.max(inverse)

            # Finally, ensure weights are valid finite numbers
            weights_ver1 = np.nan_to_num(inverse, nan=1.0, posinf=1.0, neginf=0.0).astype(np.float32)

            ## Version2: Softmax-like weighting
            beta = 0.01  # can try higher values too

            eps = 1e-6
            deviation = np.nan_to_num(cluster_vals.copy(), nan=np.nanmedian(cluster_vals))

            weights_ver2 = np.exp(-beta * deviation)
            weights_ver2 /= np.max(weights_ver2)
            weights_ver2 = weights_ver2.astype(np.float32)

            # Version3
            dev = np.nan_to_num(cluster_vals.copy(), nan=np.nanmedian(cluster_vals))
            ranks = dev.argsort().argsort()  # 0 = smallest deviation
            weights_ver3 = 1.0 - ranks / (len(ranks) - 1)
            weights_ver3 = weights_ver3.astype(np.float32)

            # candidate landmarks for uncertainty-based dropping
            candidate_mask = base_mask & ~np.isnan(cluster_vals)
            uncertain_mask = np.zeros_like(base_mask)

            if args.top_k_landmarks > 0 and candidate_mask.sum() > args.top_k_landmarks:
                valid_indices = np.where(candidate_mask)[0]
                # sort by cluster distance ascending, then take largest (most uncertain)
                sorted_valid = valid_indices[np.argsort(cluster_vals[valid_indices])]
                topk = sorted_valid[-args.top_k_landmarks:]
                uncertain_mask[topk] = True

            # final filtered mask: base visible but not top-k uncertain
            filtered_mask = base_mask & ~uncertain_mask

            # Require at least 3 landmarks after filtering
            if filtered_mask.sum() < 3:
                print(f"[Skipping] {image_name}: < 3 landmarks after filtering")
                continue

            # =====================================================
            # Build 3D landmarks as before (unchanged)
            # =====================================================
            translation_gt_adj = translation_gt.copy()
            translation_gt_adj[1] -= manual_translations_list[0, 1].item()
            rx, ry, rz = rotation_gt
            tx, ty, tz = translation_gt_adj

            volume_path = f"{args.data_dir}/{specimen_id}/{specimen_id}_CT.nii.gz"
            volume = nib.load(volume_path)
            volume_shape = volume.shape
            center = np.array(volume_shape) // 2
            spacing = volume.header.get_zooms()[:3]
            spacing_x, spacing_y, spacing_z = spacing

            Slicer_3D_landmark = np.load(
                f"data/DeepFluoro/{specimen_id}/{specimen_id}_Landmarks_3D.npy"
            ).astype(np.float64)
            Slicer_3D_landmark -= center
            Slicer_3D_landmark[:, 0] *= spacing_x
            Slicer_3D_landmark[:, 1] *= spacing_y
            Slicer_3D_landmark[:, 2] *= spacing_z
            Slicer_3D_landmark += center

            Point_3D_landmark = (Slicer_3D_landmark - center).astype(np.float32)
            Point_3D_landmark[:, 1] += ((sdd - svd) / spacing_y)
            Point_3D_landmark[:, 2] -= (manual_translations_list[0, 2].item() / spacing_z)

            Point_3D_landmark_transformed = apply_transformation(
                Point_3D_landmark, -ry, rz, rx, tx, -ty, -tz,
                center=np.array([
                    0,
                    (sdd - svd) / spacing_y,
                    -(manual_translations_list[0, 2].item() / spacing_z)
                ])
            )
            Point_2D_landmark = project_point(Point_3D_landmark_transformed, H=(sdd / spacing_y))

            # Convert to CV2 coordinates
            Point_2D_landmark_cv2 = Point_2D_landmark.copy()
            Point_2D_landmark_cv2[:, 1] *= -1
            Point_2D_landmark_cv2[:, 0] += args.image_resize / 2
            Point_2D_landmark_cv2[:, 1] += args.image_resize / 2

            # =====================================================
            # 1) Prediction error (All vs Filtered)
            # =====================================================
            pred_coords_np = pred_coords[0].cpu().numpy().copy()

            # All: base visible only
            pred_all = pred_coords_np.copy()
            pred_all[~base_mask] = np.nan
            pred_err_all = float(np.sqrt(np.nanmean((pred_all - Point_2D_landmark_cv2) ** 2)))

            # Filtered: base visible but drop top-k uncertain
            pred_filt = pred_coords_np.copy()
            pred_filt[~filtered_mask] = np.nan
            pred_err_filtered = float(np.sqrt(np.nanmean((pred_filt - Point_2D_landmark_cv2) ** 2)))

            if abs(pred_err_filtered - pred_err_all) < eps_pred:
                pred_better = "tie"
            elif pred_err_filtered < pred_err_all:
                pred_better = "filtered"
            else:
                pred_better = "all"

            # # =====================================================
            # # 2) Pose estimation errors (All vs Filtered)
            # # =====================================================
            # # All: GT coords but respect base visibility
            # gt_all = L_Proj_gt_cp.copy()
            # gt_all[~base_mask] = np.nan

            # # All
            # rot_all, trans_all = pose_estimation(
            #     Point_3D_landmark, gt_all, sdd, svd, vdd, manual_translations_list
            # )

            # if np.isnan(rot_all).any() or np.isnan(trans_all).any():
            #     print(f"[Skipping] {image_name}: not enough valid landmarks for ALL pose")
            #     continue

            # rot_err_all = np.linalg.norm(rot_all - rotation_gt)
            # rot_err_all = compute_geodesic_distance(rot_all, rotation_gt, seq='yxz')
            # trans_err_all = np.linalg.norm(trans_all - translation_gt_adj)

            # # Filtered: GT coords but drop uncertain
            # gt_filtered = L_Proj_gt_cp.copy()
            # gt_filtered[~filtered_mask] = np.nan

            # # Filtered
            # rot_filt, trans_filt = pose_estimation(
            #     Point_3D_landmark, gt_filtered, sdd, svd, vdd, manual_translations_list
            # )

            # if np.isnan(rot_filt).any() or np.isnan(trans_filt).any():
            #     print(f"[Skipping] {image_name}: not enough valid landmarks for FILTERED pose")
            #     continue

            # rot_err_filt = np.linalg.norm(rot_filt - rotation_gt)
            # rot_err_filt = compute_geodesic_distance(rot_filt, rotation_gt, seq='yxz')
            # trans_err_filt = np.linalg.norm(trans_filt - translation_gt_adj)

            # if (abs(rot_err_filt - rot_err_all) < eps_rot and
            #     abs(trans_err_filt - trans_err_all) < eps_trans):
            #     pose_better = "tie"
            # elif (rot_err_filt < rot_err_all and trans_err_filt < trans_err_all):
            #     pose_better = "filtered"
            # else:
            #     pose_better = "all"

            # print(f'[{image_name}] Pose Err All: Rot {rot_err_all:.2f}, Trans {trans_err_all:.2f} | '
            #       f'Filtered: Rot {rot_err_filt:.2f}, Trans {trans_err_filt:.2f} | Better: {pose_better}')
            # exit()

            # =====================================================
            # 2) Pose estimation errors (All vs Filtered, using PRED 2D)
            # =====================================================
            # Build predicted 2D in projection-plane coordinates for pose
            # Start from predicted coords in CV2 pixel space
            #   pred_all / pred_filt are already NaN-masked by base_mask / filtered_mask

            # --- All ---
            L_Proj_pred_all = pred_all.copy()  # shape (C, 2), some rows NaN

            # Convert from CV2 coords (x, y) to projection-plane coords:
            # center at (0,0), +y upward
            L_Proj_pred_all[:, 1] -= args.image_resize / 2
            L_Proj_pred_all[:, 0] -= args.image_resize / 2
            L_Proj_pred_all[:, 1] *= -1

            rot_all, trans_all = pose_estimation(
                Point_3D_landmark, L_Proj_pred_all, sdd, svd, vdd, manual_translations_list
            )
            # print(L_Proj_pred_all) 

            if np.isnan(rot_all).any() or np.isnan(trans_all).any():
                print(f"[Skipping] {image_name}: not enough valid landmarks for ALL pose (pred-based)")
                continue

            # rot_err_all = np.linalg.norm(rot_all - rotation_gt)
            # trans_err_all = np.linalg.norm(trans_all - translation_gt_adj)
            # rot_err_all = float(np.sqrt(np.nanmean((rot_all - rotation_gt) ** 2)))
            rot_err_all = compute_geodesic_distance(rot_all, rotation_gt, seq='yxz')
            per_axis_errors = compute_euler_error_wrapped(rot_all, rotation_gt)
            rot_err_all = float(np.sqrt(np.nanmean(per_axis_errors ** 2)))
            trans_err_all = float(np.sqrt(np.nanmean((trans_all - translation_gt_adj) ** 2)))

            # --- Filtered ---
            L_Proj_pred_filt = pred_filt.copy()

            L_Proj_pred_filt[:, 1] -= args.image_resize / 2
            L_Proj_pred_filt[:, 0] -= args.image_resize / 2
            L_Proj_pred_filt[:, 1] *= -1

            rot_filt, trans_filt = pose_estimation(
                Point_3D_landmark, L_Proj_pred_filt, sdd, svd, vdd, manual_translations_list
            )
            # print(L_Proj_pred_filt)

            if np.isnan(rot_filt).any() or np.isnan(trans_filt).any():
                print(f"[Skipping] {image_name}: not enough valid landmarks for FILTERED pose (pred-based)")
                continue

            # rot_err_filt = np.linalg.norm(rot_filt - rotation_gt)
            # trans_err_filt = np.linalg.norm(trans_filt - translation_gt_adj)
            # rot_err_filt = float(np.sqrt(np.nanmean((rot_filt - rotation_gt) ** 2)))
            rot_err_filt = compute_geodesic_distance(rot_filt, rotation_gt, seq='yxz')
            per_axis_errors = compute_euler_error_wrapped(rot_filt, rotation_gt)
            rot_err_filt = float(np.sqrt(np.nanmean(per_axis_errors ** 2)))
            trans_err_filt = float(np.sqrt(np.nanmean((trans_filt - translation_gt_adj) ** 2)))

            # Compare
            if (abs(rot_err_filt - rot_err_all) < eps_rot and
                abs(trans_err_filt - trans_err_all) < eps_trans):
                pose_better = "tie"
            elif (rot_err_filt < rot_err_all and trans_err_filt < trans_err_all):
                pose_better = "filtered"
            else:
                pose_better = "all"

            # =====================================================
            # 3) Pose estimation using WEIGHTED PREDICTED LANDMARKS
            # =====================================================

            # Convert ALL predicted CV2 coords → projection-plane
            L_Proj_pred_w = pred_filt.copy()

            L_Proj_pred_w[:, 1] -= args.image_resize / 2
            L_Proj_pred_w[:, 0] -= args.image_resize / 2
            L_Proj_pred_w[:, 1] *= -1

            # Version 1 weights
            rot_w, trans_w = pose_estimation_weighted(
                Point_3D_landmark,
                L_Proj_pred_w,
                weights_ver1,
                sdd, svd, vdd,
                manual_translations_list
            )

            if np.isnan(rot_w).any() or np.isnan(trans_w).any():
                print(f"[Skipping] {image_name}: not enough valid landmarks for WEIGHTED pose")
                continue

            # Weighted pose error
            # rot_err_w_ver1 = float(np.sqrt(np.mean((rot_w - rotation_gt)**2)))
            rot_err_w_ver1 = compute_geodesic_distance(rot_w, rotation_gt, seq='yxz')
            per_axis_errors = compute_euler_error_wrapped(rot_w, rotation_gt)
            rot_err_w_ver1 = float(np.sqrt(np.nanmean(per_axis_errors ** 2)))
            trans_err_w_ver1 = float(np.sqrt(np.mean((trans_w - translation_gt_adj)**2)))

            # Version 2 weights
            rot_w, trans_w = pose_estimation_weighted(
                Point_3D_landmark,
                L_Proj_pred_w,
                weights_ver2,
                sdd, svd, vdd,
                manual_translations_list
            )
            if np.isnan(rot_w).any() or np.isnan(trans_w).any():
                print(f"[Skipping] {image_name}: not enough valid landmarks for WEIGHTED pose")
                continue    
            # rot_err_w_ver2 = float(np.sqrt(np.mean((rot_w - rotation_gt)**2)))   
            rot_err_w_ver2 = compute_geodesic_distance(rot_w, rotation_gt, seq='yxz')
            per_axis_errors = compute_euler_error_wrapped(rot_w, rotation_gt)
            rot_err_w_ver2 = float(np.sqrt(np.nanmean(per_axis_errors ** 2)))
            trans_err_w_ver2 = float(np.sqrt(np.mean((trans_w - translation_gt_adj)**2)))

            # Version 3 weights
            rot_w, trans_w = pose_estimation_weighted(
                Point_3D_landmark,
                L_Proj_pred_w,
                weights_ver3,
                sdd, svd, vdd,
                manual_translations_list
            )
            if np.isnan(rot_w).any() or np.isnan(trans_w).any():
                print(f"[Skipping] {image_name}: not enough valid landmarks for WEIGHTED pose")
                continue    
            # rot_err_w_ver3 = float(np.sqrt(np.mean((rot_w - rotation_gt)**2)))   
            rot_err_w_ver3 = compute_geodesic_distance(rot_w, rotation_gt, seq='yxz')
            per_axis_errors = compute_euler_error_wrapped(rot_w, rotation_gt)
            rot_err_w_ver3 = float(np.sqrt(np.nanmean(per_axis_errors ** 2)))
            trans_err_w_ver3 = float(np.sqrt(np.mean((trans_w - translation_gt_adj)**2)))


            # print()
            # # print(f'[{image_name}] Pred Err All: {pred_err_all:.2f}, Filtered: {pred_err_filtered:.2f}, Better: {pred_better}')
            # print(f'[{image_name}] Pred Err All: {pred_err_all:.2f}, Filtered: {pred_err_filtered:.2f}')
            # print(f'All: Rot {rot_err_all:.2f} Trans {trans_err_all:.2f}')
            # print(f'Fil: Rot {rot_err_filt:.2f}, Trans {trans_err_filt:.2f}')
            # print(f'Wei: Rot {rot_err_w:.2f}, 'f'Trans {trans_err_w:.2f}')

            # =====================================================
            # 4) Prediction error — GT-distance filtered landmarks
            # =====================================================
            # 새 마스크 생성
            filtered_mask_gt = base_mask.copy()

            # GT 2D landmark (cv2 coords) vs predicted 2D coords 거리 계산
            dist_vals = np.full(args.n_landmarks, np.nan, dtype=np.float32)

            for c in range(args.n_landmarks):
                if base_mask[c] and not np.isnan(Point_2D_landmark_cv2[c]).any():
                    px, py = pred_coords_np[c]
                    gx, gy = Point_2D_landmark_cv2[c]
                    dist_vals[c] = np.sqrt((px - gx)**2 + (py - gy)**2)

            valid_dist_mask = base_mask & ~np.isnan(dist_vals)
            uncertain_mask_gt = np.zeros_like(base_mask)

            # GT-based Top-K 제거
            if valid_dist_mask.sum() > args.top_k_landmarks and args.top_k_landmarks > 0:
                valid_indices = np.where(valid_dist_mask)[0]
                sorted_valid = valid_indices[np.argsort(dist_vals[valid_indices])]
                topk_gt = sorted_valid[-args.top_k_landmarks:]  # distance 가장 큰 애들
                uncertain_mask_gt[topk_gt] = True

            # GT 필터링 마스크
            filtered_mask_gt = base_mask & ~uncertain_mask_gt

            # landmark 최소 개수(3개) 검사
            if filtered_mask_gt.sum() < 3:
                print(f"[Skipping GT-filter] {image_name}: < 3 landmarks after GT-distance filtering")
                filtered_mask_gt = None  # invalidate

            pred_gt = pred_coords_np.copy()
            pred_gt[~filtered_mask_gt] = np.nan
            pred_err_gt = float(np.sqrt(np.nanmean((pred_gt - Point_2D_landmark_cv2)**2)))

            # =====================================================
            # 5) Pose estimation (predicted 2D) — GT-distance filtered
            # =====================================================
            # predicted coords → projection-plane 변환
            L_Proj_pred_gt = pred_gt.copy()
            L_Proj_pred_gt[:, 1] -= args.image_resize / 2
            L_Proj_pred_gt[:, 0] -= args.image_resize / 2
            L_Proj_pred_gt[:, 1] *= -1

            rot_gt, trans_gt = pose_estimation(
                Point_3D_landmark, L_Proj_pred_gt, sdd, svd, vdd, manual_translations_list
            )

            # rot_err_gt = float(np.sqrt(np.nanmean((rot_gt - rotation_gt)**2)))
            rot_err_gt = compute_geodesic_distance(rot_gt, rotation_gt, seq='yxz')
            per_axis_errors = compute_euler_error_wrapped(rot_gt, rotation_gt)
            rot_err_gt = float(np.sqrt(np.nanmean(per_axis_errors ** 2)))
            trans_err_gt = float(np.sqrt(np.nanmean((trans_gt - translation_gt_adj)**2)))

            # All vs GT-filtered 비교
            if (abs(rot_err_gt - rot_err_all) < eps_rot and
                abs(trans_err_gt - trans_err_all) < eps_trans):
                pose_better_gt = "tie"
            elif (rot_err_gt < rot_err_all and trans_err_gt < trans_err_all):
                pose_better_gt = "gt_filtered"
            else:
                pose_better_gt = "all"

            # =====================================================
            # Store per-image results
            # =====================================================
            all_results.append({
                "image": image_name,
                "pred_err_all": pred_err_all,
                "pred_err_filtered": pred_err_filtered,
                "pred_err_gt": pred_err_gt,
                "pred_better": pred_better,

                "rot_err_all": rot_err_all,
                "rot_err_filtered": rot_err_filt,
                "rot_err_weighted_ver1": rot_err_w_ver1,
                "rot_err_weighted_ver2": rot_err_w_ver2,
                "rot_err_weighted_ver3": rot_err_w_ver3,
                "rot_err_gt": rot_err_gt,

                "trans_err_all": trans_err_all,
                "trans_err_filtered": trans_err_filt,
                "trans_err_weighted_ver1": trans_err_w_ver1,
                "trans_err_weighted_ver2": trans_err_w_ver2,
                "trans_err_weighted_ver3": trans_err_w_ver3,
                "trans_err_gt": trans_err_gt,
                
                "pose_better": pose_better,
                # "pose_better_weighted": pose_better_w,
                "pose_better_gt": pose_better_gt,
            })

    return all_results