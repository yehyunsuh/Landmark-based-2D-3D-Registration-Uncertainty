import os
import cv2
import torch
import numpy as np
import torch.nn as nn

from tqdm import tqdm

from src.train_patient_held_out.data_loader import dataloader

from src.test.data import load_or_compute_per_image_uncertainty, save_predictions_csv
from src.test.uncertainty import uncertainty_evaluation
from src.test.result import save_csv
from src.test.visualization import plot_overall_trends


def test_model(args, model, device, test_loader, overlay_dir=None):
    """
    Run *vanilla* model once per image (no MC dropout) but return
    predictions in the same shape as MC: [S=1, N, C, 2].
    """
    model.eval()  # keep dropout in eval (no randomness)

    all_preds_coords = []
    all_gt_coords = []
    all_image_names = []

    for idx, (image, specimen_id, image_name, landmarks, pose_params) in enumerate(tqdm(test_loader, desc="Testing")):
        image = image.to(device)
        landmarks = landmarks.to(device)
        if landmarks.ndim == 2:
            landmarks = landmarks.unsqueeze(0)

        # invisible -> NaN
        zero_mask = (landmarks == 0).all(dim=2)
        landmarks[zero_mask] = float('nan')
        all_gt_coords.append(landmarks)

        B, C = 1, args.n_landmarks  # test loader likely yields B=1

        # ---- single forward pass ----
        outputs = model(image)
        probs = torch.sigmoid(outputs)
        _, _, H, W = probs.shape

        flat = probs.view(B, C, -1)
        max_idx = flat.argmax(dim=2)

        preds = torch.zeros((B, C, 2), device=device)
        for c in range(C):
            idx1d = max_idx[0, c].item()
            y, x = divmod(idx1d, W)
            preds[0, c] = torch.tensor([x, y], device=device)

        # ---------- save overlay image ----------
        if overlay_dir is not None:
            img_np = image[0].detach().cpu().numpy()  # [C, H, W] (normalized)

            # ---- undo A.Normalize ----
            # Albumentations Normalize: (img - mean) / std
            # → img = img_norm * std + mean
            if img_np.ndim == 3 and img_np.shape[0] == 3:
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
                std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
                img_np = img_np * std + mean        # back to [0,1]-ish
                img_np = np.clip(img_np, 0.0, 1.0)
                img_disp = (img_np.transpose(1, 2, 0) * 255).astype(np.uint8)  # [H,W,3], RGB
                img_disp = cv2.cvtColor(img_disp, cv2.COLOR_RGB2BGR)          # to BGR for OpenCV
            else:
                # Fallback: generic min-max for non-3-channel case
                if img_np.ndim == 3:
                    if img_np.shape[0] == 1:
                        img_np = img_np[0]
                    else:
                        img_np = np.transpose(img_np, (1, 2, 0))
                img_min, img_max = img_np.min(), img_np.max()
                if img_max > img_min:
                    img_norm = (img_np - img_min) / (img_max - img_min)
                else:
                    img_norm = np.zeros_like(img_np)
                img_disp = (img_norm * 255).astype(np.uint8)
                if img_disp.ndim == 2:
                    img_disp = cv2.cvtColor(img_disp, cv2.COLOR_GRAY2BGR)
                elif img_disp.shape[2] == 1:
                    img_disp = cv2.cvtColor(img_disp, cv2.COLOR_GRAY2BGR)

            gt_np = landmarks[0].detach().cpu().numpy()      # [C, 2] with NaNs
            pred_np = preds[0].detach().cpu().numpy()        # [C, 2]

            gt_vis   = ~np.isnan(gt_np).any(axis=1)
            pred_vis = ~np.isnan(pred_np).any(axis=1)

            for c in range(C):
                # GT in BLUE
                if gt_vis[c]:
                    xg, yg = gt_np[c]
                    cv2.circle(img_disp, (int(round(xg)), int(round(yg))), 4, (255, 0, 0), -1,)
                # prediction in RED
                if pred_vis[c]:
                    xp, yp = pred_np[c]
                    cv2.circle(img_disp, (int(round(xp)), int(round(yp))), 4, (0, 0, 255), -1,)

            out_name = image_name[0]  # e.g. "0476.png"
            out_path = os.path.join(overlay_dir, out_name)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            cv2.imwrite(out_path, img_disp)

        # shape [S=1, B, C, 2] so it matches MC format
        sim_coords_batch = preds.unsqueeze(0)
        all_preds_coords.append(sim_coords_batch)
        all_image_names.append(image_name[0])

    # Concatenate over images → [1, N, C, 2]
    all_preds_coords = torch.cat(all_preds_coords, dim=1)
    all_gt_coords = torch.cat(all_gt_coords, dim=0)

    return all_preds_coords, all_gt_coords, all_image_names


def test_model_uncertainty(args, model, device, test_loader):
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout) or isinstance(m, nn.Dropout2d):
            m.train()

    all_sim_coords = []
    all_gt_coords = []
    all_image_names = []

    for idx, (image, specimen_id, image_name, landmarks, pose_params) in enumerate(tqdm(test_loader, desc="Testing Uncertainty")):
        image = image.to(device)
        landmarks = landmarks.to(device)
        if landmarks.ndim == 2: 
            landmarks = landmarks.unsqueeze(0)

        # create mask for invisible landmarks
        zero_mask = (landmarks == 0).all(dim=2)
        landmarks[zero_mask] = float('nan')

        all_gt_coords.append(landmarks)

        B, C = 1, args.n_landmarks  # test loader likely yields B=1
        sim_coords_batch = []

        for _ in range(args.n_simulations):
            outputs = model(image)
            probs = torch.sigmoid(outputs)
            _, _, H, W = probs.shape

            flat = probs.view(B, C, -1)
            max_idx = flat.argmax(dim=2)

            preds = torch.zeros((B,C,2), device=device)
            for c in range(C):
                idx1d = max_idx[0,c].item()
                y, x = divmod(idx1d, W)
                preds[0,c] = torch.tensor([x,y], device=device)
            sim_coords_batch.append(preds.clone())

        sim_coords_batch = torch.stack(sim_coords_batch)
        all_sim_coords.append(sim_coords_batch)
        all_image_names.append(image_name[0])

    all_sim_coords = torch.cat(all_sim_coords, dim=1)
    all_gt_coords = torch.cat(all_gt_coords, dim=0)

    return all_sim_coords, all_gt_coords, all_image_names


def test(args, model, model_dropout, device):
    csv_dir = os.path.join(args.vis_dir, f"prediction_{args.dropout_rate}", "csv_results")
    plot_dir = os.path.join(args.vis_dir, f"prediction_{args.dropout_rate}", "uncertainty_plots")
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    test_loader = dataloader(args, data_type='test')
    if args.test_prediction:
        preds, gt_coords, image_names = test_model(args, model, device, test_loader, overlay_dir=f'{args.vis_dir}/prediction_{args.dropout_rate}/overlays')
        save_predictions_csv(args, image_names, csv_dir, preds, gt_coords, prefix="predictions")

        mc_preds, _, _ = test_model_uncertainty(args, model_dropout, device, test_loader)
        save_predictions_csv(args, image_names, csv_dir, mc_preds, gt_coords, prefix="mc_predictions")

    df_unc, cluster_pivot = load_or_compute_per_image_uncertainty(csv_dir, dropout_rate=args.dropout_rate)

    suffix = f"perImage_{args.visibility_mode}"
    all_results = uncertainty_evaluation(args, model, test_loader, device, cluster_pivot)
    results_df = save_csv(args, all_results, suffix=suffix)
    plot_overall_trends(args, results_df, plot_dir, suffix=suffix)
