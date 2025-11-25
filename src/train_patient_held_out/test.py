import os
import torch
import torch.nn as nn

from tqdm import tqdm

from src.train_patient_held_out.data_loader import dataloader

from src.test.data import load_or_compute_per_image_uncertainty, save_mc_predictions_csv
from src.test.uncertainty import uncertainty_evaluation
from src.test.result import save_csv
from src.test.visualization import plot_overall_trends


def test_model(args, model, device, test_loader):
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout) or isinstance(m, nn.Dropout2d):
            m.train()

    all_sim_coords = []
    all_gt_coords = []
    all_image_names = []

    for idx, (image, specimen_id, image_name, landmarks, pose_params) in enumerate(tqdm(test_loader, desc="Testing")):
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
        mc_preds, gt_coords, image_names = test_model(args, model_dropout, device, test_loader)
        save_mc_predictions_csv(args, mc_preds, gt_coords, image_names, csv_dir)

    df_unc, cluster_pivot = load_or_compute_per_image_uncertainty(csv_dir, dropout_rate=args.dropout_rate)

    suffix = f"perImage_{args.visibility_mode}"
    all_results = uncertainty_evaluation(args, model, test_loader, device, cluster_pivot)
    results_df = save_csv(args, all_results, suffix=suffix)
    plot_overall_trends(args, results_df, plot_dir, suffix=suffix)
