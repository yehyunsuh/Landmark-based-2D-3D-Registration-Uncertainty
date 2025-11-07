import os
import torch
import pandas as pd

from tqdm import tqdm

from src.train_patient_specific.data_loader import dataloader
from src.train.visualization import (
    overlay_gt_masks, overlay_pred_masks, overlay_pred_coords,
    create_gif, plot_training_results
)

def test_model(args, model, device, test_loader):
    model.eval()
    all_pred_coords = []
    all_gt_coords = []

    with torch.no_grad():
        for idx, (images, _, landmarks) in enumerate(tqdm(test_loader, desc="Testing")):
            images = images.to(device)
            outputs = model(images)

            probs = torch.sigmoid(outputs)
            B, C, H, W = probs.shape
            probs_flat = probs.view(B, C, -1)
            max_indices = probs_flat.argmax(dim=2)

            pred_coords = torch.zeros((B, C, 2), device=device)
            for b in range(B):
                for c in range(C):
                    index = max_indices[b, c].item()
                    y, x = divmod(index, W)
                    pred_coords[b, c] = torch.tensor([x, y], device=device)

            gt_coords = torch.tensor(landmarks, dtype=torch.float32, device=device)
            if gt_coords.ndim == 2:
                gt_coords = gt_coords.unsqueeze(0)

            all_pred_coords.append(pred_coords)
            all_gt_coords.append(gt_coords)

    all_pred_coords = torch.cat(all_pred_coords, dim=0)  # [N, C, 2]
    all_gt_coords = torch.cat(all_gt_coords, dim=0)

    # Mask (0, 0) GT for distance calculation only
    diff = all_pred_coords - all_gt_coords
    dists = torch.norm(diff, dim=2)
    mask = (all_gt_coords != 0).any(dim=2)  # [B, C]
    dists[~mask] = float("nan")  # Don't include in distance average

    return dists


def test(args, model, device):
    history = {
        "mean_landmark_error": [],
        "landmark_errors": {str(c): [] for c in range(args.n_landmarks)},
    }

    test_loader = dataloader(args, data_type='test')

    dists = test_model(args, model, device, test_loader)

    mean_dist = torch.nanmean(dists).item()

    print(
        f"Mean Dist: {mean_dist:.4f} | "
    )

    history["mean_landmark_error"].append(mean_dist)

    for c in range(dists.shape[1]):
        history["landmark_errors"][str(c)].append(torch.nanmean(dists[:, c]).item())

    rows = []
    row = [
        history["mean_landmark_error"][-1],
    ]
    row += [history["landmark_errors"][str(c)][-1] for c in range(args.n_landmarks)]
    rows.append(row)

    columns = ["mean_dist"]
    columns += [f"landmark{c+1}_dist" for c in range(args.n_landmarks)] 

    df = pd.DataFrame(rows, columns=columns)

    os.makedirs(f'{args.result_dir}/{args.wandb_name}/test_results', exist_ok=True)
    df.to_csv(f"{args.result_dir}/{args.wandb_name}/test_results/test_results.csv", index=False)
    print(f"📄 Saved test results to {args.result_dir}/{args.wandb_name}/test_results/test_results.csv")