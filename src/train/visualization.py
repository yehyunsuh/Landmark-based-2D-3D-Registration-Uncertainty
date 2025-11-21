import os
import cv2
import torch
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def overlay_gt_masks(args, images, masks, pred_coords, gt_coords, epoch, total_epoch, idx):
    for b in range(images.shape[0]):
        pred = pred_coords[b].cpu().numpy()
        gt = gt_coords[b].cpu().numpy()

        img = images[b].cpu().permute(1, 2, 0).numpy()
        img = (img * np.array([0.229, 0.224, 0.225]) +
               np.array([0.485, 0.456, 0.406])) * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8).copy()

        gt_mask = masks[b].sum(0).cpu().numpy()
        gt_mask = (gt_mask > 0).astype(np.uint8) * 255
        gt_mask_rgb = cv2.cvtColor(gt_mask, cv2.COLOR_GRAY2BGR)

        overlay = cv2.addWeighted(img, 0.7, gt_mask_rgb, 0.3, 0)

        for c in range(pred.shape[0]):
            gx, gy = int(gt[c, 0]), int(gt[c, 1])
            if gx == 0 and gy == 0:
                continue  # Skip invisible GT landmarks (and predicted points at those channels)
            px, py = int(pred[c, 0]), int(pred[c, 1])
            cv2.circle(overlay, (gx, gy), 4, (255, 0, 0), -1)
            cv2.circle(overlay, (px, py), 4, (0, 0, 255), -1)

        if epoch % 10 == 0 or epoch == total_epoch - 1:
            os.makedirs(f"{args.vis_dir}/Epoch{epoch}", exist_ok=True)
            cv2.imwrite(
                f"{args.vis_dir}/Epoch{epoch}/Epoch{epoch}_Batch{idx}_overlay_gt.png", overlay
            )
        cv2.imwrite(f"{args.vis_dir}/Batch{idx}_overlay_gt.png", overlay)

        return overlay


def overlay_pred_masks(args, images, outputs, pred_coords, gt_coords, epoch, total_epoch, idx):
    overlay_list = []

    for b in range(images.shape[0]):
        pred = pred_coords[b].cpu().numpy()
        gt = gt_coords[b].cpu().numpy()

        img = images[b].cpu().permute(1, 2, 0).numpy()
        img = (img * np.array([0.229, 0.224, 0.225]) +
               np.array([0.485, 0.456, 0.406])) * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8).copy()

        C = outputs.shape[1]
        for c in range(C):
            mask = (torch.sigmoid(outputs[b, c]) > 0.5).float().cpu().numpy()
            mask = (mask * 255).astype(np.uint8)
            mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

            overlay = cv2.addWeighted(img, 0.7, mask_rgb, 0.3, 0)
            
            gx, gy = int(gt[c, 0]), int(gt[c, 1])
            if gx == 0 and gy == 0:
                continue  # Skip invisible GT landmarks (and predicted points at those channels)
            px, py = int(pred[c, 0]), int(pred[c, 1])
            cv2.circle(overlay, (gx, gy), 4, (255, 0, 0), -1)
            cv2.circle(overlay, (px, py), 4, (0, 0, 255), -1)

            if epoch % 10 == 0 or epoch == total_epoch - 1:
                os.makedirs(f"{args.vis_dir}/Epoch{epoch}", exist_ok=True)
                cv2.imwrite(
                    f"{args.vis_dir}/Epoch{epoch}/Epoch{epoch}_Batch{idx}_Landmark{c}.png", overlay
                )
            cv2.imwrite(f"{args.vis_dir}/Batch{idx}_Landmark{c}.png", overlay)

            overlay_list.append(overlay)

    return overlay_list


def overlay_pred_coords(args, images, pred_coords, gt_coords, epoch, total_epoch, idx, train_mode=False, test_mode=False):
    for b in range(images.shape[0]):
        pred = pred_coords[b].cpu().numpy()
        gt = gt_coords[b].cpu().numpy()

        img = images[b].cpu().permute(1, 2, 0).numpy()
        img = (img * np.array([0.229, 0.224, 0.225]) +
               np.array([0.485, 0.456, 0.406])) * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8).copy()

        for c in range(pred.shape[0]):
            gx, gy = int(gt[c, 0]), int(gt[c, 1])
            if gx == 0 and gy == 0:
                continue  # Skip invisible GT landmarks (and predicted points at those channels)
            px, py = int(pred[c, 0]), int(pred[c, 1])
            cv2.circle(img, (gx, gy), 4, (255, 0, 0), -1)
            cv2.circle(img, (px, py), 4, (0, 0, 255), -1)

        if train_mode:
            if epoch % 10 == 0 or epoch == total_epoch - 1:
                os.makedirs(f"{args.vis_dir}/Epoch{epoch}", exist_ok=True)
                cv2.imwrite(
                    f"{args.vis_dir}/Epoch{epoch}/Epoch{epoch}_Batch{idx}_overlay_pred.png", img
                )
            cv2.imwrite(f"{args.vis_dir}/Batch{idx}_overlay_pred.png", img)
        if test_mode:
            cv2.imwrite(f"{args.vis_dir}/Test_Batch{idx}_overlay_pred.png", img)

        return img


def create_gif(args, gt_mask_w_coords_image_list, pred_mask_w_coords_image_list_list, coords_image_list):
    def convert_to_numpy(image_list):
        """
        Converts a list of tensors or normalized images to uint8 numpy arrays (RGB).
        """
        converted = []
        for img in image_list:
            if isinstance(img, torch.Tensor):
                img = img.detach().cpu().numpy()
            if img.shape[0] == 3:
                img = img.transpose(1, 2, 0)
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            if img.shape[2] == 3:  # Only if it's a color image
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            converted.append(img)
        return converted

    # gt_mask_frames = convert_to_numpy(gt_mask_w_coords_image_list)

    # pred_mask_frames = []
    # for i in range(len(pred_mask_w_coords_image_list_list[0])):
    #     per_landmark = [frame_list[i] for frame_list in pred_mask_w_coords_image_list_list]
    #     pred_mask_frames.append(convert_to_numpy(per_landmark))

    coords_frames = convert_to_numpy(coords_image_list)

    # imageio.mimsave(f"{args.vis_dir}/gt_mask_with_coords.gif", gt_mask_frames, fps=10)
    # for i, frames in enumerate(pred_mask_frames):
    #     imageio.mimsave(f"{args.vis_dir}/pred_mask_with_coords_{i}.gif", frames, fps=10)
    imageio.mimsave(f"{args.vis_dir}/pred_coords_only.gif", coords_frames, fps=10)

    print("🖼️ Saved training progress GIFs to train_results/")


def plot_training_results(args, history, graph_dir):
    # Loss curve
    plt.figure(figsize=(12, 8))
    plt.plot(history["epoch"], history["train_loss"], label="Train Loss")
    plt.plot(history["epoch"], history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    # plt.savefig(f"{args.result_dir}/{args.wandb_name}/graph/loss_curve.png")
    plt.savefig(f"{graph_dir}/loss_curve.png")

    # Mean landmark error
    plt.figure(figsize=(12, 8))
    plt.plot(history["epoch"], history["mean_landmark_error"], label="Mean Landmark Error")
    plt.xlabel("Epoch")
    plt.ylabel("Error (px)")
    plt.grid(True)
    # plt.savefig(f"{args.result_dir}/{args.wandb_name}/graph/mean_landmark_error.png")
    plt.savefig(f"{graph_dir}/mean_landmark_error.png")

    # Log-scale version
    plt.figure(figsize=(12, 8))
    plt.plot(history["epoch"], history["mean_landmark_error"], label="Mean Landmark Error")
    plt.yscale("log")
    plt.gca().yaxis.set_major_locator(ticker.LogLocator(base=10.0))
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'$10^{{{int(np.log10(y))}}}$'))
    plt.xlabel("Epoch")
    plt.ylabel("Error (px)")
    plt.grid(True, which='both', linestyle='--')
    # plt.savefig(f"{args.result_dir}/{args.wandb_name}/graph/mean_landmark_error_log.png")
    plt.savefig(f"{graph_dir}/mean_landmark_error_log.png")

    # Per-landmark error
    plt.figure(figsize=(12, 8))
    for k, v in history["landmark_errors"].items():
        plt.plot(history["epoch"], v, label=f"Landmark {k}")
    plt.xlabel("Epoch")
    plt.ylabel("Error (px)")
    plt.legend()
    plt.grid(True)
    # plt.savefig(f"{args.result_dir}/{args.wandb_name}/graph/per_landmark_error.png")
    plt.savefig(f"{graph_dir}/per_landmark_error.png")

    # Log-scale per-landmark error
    plt.figure(figsize=(12, 8))
    for k, v in history["landmark_errors"].items():
        plt.plot(history["epoch"], v, label=f"Landmark {k}")
    plt.yscale("log")
    plt.gca().yaxis.set_major_locator(ticker.LogLocator(base=10.0))
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'$10^{{{int(np.log10(y))}}}$'))
    plt.xlabel("Epoch")
    plt.ylabel("Error (px)")
    plt.legend()
    plt.grid(True, which='both', linestyle='--')
    # plt.savefig(f"{args.result_dir}/{args.wandb_name}/graph/per_landmark_error_log.png")
    plt.savefig(f"{graph_dir}/per_landmark_error_log.png")

    # Mean Dice score
    plt.figure(figsize=(12, 8))
    plt.plot(history["epoch"], history["mean_dice"], label="Mean Dice")
    plt.xlabel("Epoch")
    plt.ylabel("Dice Score")
    plt.grid(True)
    # plt.savefig(f"{args.result_dir}/{args.wandb_name}/graph/mean_dice_score.png")
    plt.savefig(f"{graph_dir}/mean_dice_score.png")

    # Per-landmark Dice scores
    plt.figure(figsize=(12, 8))
    for k, v in history["dice_scores"].items():
        plt.plot(history["epoch"], v, label=f"Landmark {k}")
    plt.xlabel("Epoch")
    plt.ylabel("Dice Score")
    plt.legend()
    plt.grid(True)
    # plt.savefig(f"{args.result_dir}/{args.wandb_name}/graph/per_landmark_dice.png")
    plt.savefig(f"{graph_dir}/per_landmark_dice.png")

    plt.close("all")