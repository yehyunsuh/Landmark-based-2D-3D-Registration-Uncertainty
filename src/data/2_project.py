import os
import re
import cv2
import csv
import torch
import argparse
import numpy as np
import nibabel as nib

from glob import glob
from tqdm import tqdm

from src.train.utils import set_seed

from diffdrr.drr import DRR
from diffdrr.data import read


def project(args, device='cuda'):
    sdd = args.sdd
    height = args.height
    width = args.width
    sample_size = args.sample_size
    n_landmarks = args.n_landmarks

    if args.task_type == 'easy':
        rotation_range = [(-15, 15), (-15, 15), (-5, 5)]  # degrees
        # translation_range = [(-50, 50), (450, 550), (-50, 50)]  # mm
    elif args.task_type == 'medium':
        rotation_range = [(-30, 30), (-30, 30), (-10, 10)]  # degrees
        # translation_range = [(-50, 50), (450, 550), (-50, 50)]  # mm
    elif args.task_type == 'hard':  
        rotation_range = [(-45, 45), (-45, 45), (-15, 15)]  # degrees
        # translation_range = [(-50, 50), (450, 550), (-50, 50)]  # mm
    else:
        raise ValueError("Unknown data type")
    translation_range = [(-50, 50), (-50, 50), (-50, 50)]  # mm

    specimen_path_list = sorted(glob(f"{args.data_dir}/{args.unzip_dir}/*"))
    for specimen_path in specimen_path_list:
        print(f"Processing specimen: {specimen_path}")
        os.makedirs(f'{specimen_path}/{args.drr_dir}_{args.task_type}', exist_ok=True)
        os.makedirs(f'{specimen_path}/{args.drr_csv_dir}_{args.task_type}', exist_ok=True)
        os.makedirs(f'{specimen_path}/{args.drr_params_csv_dir}', exist_ok=True)

        specimen_id = os.path.basename(specimen_path)
        specimen_volume_path = os.path.join(specimen_path, f"{specimen_id}_CT.nii.gz")

        manual_rotations_list = torch.tensor([[0.0, 0.0, 0.0]])
        manual_translations_list = torch.tensor([[0.0, 400.0, 0.0]])
        if specimen_id == '17-1882':
            # manual_rotations_list = torch.tensor([[0.0, 0.0, 0.0]])
            manual_translations_list = torch.tensor([[0.0, 400.0, -20.0]])
        elif specimen_id == '17-1905':
            # manual_rotations_list = torch.tensor([[0.0, 0.0, 0.0]])
            manual_translations_list = torch.tensor([[0.0, 400.0, 0.0]])
        elif specimen_id == '18-0725':
            # manual_rotations_list = torch.tensor([[0.0, 0.0, 0.0]])
            manual_translations_list = torch.tensor([[0.0, 400.0, 70.0]])
        elif specimen_id == '18-1109':
            # manual_rotations_list = torch.tensor([[0.0, 0.0, 0.0]])
            manual_translations_list = torch.tensor([[0.0, 400.0, 50.0]])
        elif specimen_id == '18-2799':
            # manual_rotations_list = torch.tensor([[0.0, 0.0, 0.0]])
            manual_translations_list = torch.tensor([[0.0, 400.0, 40.0]])
        elif specimen_id == '18-2800':
            # manual_rotations_list = torch.tensor([[0.0, 0.0, 0.0]])
            manual_translations_list = torch.tensor([[0.0, 400.0, 0.0]])
        else:
            raise ValueError("Unknown specimen ID")

        ct_image = nib.load(specimen_volume_path)
        ct_header = ct_image.header
        pixel_spacing = ct_header.get_zooms()

        total_landmark_2D_array = np.zeros((n_landmarks, sample_size, 2), dtype=np.float32)

        # ======================================================
        # --- Controlled generation of rotations / translations ---
        # ======================================================
        n_rot_only = int(sample_size // 3)  # 200
        n_trans_only = int(sample_size // 3)  # 200
        n_both = sample_size - n_rot_only - n_trans_only  # 200

        # --- Initialize zero arrays for all 500 ---
        rotations_all = np.zeros((sample_size, 3), dtype=np.float32)
        translations_all = np.zeros((sample_size, 3), dtype=np.float32)

        # --- Subset A: random rotation only ---
        r1 = np.random.uniform(np.deg2rad(rotation_range[0][0]), np.deg2rad(rotation_range[0][1]), n_rot_only)
        r2 = np.random.uniform(np.deg2rad(rotation_range[1][0]), np.deg2rad(rotation_range[1][1]), n_rot_only)
        r3 = np.random.uniform(np.deg2rad(rotation_range[2][0]), np.deg2rad(rotation_range[2][1]), n_rot_only)
        rotations_all[:n_rot_only, :] = np.vstack([r1, r2, r3]).T

        # --- Subset B: random translation only ---
        tx = np.random.uniform(translation_range[0][0], translation_range[0][1], n_trans_only)
        ty = np.random.uniform(translation_range[1][0], translation_range[1][1], n_trans_only)
        tz = np.random.uniform(translation_range[2][0], translation_range[2][1], n_trans_only)
        translations_all[n_rot_only:n_rot_only + n_trans_only, :] = np.vstack([tx, ty, tz]).T

        # --- Subset C: both random rotation & translation ---
        start_idx = n_rot_only + n_trans_only
        r1 = np.random.uniform(np.deg2rad(rotation_range[0][0]), np.deg2rad(rotation_range[0][1]), n_both)
        r2 = np.random.uniform(np.deg2rad(rotation_range[1][0]), np.deg2rad(rotation_range[1][1]), n_both)
        r3 = np.random.uniform(np.deg2rad(rotation_range[2][0]), np.deg2rad(rotation_range[2][1]), n_both)
        rotations_all[start_idx:, :] = np.vstack([r1, r2, r3]).T

        tx = np.random.uniform(translation_range[0][0], translation_range[0][1], n_both)
        ty = np.random.uniform(translation_range[1][0], translation_range[1][1], n_both)
        tz = np.random.uniform(translation_range[2][0], translation_range[2][1], n_both)
        translations_all[start_idx:, :] = np.vstack([tx, ty, tz]).T

        # Convert to tensors and add manual offsets
        rotations_list = torch.tensor(rotations_all, dtype=torch.float32) + manual_rotations_list
        translations_list = torch.tensor(translations_all, dtype=torch.float32) + manual_translations_list

        # --- Ensure the first sample is always the manual pose ---
        rotations_list[0] = manual_rotations_list[0]
        translations_list[0] = manual_translations_list[0]
        print(f"Generated {n_rot_only} rot-only, {n_trans_only} trans-only, {n_both} both (total {sample_size}) samples.")

        pose_records = []
        for i in range(sample_size):
            rx, ry, rz = rotations_list[i].to(device)
            tx, ty, tz = translations_list[i].to(device)
            
            image_name = f'{specimen_id}_{i:04d}.png'
            pose_records.append([
                specimen_id, image_name, args.task_type,
                rx.item(), ry.item(), rz.item(),
                tx.item(), ty.item(), tz.item()
            ])
        
        # Save pose records to CSV
        pose_csv_path = f'{specimen_path}/{args.drr_params_csv_dir}/{specimen_id}_pose_params_{args.task_type}.csv'
        with open(pose_csv_path, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([
                'specimen_id', 'image_name', 'task_type',
                'rx', 'ry', 'rz',
                'tx', 'ty', 'tz'
            ])
            csvwriter.writerows(pose_records)
        
        for i in tqdm(range(sample_size), desc="Generating DRRs"):
            ct_volume = read(
                specimen_volume_path, 
                bone_attenuation_multiplier=torch.distributions.Uniform(1.0, 5.0).sample().item()
            )
            drr = DRR(
                ct_volume,
                sdd=sdd,
                height=height,
                width=width,
                delx=pixel_spacing[0],
                dely=pixel_spacing[1],
            ).to(device)

            rx, ry, rz = rotations_list[i].to(device)
            tx, ty, tz = translations_list[i].to(device)
            rotations = torch.tensor([[rx, ry, rz]], device=device)
            translations = torch.tensor([[tx, ty, tz]], device=device)

            img = drr(rotations, translations, parameterization="euler_angles", convention="ZXY")
            img_np = img.squeeze().detach().cpu().numpy()
            img_norm = (img_np - img_np.min()) / (img_np.max() - img_np.min())
            img_uint8 = (img_norm * 255).astype(np.uint8)
            cv2.imwrite(f'{specimen_path}/{args.drr_dir}_{args.task_type}/{specimen_id}_{i:04d}.png', img_uint8)
            del img, img_np, img_norm, img_uint8  # Free memory

            # break  # For testing, remove this line for full processing

        landmark_dir = os.path.join(specimen_path, 'gt_landmarks_3D')
        landmark_files = sorted(
            glob(f"{landmark_dir}/{specimen_id}_Landmark_*.nii.gz"),
            key=lambda x: int(re.search(r"_Landmark_(\d+)_", os.path.basename(x)).group(1))
        )
        
        for idx, landmark_file in enumerate(landmark_files):
            print(f"Processing landmark {idx + 1}/{len(landmark_files)}: {landmark_file}")
            
            landmark_volume = read(landmark_file)
            drr = DRR(
                landmark_volume,
                sdd=sdd,
                height=height,
                width=width,
                delx=pixel_spacing[0],
                dely=pixel_spacing[1],
            ).to(device)
            
            landmark_2D_array = np.zeros((sample_size, 2), dtype=np.float32)
            for i in tqdm(range(sample_size), desc="Generating DRRs for landmarks"):
                rx, ry, rz = rotations_list[i].to(device)
                tx, ty, tz = translations_list[i].to(device)

                rotations = torch.tensor([[rx, ry, rz]], device=device)
                translations = torch.tensor([[tx, ty, tz]], device=device)

                img = drr(rotations, translations, parameterization="euler_angles", convention="ZXY")
                img_np = img.squeeze().detach().cpu().numpy()
                if img_np.max() == img_np.min():
                    img_norm = np.zeros_like(img_np, dtype=np.float32)
                    img_uint8 = np.zeros_like(img_np, dtype=np.uint8)
                else:
                    img_norm = (img_np - img_np.min()) / (img_np.max() - img_np.min())
                    img_uint8 = (img_norm * 255).astype(np.uint8)

                nonzero_coords = np.column_stack(np.where(img_uint8 != 0))
                if nonzero_coords.size == 0:
                    # If no non-zero coordinates found
                    landmark_2D_array[i] = (np.nan, np.nan)
                else:
                    avg_yx = np.mean(nonzero_coords, axis=0).astype(int)
                    avg_y, avg_x = avg_yx
                    landmark_2D_array[i] = (avg_x, avg_y)

                del img, img_np, img_norm, img_uint8

                # break  # For testing, remove this line for full processing
            
            total_landmark_2D_array[idx, :, :] = np.array(landmark_2D_array)
        total_landmark_2D_array_transposed = total_landmark_2D_array.transpose(1, 0, 2)

        image_path_list = sorted(glob(f"{specimen_path}/{args.drr_dir}_{args.task_type}/{specimen_id}_*.png"))[:10]
        os.makedirs(f'visualizations/overlay/{args.task_type}/{specimen_id}', exist_ok=True)
        for i, image_path in tqdm(enumerate(image_path_list), desc="Overlaying Landmarks"):
            image = cv2.imread(image_path)
            landmarks = total_landmark_2D_array_transposed[i]
            for idx, (avg_x, avg_y) in enumerate(landmarks):
                if np.isnan(avg_x) or np.isnan(avg_y):
                    continue  # Skip NaN values
                cv2.circle(image, (int(avg_x), int(avg_y)), radius=5, color=(0, 255, 0), thickness=-1)
            cv2.imwrite(f'visualizations/overlay/{args.task_type}/{specimen_id}/{specimen_id}_{i:04d}.png', image)

        # Save the 2D landmark coordinates to CSV files
        for image_idx in range(total_landmark_2D_array_transposed.shape[0]):
            csv_file_path = f"{specimen_path}/{args.drr_csv_dir}_{args.task_type}/landmarks_{image_idx:04d}.csv"
            # print(csv_file_path)
            with open(csv_file_path, 'w') as f:
                f.write("x,y\n")
                for landmark_idx, (avg_x, avg_y) in enumerate(total_landmark_2D_array_transposed[image_idx]):
                    if np.isnan(avg_x) or np.isnan(avg_y):
                        # print(f"Warning: NaN value at landmark {landmark_idx} for image {image_idx}")
                        f.write("NaN,NaN\n")
                    else:
                        f.write(f"{avg_x},{avg_y}\n")

        # Save the 2D landmark coordinates to a single CSV file
        all_landmarks_csv_path = f"{specimen_path}/{args.drr_csv_dir}_{args.task_type}/all_landmarks.csv"
        with open(all_landmarks_csv_path, 'w') as f:
            f.write("image_index,x,y\n")
            for i in range(total_landmark_2D_array_transposed.shape[0]):
                for avg_x, avg_y in total_landmark_2D_array_transposed[i]:
                    if np.isnan(avg_x) or np.isnan(avg_y):
                        f.write(f"{i},NaN,NaN\n")
                    else:
                        f.write(f"{i},{avg_x},{avg_y}\n")

        # Save the landmark 2D coordinates to a .npy file
        landmark_2d_npy_path = f"{specimen_path}/drr_landmarks_2D.npy"
        np.save(landmark_2d_npy_path, total_landmark_2D_array_transposed)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Projection")

    parser.add_argument('--data_dir', type=str, help='Name of the directory where all the data lives in', default='data')
    parser.add_argument('--unzip_dir', type=str, help='Name of the directory where we save the decompressed version', default='DeepFluoro')

    parser.add_argument('--sdd', type=float, default=1020.0, help='Source to Detector Distance')
    parser.add_argument('--height', type=int, default=512, help='Image height')
    parser.add_argument('--width', type=int, default=512, help='Image width')
    parser.add_argument('--sample_size', type=int, default=600, help='Number of samples')
    parser.add_argument('--n_landmarks', type=int, default=14, help='Number of landmarks')

    parser.add_argument('--task_type', type=str, default='easy', choices=['easy', 'medium', 'hard'], help="Task type to process")
    parser.add_argument('--seed_value', type=int, default=42, help="Fix seed for reproducibility")

    parser.add_argument('--drr_dir', type=str, default='drr_projections', help='Directory name to save the DRR projections')
    parser.add_argument('--drr_csv_dir', type=str, default='drr_projections_csv', help='Directory name to save the DRR projections CSV files')
    parser.add_argument('--drr_params_csv_dir', type=str, default='drr_projections_csv_params', help='Directory name to save the DRR projections parameters CSV file')

    args = parser.parse_args()
    
    set_seed(args.seed_value)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    project(args, device)