import os
import cv2
import json
import h5py
import argparse
import numpy as np
import nibabel as nib


def process_patient_group(patient_id, group, output_path, sphere_radius=3):
    os.makedirs(output_path, exist_ok=True)

    # ======================================================
    # --- CT Volume Extraction ---
    # ======================================================
    if "vol" in group and "pixels" in group["vol"]:
        print(f"\n=== Processing specimen '{patient_id}' ===")
        print("Reading CT volume...")

        vol_pix = group["vol"]["pixels"][()].astype(np.float32)
        spacing = group["vol"]["spacing"][()].astype(float).flatten()
        dir_mat = group["vol"]["dir-mat"][()].astype(float)
        origin = group["vol"]["origin"][()].astype(float).flatten()

        # Build affine (index-to-physical)
        affine = np.eye(4)
        for r in range(3):
            for c in range(3):
                affine[r, c] = dir_mat[r, c] * spacing[c]
            affine[r, 3] = origin[r]

        # --- Apply same axis reordering ---
        original_shape = vol_pix.shape
        vol_pix = np.swapaxes(vol_pix, 0, 2)[::-1].copy()

        # Update affine to match flip along the first axis
        flip_mat = np.eye(4)
        flip_mat[0, 0] = -1
        flip_mat[0, 3] = (original_shape[2] - 1) * spacing[2]
        affine = affine @ flip_mat

        # Save CT as NIfTI
        ct_path = os.path.join(output_path, f"{patient_id}_CT.nii.gz")
        nib.save(nib.Nifti1Image(vol_pix, affine), ct_path)
        print(f"✅ Saved CT volume: {ct_path}")

    # ======================================================
    # --- Segmentation ---
    # ======================================================
    if "vol-seg" in group and "image" in group["vol-seg"]:
        seg = group["vol-seg"]["image"]["pixels"][()].astype(np.float32)
        seg_dir_mat = group["vol-seg"]["image"]["dir-mat"][()]
        seg_origin = group["vol-seg"]["image"]["origin"][()]
        seg_spacing = group["vol-seg"]["image"]["spacing"][()]

        seg = np.swapaxes(seg, 0, 2)[::-1].copy()

        seg_spacing = seg_spacing.flatten()
        seg_affine = seg_dir_mat @ np.diag(seg_spacing)
        seg_affine = np.concatenate([seg_affine, seg_origin], axis=1)
        seg_affine = np.vstack([seg_affine, [0, 0, 0, 1]])

        seg_nifti = nib.Nifti1Image(seg, affine=seg_affine)
        nib.save(seg_nifti, os.path.join(output_path, f"{patient_id}_Segmentation.nii.gz"))

    # ======================================================
    # --- 3D Landmarks Extraction ---
    # ======================================================
    if "vol-landmarks" in group:
        os.makedirs(os.path.join(output_path, "gt_landmarks_3D"), exist_ok=True)
        lands_g = group["vol-landmarks"]
        landmark_names = list(lands_g.keys())
        landmark_coords = np.array([lands_g[name][()] for name in landmark_names])

        # --- Save raw 3D coordinates as JSON ---
        landmarks_dict = {name: lands_g[name][()].tolist() for name in landmark_names}
        with open(os.path.join(output_path, "gt_landmarks_3D.json"), "w") as f_json:
            json.dump(landmarks_dict, f_json, indent=2)

        # Prepare empty label volume same shape as CT
        lm_vol = np.zeros_like(vol_pix, dtype=np.uint16)
        inv_affine = np.linalg.inv(affine)

        repositioned_landmarks = []
        for idx, pt in enumerate(landmark_coords, start=1):
            pt_h = np.append(pt, 1)
            voxel = inv_affine @ pt_h
            voxel = np.round(voxel[:3]).astype(int)
            x, y, z = voxel
            repositioned_landmarks.append([x, y, z])
            
            label_value = idx

            # Draw labeled sphere (and store for single files)
            mask = np.zeros_like(lm_vol, dtype=np.uint8)
            for i in range(-sphere_radius, sphere_radius + 1):
                for j in range(-sphere_radius, sphere_radius + 1):
                    for k in range(-sphere_radius, sphere_radius + 1):
                        if i**2 + j**2 + k**2 <= sphere_radius**2:
                            xi, yi, zi = x + i, y + j, z + k
                            if (
                                0 <= xi < lm_vol.shape[0]
                                and 0 <= yi < lm_vol.shape[1]
                                and 0 <= zi < lm_vol.shape[2]
                            ):
                                lm_vol[xi, yi, zi] = label_value
                                mask[xi, yi, zi] = 1  # for single landmark file

            # --- Save individual landmark NIfTI ---
            single_lm_path = os.path.join(output_path, 'gt_landmarks_3D', f"{patient_id}_Landmark_{idx:02d}_{landmark_names[idx-1]}.nii.gz")
            nib.save(nib.Nifti1Image(mask.astype(np.uint8), affine), single_lm_path)

        # --- Save combined labeled landmark map ---
        lm_path = os.path.join(output_path, f"{patient_id}_Landmarks_3D.nii.gz")
        nib.save(nib.Nifti1Image(lm_vol, affine), lm_path)
        print(f"✅ Saved Combined Landmark Volume: {lm_path}")

        # --- Save repositioned landmarks as npy ---
        repositioned_landmarks = np.array(repositioned_landmarks)
        npy_path = os.path.join(output_path, f"{patient_id}_Landmarks_3D.npy")
        np.save(npy_path, repositioned_landmarks)
        print(f"✅ Saved Repositioned Landmarks: {npy_path}")

        # --- Save repositioned landmarks as json ---
        repositioned_dict = {name: repositioned_landmarks[idx].tolist() for idx, name in enumerate(landmark_names)}
        json_path = os.path.join(output_path, f"{patient_id}_Landmarks_3D.json")
        with open(json_path, "w") as f_json:
            json.dump(repositioned_dict, f_json, indent=2)
        print(f"✅ Saved Repositioned Landmarks JSON: {json_path}")

        # --- Save label index-name mapping ---
        name_map_path = os.path.join(output_path, f"{patient_id}_Landmarks_3D.txt")
        with open(name_map_path, "w") as ftxt:
            for idx, name in enumerate(landmark_names, start=1):
                ftxt.write(f"{idx}\t{name}\n")
        print(f"✅ Saved Landmark Label Map: {name_map_path}")
        print(f"✅ Saved {len(landmark_names)} single-landmark NIfTI files.")

    # ======================================================
    # --- Projection Images ---
    # ======================================================
    if "projections" in group:
        proj_dir = os.path.join(output_path, "gt_projections")
        landmark_dir = os.path.join(output_path, "gt_landmarks_2D")
        proj_with_landmarks_dir = os.path.join(output_path, "gt_projections_with_landmarks")

        os.makedirs(proj_dir, exist_ok=True)
        os.makedirs(landmark_dir, exist_ok=True)
        os.makedirs(proj_with_landmarks_dir, exist_ok=True)

        for proj_id in list(group["projections"].keys()):
            proj_path = group["projections"][proj_id]

            img = proj_path["image"]["pixels"][()]
            img = img.astype(np.float32)
            img = (img - np.min(img)) / (np.max(img) - np.min(img))
            img = (img * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(proj_dir, f"{patient_id}_{proj_id}.png"), img)

            gt_landmarks = proj_path["gt-landmarks"]
            gt_landmarks_dict = {}
            for landmark_name in list(gt_landmarks.keys()):
                coord = gt_landmarks[landmark_name][()]
                if coord[0][0] < 0 or coord[1][0] < 0 or coord[0][0] >= img.shape[1] or coord[1][0] >= img.shape[0]:
                    gt_landmarks_dict[landmark_name] = None
                else:
                    gt_landmarks_dict[landmark_name] = coord.tolist()

            with open(os.path.join(landmark_dir, f"{patient_id}_{proj_id}_landmarks_2D.json"), "w") as f_json:
                json.dump(gt_landmarks_dict, f_json, indent=2)

            # Draw landmarks on image
            img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            for landmark_name, coord in gt_landmarks_dict.items():
                if coord is None:
                    continue
                x, y = int(coord[0][0]), int(coord[1][0])
                cv2.circle(img_color, (x, y), 10, (0, 255, 0), -1)
            cv2.imwrite(os.path.join(proj_with_landmarks_dir, f"{patient_id}_{proj_id}.png"), img_color)


def extract_content(args):
    if not os.path.exists(f'{args.data_dir}/{args.h5_file}'):
        os.system(f'unzip "{args.data_dir}/{args.zip_file}" -d "{args.data_dir}"')

    with h5py.File(f"{args.data_dir}/{args.h5_file}", "r") as f:
        for patient_id in f.keys():
            if patient_id == "proj-params":
                continue  # Skip projection params
            print(f"Processing {patient_id}")
            group = f[patient_id]
            patient_output_path = os.path.join(args.data_dir, args.unzip_dir, patient_id)
            process_patient_group(patient_id, group, patient_output_path)

    # if os.path.exists(f'{args.data_dir}/{args.zip_file}'):
    #     os.remove(f'{args.data_dir}/{args.zip_file}')
    # if os.path.exists(f'{args.data_dir}/{args.h5_file}'):
    #     os.remove(f'{args.data_dir}/{args.h5_file}')
    if os.path.exists(f'{args.data_dir}/LICENSE'):
        os.remove(f'{args.data_dir}/LICENSE')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract data from HDF5 to DeepFluoro folder structure")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument('--zip_file', type=str, help='Zip file name', default='ipcai_2020_full_res_data.zip')
    parser.add_argument("--h5_file", type=str, default="ipcai_2020_full_res_data.h5")
    parser.add_argument("--unzip_dir", type=str, default="DeepFluoro")

    args = parser.parse_args()
    extract_content(args)