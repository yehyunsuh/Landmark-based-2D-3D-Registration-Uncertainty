import os
import cv2
import json
import h5py
import argparse
import numpy as np
import nibabel as nib


def process_patient_group(patient_id,group, output_path):
    os.makedirs(output_path, exist_ok=True)

    # --- Volume ---
    if 'vol' in group and 'pixels' in group['vol']:
        volume = group['vol']['pixels'][()].astype(np.float32)
        dir_mat = group['vol']['dir-mat'][()]
        origin = group['vol']['origin'][()]
        spacing = group['vol']['spacing'][()]

        volume = np.swapaxes(volume, 0, 2)[::-1].copy()

        spacing = spacing.flatten()
        affine = dir_mat @ np.diag(spacing)
        affine = np.concatenate([affine, origin], axis=1)
        affine = np.vstack([affine, [0, 0, 0, 1]])

        volume_nifti = nib.Nifti1Image(volume, affine=affine)
        nib.save(volume_nifti, os.path.join(output_path, f"{patient_id}.nii.gz"))

    # --- Segmentation ---
    if 'vol-seg' in group and 'image' in group['vol-seg']:
        seg = group['vol-seg']['image']['pixels'][()].astype(np.float32)
        seg_dir_mat = group['vol-seg']['image']['dir-mat'][()]
        seg_origin = group['vol-seg']['image']['origin'][()]
        seg_spacing = group['vol-seg']['image']['spacing'][()]

        seg = np.swapaxes(seg, 0, 2)[::-1].copy()

        seg_spacing = seg_spacing.flatten()
        seg_affine = seg_dir_mat @ np.diag(seg_spacing)
        seg_affine = np.concatenate([seg_affine, seg_origin], axis=1)
        seg_affine = np.vstack([seg_affine, [0, 0, 0, 1]])
        
        seg_nifti = nib.Nifti1Image(seg, affine=seg_affine)
        nib.save(seg_nifti, os.path.join(output_path, f"{patient_id}_seg.nii.gz"))

    # --- Landmarks ---
    if 'vol-landmarks' in group:
        landmarks = {}
        for lm_name in group['vol-landmarks']:
            coord = group['vol-landmarks'][lm_name][()]
            landmarks[lm_name] = coord.tolist()
        
        with open(os.path.join(output_path, "gt_landmarks_3d.json"), 'w') as f_json:
            json.dump(landmarks, f_json, indent=2)
        
    # --- Projection Images ---
    if 'projections' in group:
        proj_dir = os.path.join(output_path, "gt_projections")
        landmark_dir = os.path.join(output_path, "gt_landmarks_2d")
        proj_with_landmarks_dir = os.path.join(output_path, "gt_projections_with_landmarks")
        
        os.makedirs(proj_dir, exist_ok=True)
        os.makedirs(landmark_dir, exist_ok=True)
        os.makedirs(proj_with_landmarks_dir, exist_ok=True)

        for proj_id in list(group['projections'].keys()):
            proj_path = group['projections'][proj_id]

            img = proj_path['image']['pixels'][()]
            img = img.astype(np.float32)
            img = (img - np.min(img)) / (np.max(img) - np.min(img))
            img = (img * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(proj_dir, f"{patient_id}_{proj_id}.png"), img)
            
            gt_landmarks = proj_path['gt-landmarks']
            gt_landmarks_dict = {}
            for landmark_name in list(gt_landmarks.keys()):
                coord = gt_landmarks[landmark_name][()]
                if coord[0][0] < 0 or coord[1][0] < 0 or coord[0][0] >= img.shape[1] or coord[1][0] >= img.shape[0]:
                    # print(f"Warning: Landmark {landmark_name} in projection {proj_id} of patient {patient_id} is out of image bounds.")
                    gt_landmarks_dict[landmark_name] = None
                else:
                    gt_landmarks_dict[landmark_name] = coord.tolist()
            
            with open(os.path.join(landmark_dir, f"{patient_id}_{proj_id}_landmarks_2d.json"), 'w') as f_json:
                json.dump(gt_landmarks_dict, f_json, indent=2)

            # Draw landmarks on image
            img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            for landmark_name, coord in gt_landmarks_dict.items():
                if coord is None:
                    continue
                x, y = int(coord[0][0]), int(coord[1][0])
                cv2.circle(img_color, (x, y), 10, (0, 255, 0), -1)
                # cv2.putText(img_color, landmark_name, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imwrite(os.path.join(proj_with_landmarks_dir, f"{patient_id}_{proj_id}.png"), img_color)


def esxtract_content(args):
    os.system(f'unzip "{args.data_dir}/{args.zip_file}" -d "{args.data_dir}"')

    with h5py.File(f'{args.data_dir}/{args.h5_file}', 'r') as f:
        for patient_id in f.keys():
            if patient_id == "proj-params":
                continue  # skip global projection params
            print(f'Processing {patient_id}')
            group = f[patient_id]
            patient_output_path = os.path.join(args.data_dir, args.unzip_dir, patient_id)
            process_patient_group(patient_id, group, patient_output_path)

    # os.remove(f'{args.data_dir}/{args.zip_file}')
    # os.remove(f'{args.data_dir}/{args.h5_file}')
    # os.remove(f'{args.data_dir}/LICENSE')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unzip files in a directory.")

    parser.add_argument('--data_dir', type=str, help='Name of the directory where all the data lives in', default='data')
    parser.add_argument('--zip_file', type=str, help='Zip file name', default='ipcai_2020_full_res_data.zip')
    parser.add_argument('--h5_file', type=str, help='HDF5 file name', default='ipcai_2020_full_res_data.h5')
    parser.add_argument('--unzip_dir', type=str, help='Name of the directory where we save the decompressed version', default='DeepFluoro')

    args = parser.parse_args()

    esxtract_content(args)