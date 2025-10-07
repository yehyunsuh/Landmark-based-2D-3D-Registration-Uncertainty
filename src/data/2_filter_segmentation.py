import os
import argparse
import numpy as np
import nibabel as nib

from glob import glob
from tqdm import tqdm


def filter_segmentation(args):
    specimen_path_list = sorted(glob(f"{args.data_dir}/{args.unzip_dir}/*"))
    
    print(f"Found {len(specimen_path_list)} specimens to preprocess.")
    for specimen_path in tqdm(specimen_path_list):
        specimen_id = os.path.basename(specimen_path)

        specimen_ct_path = os.path.join(specimen_path, specimen_id + ".nii.gz")
        specimen_seg_path = os.path.join(specimen_path, specimen_id + "_seg.nii.gz")

        ct_image = nib.load(specimen_ct_path)
        ct_data = ct_image.get_fdata()
        ct_affine = ct_image.affine

        seg_image = nib.load(specimen_seg_path)
        seg_data = seg_image.get_fdata()
        seg_affine = seg_image.affine

        # Reference: https://github.com/rg2/DeepFluoroLabeling-IPCAI2020
        # 1: Left Hemipelvis
        # 2: Right Hemipelvis
        # 3: Vertebrae
        # 4: Upper Sacrum
        # 5: Left Femur
        # 6: Right Femur
        # 7: Lower Sacrum
        label_filter = [1, 2, 4, 7]  # Left Hemipelvis, Right Hemipelvis, Upper Sacrum, Lower Sacrum
        seg_filtered = np.full(seg_data.shape, 0, dtype=seg_data.dtype)
        for label in label_filter:
            seg_filtered[seg_data == label] = label

        ct_masked = ct_data.copy()
        ct_masked[seg_filtered == 0] = -1000

        ct_masked_nifti = nib.Nifti1Image(ct_masked, affine=ct_affine)
        nib.save(ct_masked_nifti, os.path.join(specimen_path, f"{specimen_id}_filtered.nii.gz"))
        
        seg_filtered_nifti = nib.Nifti1Image(seg_filtered, affine=seg_affine)
        nib.save(seg_filtered_nifti, os.path.join(specimen_path, f"{specimen_id}_seg_filtered.nii.gz"))
    
    print("Filtering completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Preprocessing Script")

    parser.add_argument('--data_dir', type=str, help='Name of the directory where all the data lives in', default='data')
    parser.add_argument('--unzip_dir', type=str, help='Name of the directory where we save the decompressed version', default='DeepFluoro')

    args = parser.parse_args()

    filter_segmentation(args)