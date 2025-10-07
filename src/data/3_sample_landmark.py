import os
import argparse
import numpy as np
import nibabel as nib
import scipy.ndimage as ndi

from tqdm import tqdm
from glob import glob


def uniform_random_sampling(coords, num_points):
    indices = np.random.choice(len(coords), size=num_points, replace=False)
    return coords[indices]


def farthest_point_sampling(coords, num_points):
    # Start from one random point
    # Currently, the seed is fixed for reproducibility
    selected = [np.random.randint(len(coords))]
    distances = np.full(len(coords), np.inf)

    for _ in tqdm(range(1, num_points)):
        # Update minimum distances to the selected set
        dist_to_last = np.linalg.norm(coords - coords[selected[-1]], axis=1)
        distances = np.minimum(distances, dist_to_last)

        # Select the point farthest from current set
        next_index = np.argmax(distances)
        selected.append(next_index)

    return coords[selected]


def save_landmark_volumes(selected_coords, vol_shape, affine, specimen_path, save_dir, specimen_id):
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize merged volume with background value 0
    merged_vol = np.full(vol_shape, 0, dtype=np.int16)
    
    for i, coord_voxel in tqdm(enumerate(selected_coords), total=len(selected_coords)):
        # round coord_voxel to nearest integer
        coord_voxel = np.round(coord_voxel).astype(int)
        
        # Initialize volume for individual landmark
        vol = np.full(vol_shape, 0, dtype=np.int16)
        vol[tuple(coord_voxel)] = i + 1

        # Apply dilation to make it look like a sphere
        struct_elem = ndi.generate_binary_structure(3, 1)  # 3x3x3 connectivity
        vol_dilated = ndi.binary_dilation(vol == i + 1, structure=struct_elem, iterations=4)
        vol[vol_dilated] = i + 1

        # Save individual landmark volume
        landmark_img = nib.Nifti1Image(vol, affine)
        save_path = f'{save_dir}/{specimen_id}_landmark_{i:03d}.nii.gz'
        nib.save(landmark_img, save_path)

        # Add to merged volume (note: this will overwrite overlaps with later landmark IDs)
        merged_vol[vol_dilated] = i + 1

    # Save merged landmark volume
    merged_img = nib.Nifti1Image(merged_vol, affine)
    merged_path = f'{specimen_path}/{specimen_id}_landmarks_merged.nii.gz'
    nib.save(merged_img, merged_path)


def sample_landmark(args):
    seed_value = args.seed_value
    n_landmarks = args.n_landmarks

    specimen_path_list = sorted(glob(f"{args.data_dir}/{args.unzip_dir}/*"))
    for specimen_path in specimen_path_list:
        print(f"Processing specimen: {specimen_path}")
        specimen_id = os.path.basename(specimen_path)

        ct_path = os.path.join(specimen_path, f"{specimen_id}_filtered.nii.gz")
        ct_image = nib.load(ct_path)
        ct_data = ct_image.get_fdata()

        seg_path = os.path.join(specimen_path, f"{specimen_id}_seg_filtered.nii.gz")
        seg_image = nib.load(seg_path)
        seg_data = seg_image.get_fdata()
        seg_bin = (seg_data > 0).astype(np.uint8)

        landmark_path = os.path.join(specimen_path, f"{specimen_id}_sampled_landmarks.npy")
        coordinates = np.array(np.nonzero(seg_data)).T  # get voxel coordinates of non-zero mask

        np.random.seed(seed_value)
        print(f'Sampling {n_landmarks} landmarks from {args.sampling_region} region using {args.sampling_strategy} strategy.')
        if args.sampling_strategy == 'fps':
            selected_coords = farthest_point_sampling(coordinates, n_landmarks)
        elif args.sampling_strategy == 'uniform':
            selected_coords = uniform_random_sampling(coordinates, n_landmarks)
        else:
            raise ValueError("Unknown sampling strategy")
        
        np.save(landmark_path, selected_coords)
        
        landmark_volume_path = os.path.join(specimen_path, 'sampled_landmarks_3d')
        save_landmark_volumes(
            selected_coords=selected_coords,
            vol_shape=ct_data.shape,
            affine=ct_image.affine,
            specimen_path=specimen_path,
            save_dir=landmark_volume_path,
            specimen_id=specimen_id
        )
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Landmark sampling script")

    parser.add_argument('--data_dir', type=str, help='Name of the directory where all the data lives in', default='data')
    parser.add_argument('--unzip_dir', type=str, help='Name of the directory where we save the decompressed version', default='DeepFluoro')
    
    parser.add_argument('--sampling_region', type=str, default='pelvis', choices=['pelvis', 'surface'], help="Region to sample landmarks from: 'pelvis' or 'surface'")
    parser.add_argument('--sampling_strategy', type=str, default='fps', choices=['fps', 'uniform'], help="Sampling strategy for landmark selection.")
    parser.add_argument('--seed_value', type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument('--n_landmarks', type=int, default=100, help="Number of landmarks to sample")
    
    args = parser.parse_args()

    sample_landmark(args)