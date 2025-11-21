import os
import csv
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A

from glob import glob
from tqdm import tqdm
from scipy.ndimage import binary_dilation
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader, random_split


class SegmentationDataset(Dataset):
    def __init__(self, data_dir, specimen_id='17-1882', image_resize=512, n_landmarks=14, dilation_iters=65, invisible_landmarks=True, data_type='train', task_type='easy', model_type='patient_held_out'):
        self.data_dir = data_dir
        self.specimen_id = specimen_id
        self.image_resize = image_resize
        self.samples = []
        self.n_landmarks = n_landmarks
        self.dilation_iters = dilation_iters
        self.invisible_landmarks = invisible_landmarks
        self.data_type = data_type
        self.task_type = task_type
        self.model_type = model_type

        specimen_path_list = sorted(glob(f"{self.data_dir}/*"))
        for specimen_path in specimen_path_list:
            specimen_id = os.path.basename(specimen_path)

            # Train and validation data will be on all specimens except the held-out specimen
            if self.data_type == 'train'  or self.data_type == 'val':
                if specimen_id == self.specimen_id:
                    continue  # Skip the held-out specimen

                csv_path = f'{self.data_dir}/{specimen_id}/landmark_prediction_csv/{self.model_type}/{self.data_type}_label_{self.task_type}.csv'

                with open(csv_path, 'r') as f:
                    reader = csv.reader(f)
                    header = next(reader)  # Skip header
                    for row in reader:
                        specimen_id = row[0]
                        image_name = row[1]
                        coords = list(map(int, row[5:]))
                        assert len(coords) == 2 * n_landmarks, "Mismatch in number of landmark coordinates"
                        landmarks = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
                        self.samples.append((specimen_id, image_name, landmarks))

            # Test data will be on the held-out specimen
            elif self.data_type == 'test':
                if specimen_id == self.specimen_id:
                    csv_path = f'{self.data_dir}/{specimen_id}/landmark_prediction_csv/{self.model_type}/{self.data_type}_label_{self.task_type}.csv'

                    with open(csv_path, 'r') as f:
                        reader = csv.reader(f)
                        header = next(reader)  # Skip header
                        for row in reader:
                            specimen_id = row[0]
                            image_name = row[1]
                            coords = list(map(int, row[5:]))
                            assert len(coords) == 2 * n_landmarks, "Mismatch in number of landmark coordinates"
                            landmarks = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
                            self.samples.append((specimen_id, image_name, landmarks))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        specimen_id, image_name, landmarks = self.samples[idx]
        image_path = f'{self.data_dir}/{specimen_id}/drr_projections_{self.task_type}/{specimen_id}_{image_name}'

        # Load and convert image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply resizing and normalization
        transform = A.Compose([
                A.Resize(self.image_resize, self.image_resize),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
                A.InvertImg(p=1), 
                A.VerticalFlip(p=0.3),
                ToTensorV2()], 
            keypoint_params=A.KeypointParams(format='xy', remove_invisible=False),
        )

        transformed = transform(image=image, keypoints=landmarks)
        image_transformed = transformed['image']  # Tensor: [3, H, W]
        landmarks_transformed = transformed['keypoints']

        # Create mask with binary dilation for each landmark
        H, W = image_transformed.shape[1:]
        masks = np.zeros((self.n_landmarks, H, W), dtype=np.uint8)

        for k, (x, y) in enumerate(landmarks_transformed):
            x = int(round(x))
            y = int(round(y))

            if self.invisible_landmarks:
                # if the landmarks are nan
                if landmarks[k][0] == -1 or landmarks[k][1] == -1:
                    landmarks_transformed[k] = (0, 0)  # ← this line sets the new landmark to (0, 0)
                else:
                    if 0 <= y < H and 0 <= x < W and self.data_type == 'train':
                        masks[k, y, x] = 1
                        masks[k] = binary_dilation(masks[k], iterations=self.dilation_iters).astype(np.uint8)
            else:
                if 0 <= y < H and 0 <= x < W and self.data_type == 'train':
                    masks[k, y, x] = 1
                    masks[k] = binary_dilation(masks[k], iterations=self.dilation_iters).astype(np.uint8)
        
        if self.data_type == 'train' or self.data_type == 'val':
            mask = torch.from_numpy(masks).float()  # Shape: [n_landmarks, H, W]
            return image_transformed, mask, image_name, landmarks_transformed
        
        if self.data_type == 'test':
            current_image_name = f'{specimen_id}_{image_name}'

            pose_param_csv_path = f'{self.data_dir}/{self.specimen_id}/drr_projections_csv_params/{self.specimen_id}_pose_params_{self.task_type}.csv'
            pose_params = []
            with open(pose_param_csv_path, 'r') as f:
                reader = csv.reader(f)
                header = next(reader)  # Skip header
                for row in reader:
                    params_specimen_id = row[0]
                    params_image_name = row[1]
                    
                    if current_image_name == params_image_name:
                        params_task_type = row[2]
                        rotation_params = list(map(float, row[3:6]))  # rx, ry, rz
                        translation_params = list(map(float, row[6:9]))  # tx, ty, tz
                        pose_params = rotation_params + translation_params
                        break

            return image_transformed, self.specimen_id, image_name, landmarks_transformed, pose_params
    

def preprocessing(args):
    specimen_path_list = sorted(glob(f"{args.data_dir}/*"))
    for specimen_path in specimen_path_list:
        specimen_id = os.path.basename(specimen_path)
        if specimen_id == args.specimen_id:
            continue  # Skip the held-out specimen
        
        image_dir = f'{args.data_dir}/{specimen_id}/drr_projections_{args.task_type}'
        landmark_dir = f'{args.data_dir}/{specimen_id}/drr_projections_csv_{args.task_type}'

        # Get all landmark CSVs except the first one (assuming 'all_landmarks.csv' is first)
        landmark_data_path_list = sorted(glob(f'{landmark_dir}/*.csv'), key=lambda x: os.path.basename(x))[1:]

        rows = []
        for i, landmark_data_path in enumerate(landmark_data_path_list):
            df = pd.read_csv(landmark_data_path)

            if df.empty:
                print(f"Warning: {landmark_data_path} is empty.")
                continue

            # Image info
            image_name = f'{i:04d}.png'
            num_landmarks = len(df)

            image_path = f'{image_dir}/{specimen_id}_{image_name}'
            image = cv2.imread(image_path)
            image_width = image.shape[1]
            image_height = image.shape[0]

            # Flatten landmarks into [x0, y0, x1, y1, ..., xn, yn]
            df = df[['x', 'y']].fillna(-1)
            landmark_coords = df.astype(int).values.flatten().tolist()

            # Combine all into a row
            row = [specimen_id, image_name, image_width, image_height, num_landmarks] + landmark_coords
            rows.append(row)

        # Create column names
        column_names = ['Case ID', 'Image Name', 'Image Width', 'Image Height', 'Number of Landmarks']
        if rows:
            max_landmarks = (len(rows[0]) - 5) // 2
            for i in range(max_landmarks):
                column_names += [f'Landmark {i+1} x', f'Landmark {i+1} y']

        # Save to DataFrame
        output_df = pd.DataFrame(rows, columns=column_names)
        # output_df = output_df.replace(-1, pd.NA)  # Do not replace -1 with NaN
        # print(output_df.head())

        # Shuffle the DataFrame
        output_df = output_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

        # --- Split ---
        n = len(output_df)
        train_end = int(0.75 * n)
        val_end = int(0.9 * n)

        train_df = output_df.iloc[:train_end]
        val_df = output_df.iloc[train_end:val_end]
        test_df = output_df.iloc[val_end:]

        os.makedirs(f'{args.data_dir}/{specimen_id}/landmark_prediction_csv/{args.model_type}', exist_ok=True)
        train_df.to_csv(f'{args.data_dir}/{specimen_id}/landmark_prediction_csv/{args.model_type}/train_label_{args.task_type}.csv', index=False)
        val_df.to_csv(f'{args.data_dir}/{specimen_id}/landmark_prediction_csv/{args.model_type}/val_label_{args.task_type}.csv', index=False)
        test_df.to_csv(f'{args.data_dir}/{specimen_id}/landmark_prediction_csv/{args.model_type}/test_label_{args.task_type}.csv', index=False)

        print(f"Preprocessed {specimen_id}, total {n} images → train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)} (Without {args.specimen_id})")


def dataloader(args, data_type='train', epoch=0):
    if args.preprocess and data_type == 'train' and epoch == 0:
        preprocessing(args)

    if data_type =="train":
        val_dataset = SegmentationDataset(
            data_dir=args.data_dir,
            specimen_id=args.specimen_id,
            image_resize=args.image_resize,
            n_landmarks=args.n_landmarks,
            dilation_iters=args.dilation_iters,
            invisible_landmarks=args.invisible_landmarks,
            data_type="val",
            task_type=args.task_type,
            model_type='patient_held_out',
        )

        train_dataset = SegmentationDataset(
            data_dir=args.data_dir,
            specimen_id=args.specimen_id,
            image_resize=args.image_resize,
            n_landmarks=args.n_landmarks,
            dilation_iters=args.dilation_iters,
            invisible_landmarks=args.invisible_landmarks,
            data_type="train",
            task_type=args.task_type,
            model_type='patient_held_out',
        )

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

        print(f"Train size: {len(train_loader.dataset)}")
        print(f"Validation size: {len(val_loader.dataset)}")

        return train_loader, val_loader
    
    elif data_type == "test":
        test_dataset = SegmentationDataset(
            data_dir=args.data_dir,
            specimen_id=args.specimen_id,
            image_resize=args.image_resize,
            n_landmarks=args.n_landmarks,
            dilation_iters=args.dilation_iters,
            invisible_landmarks=args.invisible_landmarks,
            data_type="test",
            task_type=args.task_type,
            model_type='patient_held_out',
        )

        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        print(f"Test size: {len(test_loader.dataset)}")

        return test_loader
    
    else:
        raise ValueError("data_type must be 'train' or 'test'")