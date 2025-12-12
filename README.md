# Landmark-based-2D-3D-Registration-Uncertainty

- Environment Setup
```
conda create -n landmark python=3.10 -y
conda activate landmark
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip3 install opencv-python h5py numpy nibabel tqdm scipy diffdrr albumentationsx wandb segmentation_models_pytorch scikit-learn
```

- Data Setup
```
wget --no-check-certificate -O data/ipcai_2020_full_res_data.zip "http://archive.data.jhu.edu/api/access/datafile/:persistentId/?persistentId=doi:10.7281/T1/IFSXNV/EAN9GH"

python3 -m src.data.1_extract_content
CUDA_VISIBLE_DEVICES=0 python3 -m src.data.2_project
```

- Training Landmark Prediction Model  
Note that even though codes are available online and you can follow all the instructions which was the exact same code I use to obtain the results in the paper, due to the randomness in the augmentation process, the numbers in the final results might differ.
```
CUDA_VISIBLE_DEVICES=0 python3 -m src.train_patient_held_out.main --wandb --wandb_name 17-1882_hard --wandb_entity <your_wandb_entity> --specimen_id 17-1882 --preprocess
CUDA_VISIBLE_DEVICES=0 python3 -m src.train_patient_held_out.main --wandb --wandb_name 17-1905_hard --wandb_entity <your_wandb_entity> --specimen_id 17-1905 --preprocess
CUDA_VISIBLE_DEVICES=0 python3 -m src.train_patient_held_out.main --wandb --wandb_name 18-0725_hard --wandb_entity <your_wandb_entity> --specimen_id 18-0725 --preprocess
CUDA_VISIBLE_DEVICES=0 python3 -m src.train_patient_held_out.main --wandb --wandb_name 18-1109_hard --wandb_entity <your_wandb_entity> --specimen_id 18-1109 --preprocess
CUDA_VISIBLE_DEVICES=0 python3 -m src.train_patient_held_out.main --wandb --wandb_name 18-2799_hard --wandb_entity <your_wandb_entity> --specimen_id 18-2799 --preprocess
CUDA_VISIBLE_DEVICES=0 python3 -m src.train_patient_held_out.main --wandb --wandb_name 18-2800_hard --wandb_entity <your_wandb_entity> --specimen_id 18-2800 --preprocess
```

