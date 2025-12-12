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
python3 -m src.data.2_project
```

- Training Landmark Prediction Model  
Note that even though codes are available online and you can follow all the instructions which was the exact same code I use to obtain the results in the paper, due to the randomness in the augmentation process, the numbers in the final results might differ.
```
python3 -m src.train_patient_held_out.main --wandb --wandb_name 17-1882 --wandb_entity <your_wandb_entity> --specimen_id 17-1882 --preprocess --train_mode
python3 -m src.train_patient_held_out.main --wandb --wandb_name 17-1905 --wandb_entity <your_wandb_entity> --specimen_id 17-1905 --preprocess --train_mode
python3 -m src.train_patient_held_out.main --wandb --wandb_name 18-0725 --wandb_entity <your_wandb_entity> --specimen_id 18-0725 --preprocess --train_mode
python3 -m src.train_patient_held_out.main --wandb --wandb_name 18-1109 --wandb_entity <your_wandb_entity> --specimen_id 18-1109 --preprocess --train_mode
python3 -m src.train_patient_held_out.main --wandb --wandb_name 18-2799 --wandb_entity <your_wandb_entity> --specimen_id 18-2799 --preprocess --train_mode
python3 -m src.train_patient_held_out.main --wandb --wandb_name 18-2800 --wandb_entity <your_wandb_entity> --specimen_id 18-2800 --preprocess --train_mode
```

- Finetuning Landmark Prediction Model using Pose Estimation Error   
```
python3 -m src.train_patient_held_out.main --wandb --wandb_name 17-1882 --wandb_entity <your_wandb_entity> --specimen_id 17-1882 --batch_size 1 --n_simulations 40 --lr 1e-6 --finetune_mode
python3 -m src.train_patient_held_out.main --wandb --wandb_name 17-1905 --wandb_entity <your_wandb_entity> --specimen_id 17-1905 --batch_size 1 --n_simulations 40 --lr 1e-6 --finetune_mode
python3 -m src.train_patient_held_out.main --wandb --wandb_name 18-0725 --wandb_entity <your_wandb_entity> --specimen_id 18-0725 --batch_size 1 --n_simulations 40 --lr 1e-6 --finetune_mode
python3 -m src.train_patient_held_out.main --wandb --wandb_name 18-1109 --wandb_entity <your_wandb_entity> --specimen_id 18-1109 --batch_size 1 --n_simulations 40 --lr 1e-6 --finetune_mode
python3 -m src.train_patient_held_out.main --wandb --wandb_name 18-2799 --wandb_entity <your_wandb_entity> --specimen_id 18-2799 --batch_size 1 --n_simulations 40 --lr 1e-6 --finetune_mode
python3 -m src.train_patient_held_out.main --wandb --wandb_name 18-2800 --wandb_entity <your_wandb_entity> --specimen_id 18-2800 --batch_size 1 --n_simulations 40 --lr 1e-6 --finetune_mode
```

- Testing
Original Model
```
python3 -m src.train_patient_held_out.main --wandb_name 17-1882 --wandb_entity <your_wandb_entity> --specimen_id 17-1882 --dropout_rate 0.1 --top_k_landmarks 3 --test_prediction --test_mode 
python3 -m src.train_patient_held_out.main --wandb_name 17-1905 --wandb_entity <your_wandb_entity> --specimen_id 17-1905 --dropout_rate 0.1 --top_k_landmarks 3 --test_prediction --test_mode 
python3 -m src.train_patient_held_out.main --wandb_name 18-0725 --wandb_entity <your_wandb_entity> --specimen_id 18-0725 --dropout_rate 0.1 --top_k_landmarks 3 --test_prediction --test_mode 
python3 -m src.train_patient_held_out.main --wandb_name 18-1109 --wandb_entity <your_wandb_entity> --specimen_id 18-1109 --dropout_rate 0.1 --top_k_landmarks 3 --test_prediction --test_mode 
python3 -m src.train_patient_held_out.main --wandb_name 18-2799 --wandb_entity <your_wandb_entity> --specimen_id 18-2799 --dropout_rate 0.1 --top_k_landmarks 3 --test_prediction --test_mode 
python3 -m src.train_patient_held_out.main --wandb_name 18-2800 --wandb_entity <your_wandb_entity> --specimen_id 18-2800 --dropout_rate 0.1 --top_k_landmarks 3 --test_prediction --test_mode 
```

Finetuned Model
```
python3 -m src.train_patient_held_out.main --model_weight_name finetune --wandb_name 17-1882 --wandb_entity <your_wandb_entity> --specimen_id 17-1882 --dropout_rate 0.1 --top_k_landmarks 3 --test_prediction --test_mode
python3 -m src.train_patient_held_out.main --model_weight_name finetune --wandb_name 17-1905 --wandb_entity <your_wandb_entity> --specimen_id 17-1905 --dropout_rate 0.1 --top_k_landmarks 3 --test_prediction --test_mode
python3 -m src.train_patient_held_out.main --model_weight_name finetune --wandb_name 18-0725 --wandb_entity <your_wandb_entity> --specimen_id 18-0725 --dropout_rate 0.1 --top_k_landmarks 3 --test_prediction --test_mode
python3 -m src.train_patient_held_out.main --model_weight_name finetune --wandb_name 18-1109 --wandb_entity <your_wandb_entity> --specimen_id 18-1109 --dropout_rate 0.1 --top_k_landmarks 3 --test_prediction --test_mode
python3 -m src.train_patient_held_out.main --model_weight_name finetune --wandb_name 18-2799 --wandb_entity <your_wandb_entity> --specimen_id 18-2799 --dropout_rate 0.1 --top_k_landmarks 3 --test_prediction --test_mode
python3 -m src.train_patient_held_out.main --model_weight_name finetune --wandb_name 18-2800 --wandb_entity <your_wandb_entity> --specimen_id 18-2800 --dropout_rate 0.1 --top_k_landmarks 3 --test_prediction --test_mode
```