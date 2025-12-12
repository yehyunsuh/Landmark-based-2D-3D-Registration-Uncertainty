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