# Landmark-based-2D-3D-Registration-Uncertainty

- Environment Setup
```
conda create -n landmark python=3.10 -y
conda activate landmark
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip3 install opencv-python h5py numpy nibabel tqdm scipy diffdrr 
```

- Data Setup
```
mkdir data
wget --no-check-certificate -O data/ipcai_2020_full_res_data.zip "http://archive.data.jhu.edu/api/access/datafile/:persistentId/?persistentId=doi:10.7281/T1/IFSXNV/EAN9GH"

CUDA_VISIBLE_DEVICES=7 python3 -m src.data.1_extract_content --sample_size 30
CUDA_VISIBLE_DEVICES=7 python3 -m src.data.2_filter_segmentation --sample_size 30
CUDA_VISIBLE_DEVICES=7 python3 -m src.data.3_sample_landmark --sample_size 30
CUDA_VISIBLE_DEVICES=7 python3 -m src.data.4_project --sample_size 30
```