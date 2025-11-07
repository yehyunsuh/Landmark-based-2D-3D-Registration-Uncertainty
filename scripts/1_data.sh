python3 -m src.data.1_extract_content
CUDA_VISIBLE_DEVICES=0 python3 -m src.data.2_project --task_type easy
CUDA_VISIBLE_DEVICES=0 python3 -m src.data.2_project --task_type medium
CUDA_VISIBLE_DEVICES=0 python3 -m src.data.2_project --task_type hard