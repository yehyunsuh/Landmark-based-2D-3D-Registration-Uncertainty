import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic._internal._generate_schema")

import os
import torch
import argparse

from src.train.utils import set_seed, str2bool, arg_as_list
from src.train.model import UNet

from src.train_patient_agnostic.log import initiate_wandb
from src.train_patient_agnostic.train import train


def landmark_prediction_train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if args.wandb:
        initiate_wandb(args)

    model = UNet(args, device)

    train(args, model, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script for anatomical landmark detection with U-Net.")

    # Reproducibility
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")

    # Data paths
    parser.add_argument("--data_dir", type=str, default="data/DeepFluoro", help="Directory containing training images")
    parser.add_argument("--csv_file", type=str, default="train_label.csv", help="CSV file containing training annotations")
    parser.add_argument("--model_weight_dir", type=str, default="model_weight", help="Directory to save model weights")
    parser.add_argument('--task_type', type=str, default='easy', choices=['easy', 'medium', 'hard'], help="Task type to process")

    # Image/label settings
    parser.add_argument("--image_resize", type=int, default=512, help="Target image size after resizing (must be divisible by 32)")
    parser.add_argument("--n_landmarks", type=int, default=100, help="Number of landmarks in total")
    parser.add_argument("--landmark_to_predict", type=int, default=0, help="Landmark index to predict")
    parser.add_argument("--invisible_landmarks", type=str2bool, default=True, choices=[True, False], help="Whether there are invisible landmarks in the dataset")

    # Model parameters
    parser.add_argument("--encoder_depth", type=int, default=5, help="Depth of the encoder in the U-Net model")
    parser.add_argument("--decoder_channels", type=arg_as_list, default="[256, 128, 64, 32, 16]", help="List of channels in the decoder of the U-Net model")

    # Training parameters
    parser.add_argument("--preprocess", action="store_true", help="Run preprocessing before training")
    parser.add_argument("--batch_size", type=int, default=18, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=350, help="Number of training epochs")
    parser.add_argument("--dilation_iters", type=int, default=65, help="Number of iterations for binary dilation")
    parser.add_argument("--erosion_freq", type=int, default=50, help="Apply erosion every N epochs")
    parser.add_argument("--erosion_iters", type=int, default=10, help="Number of iterations for binary erosion")

    # Visualization options
    parser.add_argument("--vis_dir", type=str, default="visualization_tmp", help="Directory to save visualization results")
    parser.add_argument("--result_dir", type=str, default="result_tmp", help="Directory to save training results")

    # Wandb parameters
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="Landmark based Registration with Uncertainty", help="Weights & Biases project name")
    parser.add_argument("--wandb_entity", type=str, default="your_entity", help="Weights & Biases entity name")
    parser.add_argument("--wandb_name", type=str, default="baseline", help="Weights & Biases run name")

    args = parser.parse_args()

    # Fix randomness
    set_seed(args.seed)

    # Create necessary directories
    os.makedirs(f"{args.result_dir}/{args.wandb_name}/visualization", exist_ok=True)
    os.makedirs(f"{args.result_dir}/{args.wandb_name}/graph", exist_ok=True)
    os.makedirs(f"{args.model_weight_dir}", exist_ok=True)
    os.makedirs(f"{args.result_dir}/{args.wandb_name}/train_results", exist_ok=True)

    landmark_prediction_train(args)