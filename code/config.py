import os
import sys
import argparse


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing complete dataset with subfolders",
    )
    parser.add_argument(
        "--name", type=str, help="Name of the project", default="project"
    )
    parser.add_argument(
        "--num_classes", type=int, help="Number of target classes", required=True
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        help="directory to save all the results",
        required=True,
    )
    parser.add_argument("--batch", type=int, help="batch size", default=8)
    parser.add_argument("--epochs", type=int, help="Number of epochs", default=100)
    parser.add_argument(
        "--workers",
        type=int,
        help="Number of workers to load data",
        default=8,
    )
    parser.add_argument(
        "--test_split", type=float, help="Test split fraction in dataset", default=0.1
    )
    parser.add_argument(
        "--val_split",
        type=float,
        help="Validation split fraction in dataset",
        default=0.2,
    )
    parser.add_argument("--imgsz", type=int, help="Input image size", default=256)
    parser.add_argument(
        "--average",
        type=str,
        help="Averaging technique (micro, macro, weighted)",
        default="micro",
    )
    parser.add_argument("--lr", type=float, help="Learning rate", default=1e-3)
    parser.add_argument(
        "--tune", type=bool, help="True to perform hyperparameter tuning", default=False
    )

    opt = parser.parse_args()

    return opt
