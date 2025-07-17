from ultralytics import YOLO
import argparse
from pathlib import Path

EPOCHS = 100
NUM_FOLDS = 5
SEED = 999

parser = argparse.ArgumentParser(description="Train a YOLO model with the specified fold.")
parser.add_argument("--model-size", type=str, required=True, choices=["n", "s", "m", "l", "x"], help="YOLO model size")
parser.add_argument("--batch-size", type=int, required=True, help="Batch size")
parser.add_argument("--dataset", type=str, required=True, choices=["original", "refined", "upscaled"], help="Dataset name")
parser.add_argument("--fold", type=int, required=True, choices=range(0, NUM_FOLDS + 1), help=f"Fold number")

args = parser.parse_args()
parent_dir = Path.cwd().parent

dataset_name = f"data={args.dataset}-folds={NUM_FOLDS}-seed={SEED}" 
project_name = dataset_name + f"-yolo={args.model_size}-epochs={EPOCHS}"

if args.fold == 0:
    project_path = parent_dir / "results"
    dataset_config_path = parent_dir / "datasets" / dataset_name / "full" / "config.yaml"
else:
    project_path = parent_dir / "results" / project_name
    dataset_config_path = parent_dir / "datasets" / dataset_name / f"fold_{args.fold}" / "config.yaml"

model = YOLO(f"yolo11{args.model_size}.pt", task="detect")
model.train(
    data=dataset_config_path,
    project=project_path,
    name="final-model" if args.fold == 0 else f"fold_{args.fold}",
    device="cuda",
    epochs=EPOCHS,
    seed=SEED,
    batch=args.batch_size,
    imgsz=1280 if args.dataset == "upscaled" else 640,
    scale=0.2, # The default 0.5 (scale up to +/- 50%) is a little too much for our use case
    val=False if args.fold == 0 else True
)

