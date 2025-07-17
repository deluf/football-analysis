import subprocess
import argparse

NUM_FOLDS = 5

parser = argparse.ArgumentParser(description="Train a YOLO model using k-fold validation.")
parser.add_argument("--model-size", type=str, required=True, choices=["n", "s", "m", "l", "x"], help="YOLO model size")
parser.add_argument("--batch-size", type=int, required=True, help="Batch size")
parser.add_argument("--dataset", type=str, required=True, choices=["original", "refined", "upscaled"], help="Dataset name")
parser.add_argument("--starting-fold", type=int, default=1, choices=range(1, NUM_FOLDS + 1), help=f"Optional starting fold number")

args = parser.parse_args()

for fold in range(args.starting_fold, NUM_FOLDS + 1):
    print(f"⏳️ Starting training: model-size={args.model_size}, batch-size={args.batch_size}, dataset={args.dataset}, fold={fold}")
    try:
        subprocess.run(
            ["python3", "train-single-fold.py",
             "--model-size", args.model_size,
             "--batch-size", str(args.batch_size),
             "--dataset", args.dataset,
             "--fold", str(fold)],
            check=True
        )
        print(f"✅ Completed: model-size={args.model_size}, batch-size={args.batch_size}, dataset={args.dataset}, fold={fold}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: model-size={args.model_size}, batch-size={args.batch_size}, dataset={args.dataset}, fold={fold}")
        print(f"Error: {e}")

