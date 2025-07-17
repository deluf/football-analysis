import subprocess

def run_training(model_size, batch_size, dataset):
    try:
        subprocess.run(
            ["python3", "train-single-model.py", 
             "--model-size", model_size,
             "--batch-size", str(batch_size),
             "--dataset", dataset],
            check=True
        )
    except subprocess.CalledProcessError as e:
        pass # train-single-model.py already provides error handling

# Batch sizes are tuned for our hardware (RTX 2060 Super 8GB)
refined_configs = [
    ("n", 32),
    ("s", 24),
    ("m", 14),
    ("l", 10),
    ("x", 6)
]
upscaled_configs = [
    ("n", 11),
    ("s", 6),
    ("m", 3),
    ("l", 2),
    ("x", 1)
]

for model_size, batch_size in refined_configs:
    run_training(model_size, batch_size, "refined")

#for model_size, batch_size in upscaled_configs:
#    run_training(model_size, batch_size, "upscaled")

