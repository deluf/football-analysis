# Football analysis

### Instructions for python's virtual enviroments

1. Create the virtual enviroment
```bash
python3 -m venv YOUR_ENV_NAME
source YOUR_ENV_NAME/bin/activate
```
2. Follow the instructions on https://pytorch.org/get-started/locally/
3. Install the required packages
```bash
pip install notebook opencv-python pyyaml matplotlib pandas numpy supervision ultralytics scikit-learn tqdm ipywidgets
```

---

### Instructions for conda enviroments (if you really want to go this way)

1. Create the virtual enviroment
```bash
conda create --name YOUR_ENV_NAME python=3.11 -y
conda activate YOUR_ENV_NAME
```
2. Follow the instructions on https://pytorch.org/get-started/locally/
3. Install the required packages
```bash
conda install -c conda-forge -y notebook opencv pyyaml matplotlib pandas numpy supervision ultralytics scikit-learn tqdm ipywidgets
```

> Note: Pytorch's conda package is no longer maintained. The reccomended way of downloading it is through pip. In order to not mix up pip and conda packages in the same enviroment, we recommend installing everything with pip (steps above). You can still use conda to create and manage the virtual enviroment, then use pip to download the required packages.
