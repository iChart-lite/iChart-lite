# iChart-lite

iChart-lite is a real-time assistive system that automatically extracts data from bar charts and generates textual and auditorial descriptions. It utilizes real-time object detection and heuristics, enabling it to be lightweight enough to run on portable devices.

## Demo Video

[![iChart Demo Video Thumbnail](https://img.youtube.com/vi/jPnuxwI-Nys/0.jpg)](https://youtu.be/jPnuxwI-Nys)

## Running Demo Yourself

This is tested on a Windows machine. While training should work on other machines, the demo only works on Windows.

### Setting Environment

1. Install [Python 3.11](https://www.python.org/downloads/).

2. Install [Poetry](https://python-poetry.org/docs/#installation).

3. Run this command to install dependencies.

  ```console
  poetry update
  ```

### Training

Run this command to train a YOLOv8 model.

```console
poetry run python train.py
```

If you encounter an error message saying that the dataset images are not found, open `~\AppData\Roaming\Ultralytics\settings.yaml` and change `datasets_dir` to `null`.

After training, the model will be saved to `run/train/weights/best.pt`. You can copy this file to `models` directory to use it in demo.

### Running Demo

Run this command to run the demo.

```console
poetry run python demo.py
```

You can change the YOLOv8 model used in demo by changing `MODEL_PATH` in `demo.py`.
