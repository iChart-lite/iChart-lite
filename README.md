# iChart-lite: Accessible Chart for the Visually Impaired using Lightweight Extraction Algorithm

iChart-lite is a real-time assistive system that automatically extracts data from bar charts and generates textual and auditorial descriptions. It utilizes real-time object detection and heuristics, enabling it to be lightweight enough to run on portable devices.

## Demo Video

Click the image below to watch the demo video.

[![iChart Demo Video Thumbnail](https://img.youtube.com/vi/91JDoD8HJxE/0.jpg)](https://youtu.be/91JDoD8HJxE)

## Running Demo Yourself

This is tested on a Windows machine. While training should work on other machines, the demo only works on Windows.

### Setting Environment

1. Install [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) for Windows. Make sure to install it for all users.

2. Install [Python 3.11](https://www.python.org/downloads/).

3. Install [Poetry](https://python-poetry.org/docs/#installation) Python package manager.

    Run this command to install dependencies.

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

You can change the YOLOv8 model used in the demo by changing `MODEL_PATH` in `demo.py`.
