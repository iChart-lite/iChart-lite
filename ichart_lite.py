import re
from typing import TypedDict

import cv2
import numpy as np
import tesserocr
from PIL import Image

PYTESSERACT_CONFIG = "--psm 6"
Y_LABEL_REGEX = re.compile(r"(\D*)(\d+(?:\.\d+)?)(\D*)")

tessdata_path = "C:\\Program Files\\Tesseract-OCR\\tessdata"
tesserocr_api = tesserocr.PyTessBaseAPI(path=tessdata_path, psm=6)

class Detection:
  xyxy: tuple[int, int, int, int] # top-left x, y and bottom-right x, y

  def __init__(self, box):
    self.xyxy = (
      int(box.xyxy[0][0]),
      int(box.xyxy[0][1]),
      int(box.xyxy[0][2]),
      int(box.xyxy[0][3])
    )

  def xl(self) -> int:
    return self.xyxy[0]

  def xr(self) -> int:
    return self.xyxy[2]

  def yb(self) -> int:
    return self.xyxy[3]

  def yt(self) -> int:
    return self.xyxy[1]

  def center(self) -> tuple[int, int]:
    return ((self.xl() + self.xr()) // 2, (self.yt() + self.yb()) // 2)

  def width(self) -> int:
    return self.xr() - self.xl()

  def height(self) -> int:
    return -(self.yt() - self.yb())

  def area(self) -> int:
    return self.width() * self.height()

class LabelDetection(Detection):
  label: str

  def __init__(self, box, orig_image):
    super().__init__(box)

    detection_image = orig_image[self.xyxy[1]:self.xyxy[3], self.xyxy[0]:self.xyxy[2]]
    detection_image = cv2.cvtColor(detection_image, cv2.COLOR_BGR2GRAY)

    detection_image = Image.fromarray(detection_image)
    tesserocr_api.SetImage(detection_image)
    label = tesserocr_api.GetUTF8Text()

    self.label = label    \
      .strip()            \
      .replace("\n", " ") \
      .replace(",", "")   \
      .replace(".", "")   \
      .replace("‘", "")

class YLabelDetection(LabelDetection):
  value: float
  prefix: str
  suffix: str

  def __init__(self, box, orig_image):
    super().__init__(box, orig_image)

    # print(self.label)

    # big O and small o to zero (0)
    self.label = self.label.replace("O", "0").replace("o", "0")

    regex_result = Y_LABEL_REGEX.search(self.label)
    if regex_result is None:
      self.value = -1
      self.prefix = "False"
      self.suffix = "False"
    else:
      self.value = float(regex_result.groups()[1])
      self.prefix = regex_result.groups()[0]
      self.suffix = regex_result.groups()[2]

class Detections(TypedDict):
  bar: list[Detection]
  x_label: list[LabelDetection]
  y_label: list[YLabelDetection]

def remove_outlier(l: list, key=lambda v: v, offset=7) -> list:
  q1 = np.quantile([key(v) for v in l], 0.25)
  q3 = np.quantile([key(v) for v in l], 0.75)
  iqr = q3 - q1

  return list(filter(
      lambda v: (key(v) <= q3 + 1.5*iqr + offset) and (key(v) >= q1 - 1.5*iqr - offset),
      l
  ))

def extract_data(result, image) -> tuple[list[str], list[float], str, str, float, float] | None:
  detections: Detections = { "bar": [], "x_label": [], "y_label": [] }
  MODEL_NAMES = ["bar", "x", "y"]

  for box in result.boxes:
    name = MODEL_NAMES[int(box.cls)]
    if name == "bar":
      detections["bar"].append(Detection(box))
    elif name == "x":
      detection = LabelDetection(box, image)
      if detection is not None:
        detections["x_label"].append(detection)
    elif name == "y":
      detection = YLabelDetection(box, image)
      if detection is not None:
        detections["y_label"].append(detection)

  detections["bar"] = remove_outlier(detections["bar"], lambda v: v.yb(), 10)
  detections["bar"].sort(key=lambda v: v.center()[0])

  detections["x_label"] = remove_outlier(detections["x_label"], lambda v: v.center()[1])
  detections["x_label"].sort(key=lambda v: v.center()[0])

  detections["y_label"] = list(filter(lambda v: v.prefix != "False", detections["y_label"]))
  detections["y_label"].sort(key=lambda v: v.center()[1])
  detections["y_label"].reverse()

  # print([v.yb() for v in detections["bar"]])
  # print([v.label for v in detections["x_label"]])

  # x_labels = [x.label for x in detections["x_label"]]

  # y_label이 2개 보다 적게 나오면
  if len(detections["y_label"]) < 2:
    return None

  # print([v.value for v in detections["y_label"]])
  y_bottom = detections["y_label"][0]
  y_top = detections["y_label"][-1]

  def get_bar_value(bar_detection):
    ratio = (y_top.value - y_bottom.value)/-(y_top.center()[1] - y_bottom.center()[1])

    return ratio*(bar_detection.height() - (-(y_bottom.center()[1] - bar_detection.yb()))) + y_bottom.value

  # bar_values = []
  # for bar, x in zip(detections["bar"], detections["x_label"]):
  #   bar_height = bar.height()
  #   bar_values.append(bar.height()/y_height*(y_top.value - y_bottom.value) + y_bottom.value)

  bar_count_estimate = max(len(detections["bar"]), len(detections["x_label"]))

  bar_values = []
  x_labels = []
  bar_i = 0
  x_label_i = 0
  while True:
    if len(bar_values) == len(x_labels) and len(bar_values) == bar_count_estimate:
      break

    if len(detections["bar"]) <= bar_i:
      bar_values.append(y_bottom.value) # placeholder

      x_labels.append(detections["x_label"][x_label_i].label)
      x_label_i += 1

      continue

    if len(detections["x_label"]) <= x_label_i:
      # bar_values.append(detections["bar"][bar_i].height()/y_height*(y_top.value - y_bottom.value) + y_bottom.value)
      bar_values.append(get_bar_value(detections["bar"][bar_i]))
      bar_i += 1

      x_labels.append("[Unknown]") # placeholder

      continue

    if (detections["x_label"][x_label_i].center()[0] > detections["bar"][bar_i].xl() and \
        detections["x_label"][x_label_i].center()[0] < detections["bar"][bar_i].xr()) or \
       (detections["bar"][bar_i].center()[0] > detections["x_label"][x_label_i].xl() and
        detections["bar"][bar_i].center()[0] < detections["x_label"][x_label_i].xr()):
      # add bar value
      # bar_values.append(detections["bar"][bar_i].height()/y_height*(y_top.value - y_bottom.value) + y_bottom.value)
      bar_values.append(get_bar_value(detections["bar"][bar_i]))
      bar_i += 1

      # add x label
      x_labels.append(detections["x_label"][x_label_i].label)
      x_label_i += 1
    else:
      # bar is missing
      if detections["x_label"][x_label_i].center()[0] < detections["bar"][bar_i].center()[0]:
        bar_values.append(y_bottom.value) # placeholder

        # add x label
        x_labels.append(detections["x_label"][x_label_i].label)
        x_label_i += 1
      # x_label is missing
      else:
        # add bar value
        # bar_values.append(detections["bar"][bar_i].height()/y_height*(y_top.value - y_bottom.value) + y_bottom.value)
        bar_values.append(get_bar_value(detections["bar"][bar_i]))
        bar_i += 1

        x_labels.append("[Unknown]") # placeholder

  prefix = max(y_bottom, y_top, key=lambda v: v.prefix).prefix
  suffix = max(y_bottom, y_top, key=lambda v: v.suffix).suffix

  return (x_labels, bar_values, prefix, suffix, y_bottom.value, y_top.value)

def create_description(x_labels: list[str], bar_values: list[float], prefix: str, suffix: str) -> str:
  s = ""

  for x_label, bar_value in zip(x_labels, bar_values):
    bar_value_str = f"{prefix}{bar_value:.2f}{suffix}"
    s += f"The value of {x_label} is {bar_value_str}.\n"

  return s
