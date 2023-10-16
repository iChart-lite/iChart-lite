import os
import threading
import tkinter as tk
import time
from playsound import playsound

import cv2
from gtts import gTTS
from PIL import Image, ImageTk
from ultralytics import YOLO

from ichart_lite import *

# Change this to the model you want to use
MODEL_PATH = "./models/yolov8s-best.pt"

FRAMERATE = 30

image_orig = None
image_yolo = 0
result_and_image_yolo = 0

def read_description(generated_description):
    def job():
        print("start TTS generation")
        temp_audio_filename = "temp.mp3"
        gTTS(generated_description).save(temp_audio_filename)
        print("end TTS generation")
        playsound(temp_audio_filename, block=True)
        os.remove(temp_audio_filename)

    t = threading.Thread(target=job)
    t.daemon = True
    t.start()

class App(threading.Thread):
    def __init__(self, window, window_title):
        super().__init__()

        self.window = window
        self.window.title(window_title)
        self.window.geometry("1280x800+100+100")

        self._frame = None
        self.switch_frame(StartPage)

        self.window.mainloop()

    def switch_frame(self, FrameClass):
        frame = FrameClass(self.window, self)
        if self._frame is not None:
            self._frame.destroy()

        self._frame = frame
        self._frame.pack(expand=True, fill="both")

class StartPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)

        label1 = tk.Label(self, text="iChart-lite: Accessible Chart for the Visually Impaired\nusing Lightweight Extraction Algorithm", font="Helvetica 35 bold")
        label2 = tk.Label(self, text="Real-time Chart Description Demo", font="Helvetica 25")

        button = tk.Button(self, text="Start", command=lambda: controller.switch_frame(MainPage))

        label1.pack(pady=(300, 0))
        label2.pack()
        button.pack()

class MainPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)

        self.delay = 20

        self.current_task = tk.Label(self, justify=tk.LEFT, anchor="w", text="Task: STEP 1 - Object Detection", font=("Helvetica", 40), bg="blue", fg="white")
        self.label1 = tk.Label(self, text="Original Image", font=("Helvetica", 30))
        self.label2 = tk.Label(self, text="YOLOv8 Detection", font=("Helvetica", 30))
        self.canvas1 = tk.Canvas(self, width=640, height=480)
        self.canvas2 = tk.Canvas(self, width=640, height=480)
        self.button = tk.Button(self, text="Generate Description", command=self.generate_description1)
        self.description = tk.Label(self, text="", font=("Helvetica", 30))
        # self.current_task = tk.Label(self, text="")

        self.current_task.grid(row=0, column=0, columnspan=2, sticky=tk.W)
        self.label1.grid(row=1, column=0)
        self.label2.grid(row=1, column=1)
        self.canvas1.grid(row=2, column=0)
        self.canvas2.grid(row=2, column=1)
        self.button.grid(row=3, column=0)
        self.description.grid(row=3, column=1)

        t1 = threading.Thread(target=capture)
        t1.daemon = True
        t1.start()

        self.update()

    def update(self):
        try:
            if image_orig is not None:
                image1 = cv2.resize(image_orig, (640, 480))
                image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
                self.photo1 = ImageTk.PhotoImage(image=Image.fromarray(image1))
                self.canvas1.create_image(0, 0, image=self.photo1, anchor=tk.NW)

            image2 = cv2.cvtColor(image_yolo, cv2.COLOR_BGR2RGB)
            image2 = cv2.resize(image2, (640, 480))
            self.photo2 = ImageTk.PhotoImage(image=Image.fromarray(image2))
            self.canvas2.create_image(0, 0, image=self.photo2, anchor=tk.NW)
        except:
            pass

        self.after(self.delay, self.update)

    def generate_description1(self):
        self.current_task.config(text="Task: STEP 2 - Extracting Labels and Bar Heights")
        self.description.config(text="Generating description...")
        self.after(10, self.generate_description2)

    def generate_description2(self):
        result, image = result_and_image_yolo
        data = extract_data(result, image)
        if data is not None:
            x_labels, bar_values, prefix, suffix, yb_value, yt_value = data
            self.generated_description = create_description(x_labels, bar_values, prefix, suffix)

            self.current_task.config(text="Task: STEP 3 - Synthesizing Speech")
            self.description.config(text=self.generated_description)
            read_description(self.generated_description)
        else:
            self.description.config(text="Extraction failed")

def capture():
    global image_orig

    video = cv2.VideoCapture("sample.mp4")
    count_video_frames = int(video.get(7))
    print(count_video_frames)

    i = 0
    while True:
        ret, image = video.read()
        if ret == False:
            break

        image_orig = image
        time.sleep(1 / FRAMERATE)

        if count_video_frames != -1:
            i += 1
            print(f"Frame {i}/{count_video_frames}")

def predict():
    global image_yolo
    global result_and_image_yolo

    model = YOLO(f"./models/yolov8s.yaml").load(MODEL_PATH)
    names = ["bar", "x", "y"]
    for i, name in enumerate(names):
        model.names[i] = name

    while True:
        if image_orig is None:
            continue

        image = image_orig.copy()
        results = model.predict(image, conf=0.3, imgsz=640, verbose=False, stream=True)

        for result in results:
            annotated_image = result.plot()

            image_yolo = annotated_image
            result_and_image_yolo = (result, image)

        time.sleep(0.3)

if __name__ == "__main__":
    t1 = threading.Thread(target=predict)
    t1.daemon = True
    t1.start()

    t2 = App(tk.Tk(), "Demo")
    t2.daemon = True
    t2.start()
