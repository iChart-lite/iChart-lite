from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(f"./models/yolov8s.yaml")

    # Load pre-trained model and start training
    model = model.load("./models/yolov8s.pt")
    model.train(data=f"./data_82.yaml", batch=16, epochs=30, imgsz=640)
    # model.train(data=f"./data_92.yaml", batch=16, epochs=30, imgsz=640)
