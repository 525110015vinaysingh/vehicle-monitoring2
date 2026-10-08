from ultralytics import YOLO

model = YOLO("yolov8n.pt")
results = model("https://ultralytics.com/images/bus.jpg")
results[0].save("result.jpg")

for box in results[0].boxes:
    print(model.names[int(box.cls)], round(float(box.conf), 2))