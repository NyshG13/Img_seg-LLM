from ultralytics import YOLO 

# Load the pretrained YOLOv8 segmentation model
model = YOLO("yolov8s-seg.pt")  # You can also try yolov8m-seg.pt or yolov8n-seg.pt for speed

# Run segmentation on webcam (0 = default webcam)
model.predict(
    source=0,              # Use webcam
    show=True,             # Show the video in a popup window
    stream=False,          # Don't stream results frame-by-frame (set to True for frame access)
    conf=0.3,              # Confidence threshold for filtering weak detections
    save=False             # Don't save output unless needed
)
