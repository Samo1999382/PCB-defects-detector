from ultralytics import YOLO
import cv2

model_path = "resist/model/resistor_model.pt"
image_path = "/home/armo38/Pictures/Screenshots/Screenshot From 2025-12-04 22-35-43.png"

# Load model
model = YOLO(model_path)

# Run inference
results = model(image_path)

# Plot detections on image
img_with_boxes = results[0].plot()

# Show window
cv2.imshow("Detections", img_with_boxes)
cv2.waitKey(0)
cv2.destroyAllWindows()
