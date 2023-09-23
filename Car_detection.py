import cv2
import numpy as np

# Load YOLO model and configuration
yolo_net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# Load image
image = cv2.imread("traffic3.jpg")

# Preprocess image
blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)

# Set the input to the YOLO model
yolo_net.setInput(blob)

# Perform object detection
layer_names = yolo_net.getUnconnectedOutLayersNames()
outputs = yolo_net.forward(layer_names)

# Initialize lists to store class IDs, confidence scores, and bounding boxes
class_ids = []
confidences = []
boxes = []
vechiles=["car","motorbike","bus","truck"]
# Filter detections for vechiles
for output in outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.4 and classes[class_id] in vechiles:
            center_x, center_y, width, height = map(int, detection[0:4] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]]))
            x, y = int(center_x - width / 2), int(center_y - height / 2)
            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([x, y, int(width), int(height)])

# Apply non-maximum suppression to remove overlapping bounding boxes
indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.4)

# Draw bounding boxes on the image
if len(indices) > 0:
    for i in indices.flatten():
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = confidences[i]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display or save the result
#cv2.imshow("Car Detection", image)
cv2.imwrite("result.jpg",image)
cv2.waitKey(0)
cv2.destroyAllWindows()