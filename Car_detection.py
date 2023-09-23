import cv2
import numpy as np

# Specify the input video file path
input_path = "traffic_viedo3.mp4"  # Replace with the path to your video file

# Specify the output video file path
output_path = "output_video.mp4"  # Change the file extension to match your desired video format (e.g., .avi, .mp4)
# Load YOLO model and configuration
yolo_net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
yolo_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
yolo_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# Load image
#image = cv2.imread("traffic3.jpg")
layer_names = yolo_net.getUnconnectedOutLayersNames()

# Create VideoCapture object for input video
cap = cv2.VideoCapture(input_path)

# Get video frame properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_size = (frame_width, frame_height)

# Define the codec and create a VideoWriter object for output video
fourcc = cv2.VideoWriter_fourcc(*'mpv4')  # You can change the codec as needed
out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

#vechile_cnt={"car":0,"motorbike":0,"truck":0,"bus":0}

while True:
    ret, image = cap.read()  # Read a frame from the input video
    if not ret:
        print("failed")
        break  # Break the loop if no more frames are available
    vechile_cnt = {"car": 0, "motorbike": 0, "truck": 0, "bus": 0}
#Preprocess image
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    # Set the input to the YOLO model
    yolo_net.setInput(blob)

    # Perform object detection
    #layer_names = yolo_net.getUnconnectedOutLayersNames()
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
                center_x, center_y, width, height = map(int, detection[0:4] * np.array([frame_width, frame_height, frame_width, frame_height]))
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
            if label=="car":
                vechile_cnt["car"]=vechile_cnt["car"]+1
            elif label=="motorbike":
                vechile_cnt["motorbike"] = vechile_cnt["motorbike"] + 1
            elif label=="bus":
                vechile_cnt["bus"] = vechile_cnt["bus"] + 1
            elif label=="truck":
                vechile_cnt["truck"] = vechile_cnt["truck"] + 1
            confidence = confidences[i]
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    if not out.isOpened():
       print("Error: VideoWriter is not opened.")
       break
    print("hello")
    cv2.putText(image,f"cars: {vechile_cnt['car']},bikes: {vechile_cnt['motorbike']},tucks: {vechile_cnt['truck']},bus: {vechile_cnt['bus']}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),2)
    out.write(image)
    # cv2.imshow("Object Detection", image)
    #
    # # Check for the 'q' key to exit the loop
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

out.release()
cap.release()
cv2.destroyAllWindows()

# Display a message to confirm that the video has been saved
print(f"Video saved as {output_path}")

# Display or save the result
#cv2.imshow("Car Detection", image)
# cv2.imwrite("result.jpg",image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()