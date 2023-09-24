import cv2
from ultralytics import YOLO


model = YOLO("yolov8m.pt")


viedo="traffic_viedo3.mp4"
cap = cv2.VideoCapture(viedo)

# Get video frame properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_size = (frame_width, frame_height)

# Define the codec and create a VideoWriter object for output video
fourcc = cv2.VideoWriter_fourcc(*'mpv4')  # You can change the codec as needed
out = cv2.VideoWriter("output.mp4", fourcc, fps, frame_size)
results = model.predict(viedo, conf=0.1, iou=0.3, classes=[2, 3, 5, 7],stream=True)

for result in results:
    annote=result.plot()
    vechile_cnt = {"car": 0, "bike": 0, "truck": 0, "bus": 0}
    for i in range(len(result.boxes)):
        box=result.boxes[i]
        class_id=box.cls[0].item()
        if class_id==2:
            vechile_cnt["car"]=vechile_cnt["car"]+1
        elif class_id==3:
            vechile_cnt["bike"]=vechile_cnt["bike"]+1
        elif class_id==5:
            vechile_cnt["bus"]=vechile_cnt["bus"]+1
        if class_id==7:
            vechile_cnt["truck"]=vechile_cnt["truck"]+1
    cv2.putText(annote,f"cars: {vechile_cnt['car']},bikes: {vechile_cnt['bike']},tucks: {vechile_cnt['truck']},bus: {vechile_cnt['bus']}",(10,30),cv2.FONT_ITALIC,0.5,(255,0,0),2)
    out.write(annote)
