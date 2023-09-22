import cv2
import numpy as np

cap = cv2.VideoCapture("C:\\Users\\user\\Downloads\\video.mp4")  # change path
min_width_rect = 80
min_height_rect = 80
line = 550

# Init
algo = cv2.createBackgroundSubtractorMOG2()

# fxn to count vechiles
def cen_hand(x,y,w,h):
    x1=int(w/2)
    y1=int(w/2)
    cx=x+x1
    cy=y+y1
    return cx,cy

detect=[]
offset=6
counter=0

while True:
    ret, frame1 = cap.read()
    
    if not ret:
        break  # Break the loop if there are no more frames
    
    grey = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (3, 3), 5)
    img_sub = algo.apply(blur)
    dilat = cv2.dilate(img_sub, np.ones((5, 5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
    dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)

    counter_shape, h = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.line(frame1, (25, line), (1200, line), (255, 127, 0), 3)

    for (i, c) in enumerate(counter_shape):
        (x, y, w, h) = cv2.boundingRect(c)
        validate_counter = (w >= min_width_rect) and (h >= min_height_rect)
        if not validate_counter:
            continue
        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)

        center=cen_hand(x,y,w,h)
        detect.append(center)
        cv2.circle(frame1,center,4,(0,0,255),-1)

        for(x,y) in detect:
            if y<(line+offset) and y>(line-offset):
                counter+=1
            cv2.line(frame1, (25, line), (1200, line), (0, 127, 255), 3)
            detect.remove((x,y))
            print("Car counter is :- "+str(counter))

    cv2.putText(frame1,"Vechile Counter: "+str(counter),(450,70),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),5)

    cv2.imshow('video original', frame1)
    if cv2.waitKey(1) == 13:
        break

cv2.destroyAllWindows()
cap.release()
