import time
import cv2 as cv
import numpy as np

# Parameters
CONFIDENCE = 0.5
THRESHOLD_SCORE = 0.5
THRESHOLD_IOU = 0.5

# Model Configuration
CONFIG_PATH = 'pretrained/yolov4.cfg'
WEIGHT_PATH = 'pretrained/yolov4.weights'

# BBOX , Class Overlay Info
FONT_SCALE = 3
THICKNESS = 3

# Labels name and color
LABELS = open("../Open-Detector/dataset/coco.names").read().strip().split("\n")
COLOR = np.random.randint(0, 255, size=(len(LABELS),3), dtype='uint8')


# Loading YOLO using OpenCV
net = cv.dnn.readNetFromDarknet(CONFIG_PATH, WEIGHT_PATH)

# get layers
ln = net.getLayerNames()
ln =[ln[i[0]-1] for i in net.getUnconnectedOutLayers()]

cap = cv.VideoCapture(2)

while True:
    _, image = cap.read()
    h, w = image.shape[:2]

    # Normalize, Scale and Reshape image to blob darknet input
    blob = cv.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # get feed forward output time
    start = time.perf_counter()
    layer_outputs = net.forward(ln)
    time_taken = time.perf_counter() - start
    print(f'Fwd propagation time : {time_taken:.2f}s')

    # prediction accumulators
    boxes, confidences, class_ids = [],[],[]

    for output in layer_outputs:

        for detection in output:

            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # discarding weak predictions
            if confidence > CONFIDENCE:
                
                # b-box scaled to image size
                box = detection[:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype('int')

                # top-left co-ordinates derived from center
                x = int(centerX - (width / 2))
                y = int(centerY -(height / 2))

                # b-box co-ordinates, confidence and classes to accumulators
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)


    # Non-max supression
    indices = cv.dnn.NMSBoxes(boxes, confidences, THRESHOLD_SCORE, THRESHOLD_IOU )

    if len(indices) > 0:

        for i in indices.flatten():

            # boundary boxes
            x, y = boxes[i][0], boxes[i][1]
            w, h = boxes[i][2], boxes[i][3]

            # draw bounding boxes
            color = [int(c) for c in COLOR[class_ids[i]]]
            cv.rectangle(image, (x, y), (x+y, y+h), color=color, thickness=THICKNESS)
            text = f'{LABELS[class_ids[i]]} : {confidences[i]:.2f}'

            # bbox overlay
            (text_width, text_height) = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, fontScale = FONT_SCALE, thickness=THICKNESS)[0]
            
            text_offset_x = x
            text_offset_y = y - 5

            bbox_coords = ((text_offset_x, text_offset_y),(text_offset_x + text_width + 2, text_offset_y - text_height ))
            
            overlay = image.copy()
            cv.rectangle(overlay, bbox_coords[0], bbox_coords[1], color=color, thickness=cv.FILLED)

            # opacity
            image = cv.addWeighted(overlay, 0.6, image, 0.4, 0)

            # label info ( label : confidence %)
            cv.putText(image, text, (x,y-5), cv.FONT_HERSHEY_SIMPLEX, fontScale=FONT_SCALE, color=(0,0,0), thickness=THICKNESS)

    # show output
    cv.imshow('image', image)

    # press 'q' to break
    if ord('q') == cv.waitKey(1):
        break

# Release the capturing
cap.release()
cv.destroyAllWindows()