# Dependencies

import time
import os, sys
import cv2 as cv
import numpy as np

# MS COCO labels
labels = open('dataset/coco.names').read().strip().split('\n')

# Parameters
CONFIDENCE = 0.5
THRESHOLD_SCORE = 0.5
THRESHOLD_IOU = 0.5

# Model Configuration
CONFIG_PATH = 'pretrained/yolov3.cfg'
WEIGHT_PATH = 'pretrained/yolov3.weights'

# Random boundary box colors
COLORS = np.random.randint(0, 255, size=(len(labels),3), dtype='uint8')

# Loading YOLO using OpenCV
net = cv.dnn.readNetFromDarknet(CONFIG_PATH, WEIGHT_PATH)

# IO path
path_io = 'io/sample.jpg'
image = cv.imread(path_io)
file_name = os.path.basename(path_io)
filename, ext = file_name.split('.')

# Normalize, Scale and Reshape image
h, w = image.shape[:2]

# blob (model input)
blob = cv.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)

print('image.shape :', image.shape)
print('blob.shape  :', blob.shape)

# Make prediction
net.setInput(blob)

# get layers
ln = net.getLayerNames()
ln =[ln[i[0]-1] for i in net.getUnconnectedOutLayers()]

# get feed forward output
start = time.perf_counter()

layer_outputs = net.forward(ln)

time_taken = time.perf_counter() - start
print(f'Fwd propagation time : {time_taken:.2f}s')

# Iterate over output discarding results below CONFIDENCE
font_scale = 3
thickness = 3

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
        color = [int(c) for c in COLORS[class_ids[i]]]
        cv.rectangle(image, (x, y), (x+y, y+h), color=color, thickness=thickness)
        text = f'{labels[class_ids[i]]} : {confidences[i]:.2f}'

        # bbox overlay
        (text_width, text_height) = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, fontScale = font_scale, thickness=thickness)[0]
        
        text_offset_x = x
        text_offset_y = y - 5

        bbox_coords = ((text_offset_x, text_offset_y),(text_offset_x + text_width + 2, text_offset_y - text_height ))
        
        overlay = image.copy()
        cv.rectangle(overlay, bbox_coords[0], bbox_coords[1], color=color, thickness=cv.FILLED)

        # opacity
        image = cv.addWeighted(overlay, 0.6, image, 0.4, 0)

        # label info ( label : confidence %)
        cv.putText(image, text, (x,y-5), cv.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, color=(0,0,0), thickness=thickness)

cv.imwrite(filename + '_yolo3.' + ext, image)