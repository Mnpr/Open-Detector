import cv2 as cv
import matplotlib.pyplot as plt

from torch_utils import *
from torch_darknet import Darknet

labels_file = 'dataset/coco.names'

# Set the NMS and IOU Thresholds
THRESHOLD_SCORE = 0.5
THRESHOLD_IOU = 0.4

# Model Configuration
CONFIG_PATH = 'pretrained/yolov3.cfg'
WEIGHT_PATH = 'pretrained/yolov3.weights'

model = Darknet(CONFIG_PATH)
model.load_weights(WEIGHT_PATH)
model.print_network()

class_names = load_class_names(labels_file)

original_image = cv.imread("io/sample.jpg")
original_image = cv.cvtColor(original_image, cv.COLOR_BGR2RGB)

image = cv.resize(original_image, (model.width, model.height))

# detect the objects
boxes = detect_objects( model, image, THRESHOLD_IOU, THRESHOLD_SCORE )

# plot the image with the bounding boxes and corresponding object class labels
plot_boxes(original_image, boxes, class_names, plot_labels=True)