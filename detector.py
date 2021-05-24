import cv2 as cv
from utils import get_prediction


# Read Labels
with open('./dataset/labels/coco.names', "r", encoding="utf-8" ) as f:
    labels = f.read().strip().split("\n")


# Config Parameters
yolo_configs = "pretrained/yolov3.cfg"
yolo_weights = "pretrained/yolov3.weights"

# IO

# Sample video
input = 2

cuda = True
show_display = True

write_output = False
out_video_path = "io/"

# Model Parameters
confidence_threshold = 0.5
overlapping_threshold = 0.3

net = cv.dnn.readNetFromDarknet(yolo_configs, yolo_weights)

if cuda: 
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)


if __name__ == '__main__':

    get_prediction.get_yolo_preds(
        net,
        input,
        out_video_path,
        confidence_threshold,
        overlapping_threshold,
        write_output,
        show_display,
        labels
    )
