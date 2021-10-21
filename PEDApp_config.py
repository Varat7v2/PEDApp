#!/usr/bin/env python3

FRAMEWORK = 'PYTORCH'	# Choose either 'TENSORFLOW' or 'PYTORCH'
MODEL_TYPE = 'frozenGraph'

### FROZEN GRAPH --> PEDAPP
FROZEN_GRAPH_PEDAPP = './models/ECD_MobileNetV2.pb'
COMPONENTS_LABELS = './myTensorFlow/labels/PEDApp_labels.pbtxt'
COMPONENTS_CLASSES = 9

# ### YOLO-V2 and YOLO-TINY MODEL PARAMETERS
# YOLO_CONFIDENCE = 0.5	#reject boxes with confidence < 0.5
# YOLO_THRESHOLD = 0.5	#to apply Non-Max Supression
# YOLO_WEIGHTS = './models/yolo_models/yolov3-416.weights'
# YOLO_CONFIG = './models/yolo_models/yolov3-416.cfg'
# YOLO_LABELS = './models/yolo_models/coco-labels'

# ### TFLITE MODEL PARAMETERS
# TFLITE_MODEL = './models/ssd_mobilenetv2_person_f32.tflite'
# TFLITE_THRESHOLD = 0.4

#CAMERA AND VIDEO PARAMETERS
IMG_FLAG = True
DEBUG_FLAG = False
WEBCAM_FLAG = False
VIDEO_FLAG = True
WRITE_VIDEO = True
myVIDEO_STREAM = False
CAMERA_ID = 0
VIDEO_INPUT = './videos/input.mov'
# IMAGES_PATH = './../DATASET-ONLINE-DOWNLOADER/data/data_v1'
VIDEO_OUTPUT = './outputs/PEDApp_{}.avi'.format(MODEL_TYPE)
IMG_PATH = 'images/test2.jpg'


# ### PYTORCH CONFIGURATIONS - 80 CLASSES OD
# OUTPUT='./yolor/inference/output'
# SOURCE='./yolor/inference/images/test2.jpg'
# WEIGHTS='./models/yolor_p6.pt'
# DEVICE='0' # CUDA DEVICE - 0,1,2,3 OR cpu
# IMG_SIZE=1280 # INPUT IMAGE SIZE (PIXELS)
# CONFIG_YOLOR='./yolor/cfg/yolor_p6.cfg' # MODEL CONFIG FILE
# NAMES='./yolor/data/coco.names' # CLASSES NAME LISTS
# IOU_THRESHOLD=0.5 # IOU THRESHOLD FOR NMS
# CONFIDENCE_THRESHOLD=0.4 # OBJECT CONFIDENCE THRESHOLD


### PYTORCH CONFIGURATIONS - PROJECT-ECD
OUTPUT='./outputs'
SOURCE='./images/test2.jpg'
WEIGHTS='./models/ECD_YOLOR_PYTORCH.pt'
DEVICE='0' # CUDA DEVICE - 0,1,2,3 OR cpu
IMG_SIZE=1280 # INPUT IMAGE SIZE (PIXELS)
CONFIG_YOLOR='./data/yolor_p6_ecd.cfg' # MODEL CONFIG FILE
NAMES='./data/ecd.names' # CLASSES NAME LISTS
IOU_THRESHOLD=0.5 # IOU THRESHOLD FOR NMS
CONFIDENCE_THRESHOLD=0.4 # OBJECT CONFIDENCE THRESHOLD