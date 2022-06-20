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
WEBCAM_FLAG = False
VIDEO_FLAG = True
WRITE_VIDEO = True
myVIDEO_STREAM = False
DEBUG_PEDAPP = True
CAMERA_ID = 0
VIDEO_INPUT = './videos/input.mov'
# IMAGES_PATH = './../DATASET-ONLINE-DOWNLOADER/data/data_v1'
VIDEO_OUTPUT = './outputs/PEDApp_{}.avi'.format(MODEL_TYPE)
# IMG_PATH = 'images/test1.jpg'

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
SOURCE='./images/sample.jpg'
# SOURCE = '/home/varat/myPhD/PEDApp/power_converter_dataset/train/boost/b10.jpg'
WEIGHTS='./models/ECD_YOLOR_PYTORCH.pt'
DEVICE='0' # CUDA DEVICE - 0,1,2,3 OR cpu
IMG_SIZE=1280 # INPUT IMAGE SIZE (PIXELS)
CONFIG_YOLOR='./data/yolor_p6_ecd.cfg' # MODEL CONFIG FILE
NAMES='./data/ecd.names' # CLASSES NAME LISTS
IOU_THRESHOLD=0.5 # IOU THRESHOLD FOR NMS
CONFIDENCE_THRESHOLD=0.6 # OBJECT CONFIDENCE THRESHOLD

### GITHUB PUSH CONFIGURATION
IGNORE_FOLDERS = [
                      'FreeRouting_Example',
                      'models',
                      '__pycache__',
                      'yolor',
                      'git_push.sh',
                      'get_size.py',
                      'pyspice_examples.py',
                      'test_design.py',
                      'rawread.py',
                      'readFile.py',
                      'skidl_examples.py',
                      'skidl_examples_lib_sklib.py',
                      'skidl_lib_sklib.py',
                      'pyspice_examples.py',
                      'pcbnew_layout.py',
                      'images',
                      'plots',
                      'version_1',
                 ]

### LINE DETECTION PARAMETERS - HOUGH TRANSFORM
LINES_THERESHOLD = 120
DEBUG_FLAG = True   ## Draw HoughLine transformed lines
STRETCH_LINE = True
NODES_NUMBER = 8
HOUGH_TRANSFORM_TYPE = 'Standard'   # 1) Standard or 2) Probablistic

### PLOT Configurations
X_LIMIT = 100000

### ELECTRIC CIRCUIT PARAMETERS
PROJECT = 'UH_ECD'
ANALYSIS = 'TRANSIENT' # 1)OPERATING_POINT, 2)TRANSIENT, 3)AC_ANALYSIS, 4)DC_SWEEP
DATA = './data'
ESR_FLAG = False
ESR_Value = 0.001 # Ohms
LIB_PATH = 'libraries'
DIODE_TYPE = '1N4148'
MOSFET_TYPE = 'irf150'

SAVE_PLOT = True
PLOT_FLAG = False
