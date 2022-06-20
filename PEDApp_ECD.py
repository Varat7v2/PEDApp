import sys, os
import argparse
import time
from datetime import datetime
import cv2
import  numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor

from hough_line_transform_v5 import NODE_DETECTION
from myPYTORCH_INFERENCE import PYTORCH_INFERENCE
from myPySPICE_v2 import myPYSPICE
import config as myconfig

if __name__ == "__main__":

    ### PARSER INPUT ARGUMENTS - NOW ONLY FOR PYTORCH; LATER CAN BE MADE FLEXIBLE FOR OTHER MODELS
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=myconfig.WEIGHTS, help='model.pt path(s)')
    parser.add_argument('--source', type=str, default=myconfig.SOURCE, help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default=myconfig.OUTPUT, help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=myconfig.IMG_SIZE, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=myconfig.CONFIDENCE_THRESHOLD, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=myconfig.IOU_THRESHOLD, help='IOU threshold for NMS')
    parser.add_argument('--device', default=myconfig.DEVICE, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--cfg', type=str, default=myconfig.CONFIG_YOLOR, help='*.cfg path')
    parser.add_argument('--names', type=str, default=myconfig.NAMES, help='*.cfg path')
    opt = parser.parse_args()

    # tf_detector = FROZEN_GRAPH_INFERENCE(myconfig.FROZEN_GRAPH_PEDAPP)
    pt_detector = PYTORCH_INFERENCE(parser)
    node_det = NODE_DETECTION()
    pyspice = myPYSPICE(myconfig.PROJECT_NAME)

    if myconfig.IMG_FLAG:
        frame = cv2.imread(myconfig.SOURCE)
        frameDebug = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        im_height, im_width, im_channel = frame.shape

        ### COMPONENTS DETECTION
        if myconfig.FRAMEWORK == 'TENSORFLOW':
            ### TENSORFLOW FRAMEWORK
            pass
            # _, components = tf_detector.run_frozen_graph(frame, im_width, im_height)
        elif myconfig.FRAMEWORK == 'PYTORCH':
            ### PYTORCH FRAMEWORK
            with torch.no_grad():
                components = pt_detector.detect()

        for component in components:
            cv2.drawMarker(frame, component['center'], (255, 0, 255), markerType=cv2.MARKER_CROSS,
                           markerSize=20, thickness=2, line_type=cv2.LINE_AA)
            cv2.rectangle(frame, (component['left'], component['top']),
                (component['right'], component['bottom']), (0, 255, 0), 2, 8)
            cv2.putText(frame, component['label'], (component['right']-20, component['top']-5),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow('Source', frame)
        cv2.waitKey(0)
