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

# from myTensorFlow.utils import label_map_util
# from myFROZEN_GRAPH import FROZEN_GRAPH_INFERENCE
from hough_line_transform_v5 import NODE_DETECTION
from myPYTORCH_INFERENCE import PYTORCH_INFERENCE
from myPySPICE_v2 import myPYSPICE

import config as myconfig

def remove_components(gray, component):
    mask = np.full((component['height'], component['width']), 255, dtype=np.uint8)
    gray[component['top']:component['bottom'], component['left']:component['right']] = mask
    return gray

def euclidean_dist(component, node):
    return np.sqrt((node[0]-component[0])**2 + (node[1]-component[1])**2)

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
            # print(component['label'])
            gray = remove_components(gray, component)
            # cv2.circle(frame, component['center'], 5, (255, 0, 255), -1)
            cv2.drawMarker(frame, component['center'], (255, 0, 255), markerType=cv2.MARKER_CROSS,
                           markerSize=20, thickness=2, line_type=cv2.LINE_AA)
            cv2.rectangle(frame, (component['left'], component['top']),
                (component['right'], component['bottom']), (0, 255, 0), 2, 8)
            cv2.putText(frame, component['label'], (component['right']-20, component['top']-5),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        ### NODE DETECTION
        horizontal_lines, vertical_lines = node_det.houghTransform(gray)
        frameDebug, nodes, node_yavg = node_det.findNodes(frameDebug, im_height, im_width, horizontal_lines, vertical_lines)
        print('Nodes =', nodes)
        cv2.line(frame, (0,node_yavg), (im_width,node_yavg), (0,255,255),2,cv2.LINE_AA)

        # node1, node2 = list(), list()
        # for node in nodes:
        #     node1.append(node) if node[1] < node_yavg else node2.append(node)
        #     cv2.circle(frame, (node[0], node[1]), 10, (0, 255, 0), -1)
        # node1.sort(key=lambda x:x[0])
        # node2.sort(key=lambda x:x[0])
        
        # node_dict = dict()
        # for itr, node in enumerate(node1+node2):
        #     node_name = itr+1
        #     # ### ASSIGNING NODE NUMBERED GREATER THAN '5' TO '0'
        #     # if node_name >= 5:
        #     #     node_name = 0
        #     node_dict[node_name] = node
        #     cv2.putText(frame, str(node_name), (node[0]+5, node[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # # ### NODE and COMPONENT MATCHING
        # # netlist_dict = dict()
        # # for idx, component in enumerate(components):
        # #     dist = list()
        # #     temp = list()
        # #     for node in  list(node_dict.keys()):
        # #         dist.append(euclidean_dist(component['center'], node_dict[node]))
        # #         temp.append(node)
        # #     min_arg = np.argmin(dist)
        # #     first_node = temp[min_arg]
        # #     dist.remove(dist[min_arg])
        # #     temp.remove(temp[min_arg])
        # #     second_node = temp[np.argmin(dist)]
        # #     node_list = sorted([first_node, second_node]) 
        # #     netlist_dict['{}'.format(idx)]=[[component['label']], node_list]

        cv2.imshow('Source', frame)
        cv2.waitKey(0)
