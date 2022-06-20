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
plt.rcParams.update({'figure.max_open_warning': 0})

# from myTensorFlow.utils import label_map_util
from ecd.myFROZEN_GRAPH import FROZEN_GRAPH_INFERENCE
from line_detection.hough_line_transform_v7 import NODE_DETECTION
from ecd.myPYTORCH_INFERENCE import PYTORCH_INFERENCE
from pyspice.myPySPICE_v2 import myPYSPICE

import config as myconfig

import matplotlib
matplotlib.use('TkAgg')

# def load_labelmap():
#     # List of the strings that is used to add correct label for each box.
#     label_map = label_map_util.load_labelmap(myconfig.COMPONENTS_LABELS)
#     categories = label_map_util.convert_label_map_to_categories(label_map, 
#         max_num_classes=myconfig.COMPONENTS_CLASSES, use_display_name=True)
#     category_index = label_map_util.create_category_index(categories)

def remove_components(gray, component):
    mask = np.full((component['height'], component['width']), 255, dtype=np.uint8)
    gray[component['top']:component['bottom'], component['left']:component['right']] = mask
    return gray

def euclidean_dist(component, node):
    return np.sqrt((node[0]-component[0])**2 + (node[1]-component[1])**2)

def simulation_plots(analysis):
    dict_V = dict()
    dict_A = dict()

    for node in analysis.nodes.values():
        voltages = np.array(node)
        dict_V[str(node)] = voltages

        plt.figure()
        plt.plot(range(voltages.shape[0]), voltages)
        plt.title('Voltage at node {}'.format(str(node)))
        plt.xlabel('time (us)')
        plt.ylabel('Voltages (V)')
        plt.grid()
        plt.tight_layout()
        plt.xlim([0, myconfig.X_LIMIT])

        if (myconfig.PLOT_FLAG) and (str(node) in ['1','4','gate']):
            plt.show()
        if myconfig.SAVE_PLOT:
            plt.savefig('plots/voltage/{}.png'.format(str(node)))

    for item, node in enumerate(analysis.branches.values()):
        currents = np.array(node)
        dict_A[str(node)] = currents
    
        plt.figure()
        plt.plot(range(currents.shape[0]), currents)
        plt.title('Current at node {}'.format(str(node)))
        plt.xlabel('time (us)')
        plt.ylabel('current (A)')
        plt.grid()
        plt.tight_layout()
        plt.xlim([0, myconfig.X_LIMIT])
        
        if (myconfig.PLOT_FLAG) and (str(node) in ['1','4']):
            plt.show()
        if myconfig.SAVE_PLOT:
            plt.savefig('plots/current/{}.png'.format(str(node)))

    print('All voltage and current waveforms successfully saved!!!')

### NODE and COMPONENT MATCHING
def node_component_match(components, nodes):
    netlist_dict = dict()
    for idx, component in enumerate(components):
        dist = list()
        for node in nodes:
            dist.append(euclidean_dist(component['center'], node))
        argmin = np.argsort(dist)+1 # Add 1 to name nodes from 1,2,3,...
        netlist_dict[idx]=[[component['label']], sorted(argmin[:2])]

    return netlist_dict

def identify_circuit(netlist_dict):
    BUCK_left, BUCK_right, BUCK_bottom = False, False, False
    BOOST_left, BOOST_right, BOOST_bottom = False, False, False
    BUCKBOOST_left, BUCKBOOST_right, BUCKBOOST_bottom = False, False, False

    converter = 'Unknown Converter'
    for branch, component_info in netlist_dict.items():
        comp_name = component_info[0][0]
        left_node = component_info[1][0]
        right_node = component_info[1][1]

        ### BUCK CONVERTER CHECK
        if left_node == 2:
            if right_node == 3:
                if comp_name == 'inductor':
                    BUCK_right = True
                if comp_name == 'diode':
                    BOOST_right = True
                    BUCKBOOST_right = True # diode direction is different that of boost
            if right_node == 6:
                if comp_name == 'diode':
                    BUCK_bottom = True
                if comp_name=='switch' or comp_name=='transistor' or comp_name=='mosfet':
                    BOOST_bottom = True
                if comp_name == 'inductor':
                    BUCKBOOST_bottom = True
        if right_node == 2:
            if left_node == 1:
                if comp_name=='switch' or comp_name=='transistor' or comp_name=='mosfet':
                    BUCK_left = True
                    BUCKBOOST_left = True
                if comp_name == 'inductor':
                    BOOST_left = True

    if (BUCK_left and BUCK_right and BUCK_bottom):
        converter = 'Buck Converter'
    elif (BOOST_left and BOOST_right and BOOST_bottom):
        converter = 'Boost Converter'
    elif (BUCKBOOST_left and BUCKBOOST_right and BUCKBOOST_bottom):
       converter = 'Buck-Boost Converter'

    return converter

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

    # ecd_detector = FROZEN_GRAPH_INFERENCE(myconfig.FROZEN_GRAPH_PEDAPP)
    pt_detector = PYTORCH_INFERENCE(parser)
    node_det = NODE_DETECTION()
    pyspice = myPYSPICE(myconfig.PROJECT)

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

        ### Horizontal and Vertical lines detectione
        frameDebug, horizontal_lines, vertical_lines = node_det.houghTransform(gray, im_width, im_height)

        ### Node detection using Point-of-Intersection method
        if len(horizontal_lines)>0 and len(vertical_lines)>0:
            frameDebug, nodes = node_det.findNodes(frame, horizontal_lines, vertical_lines)

        for idx, node in enumerate(nodes):
            node_num = idx+1
            # # Assigning node greater than 5 to 0
            # if idx >= 4:
            #     idx = 0
            cv2.circle(frame, (node[0], node[1]), 10, (0,255,0), -1)
            cv2.putText(frame, str(node_num), (node[0]+5, node[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        netlist_dict = node_component_match(components, nodes)

        if myconfig.DEBUG_PEDAPP:
            print('[INFO]: Netlist Generated: ')
            for key, item in netlist_dict.items():
                print('\t\t{} {}: {}'.format(key, item[0][0], item[1]))
        # Identify the power converter
        converter = identify_circuit(netlist_dict)
        print('[INFO]: Converter type:', converter)

        # ### CIRCUIT SIMULATION
        # circuit, analysis = pyspice.simulate_ckt(netlist_dict, converter)
        # print('\n\n CIRCUIT NETLIST \n', circuit, '\n')
        #
        # ### TODO: PLOT THE GRAPHS
        # simulation_plots(analysis)

        # ### TODO: PCB DESIGN - PLACING COMPONENTS
        # # df_BOM = pcb
        #
        # ### TODO: PCB DESIGN - AUTOROUTING
        #
        # ### TODO: SCHEMDRAW THE CIRCUIT

        if not myconfig.DEBUG_FLAG:
            cv2.imshow('Source', frame)
        else:
            cv2.imshow('Debug', frameDebug)
        curr_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cv2.imwrite('./outputs/{}.jpg'.format(curr_datetime), frame)
        cv2.waitKey(0)

    # else:
    #     if myconfig.WEBCAM_FLAG:
    #         source = 0
    #     elif myconfig.VIDEO_FLAG:
    #         source = myconfig.VIDEO_INPUT
    #     cap = cv2.VideoCapture(source)
    #     im_width = int(cap.get(3))
    #     im_height = int(cap.get(4))
    #
    #     if myconfig.WRITE_VIDEO:
    #         fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #         video_output = cv2.VideoWriter(myconfig.VIDEO_OUTPUT, fourcc, 30, (im_width, im_height))
    #         frame_index = -1
    #
    #     while True:
    #         t1 = time.time()
    #         ret, frame = cap.read()
    #
    #         if ret == 0:
    #             break
    #
    #         im_height, im_width, im_channel = frame.shape
    #         # frame = cv2.flip(frame, 1)
    #
    #         frame, components = ecd_detector.run_frozen_graph(frame, im_width, im_height)
    #         for component in components:
    #             cv2.rectangle(frame, (component['left'], component['top']),
    #                 (component['right'], component['bottom']), (0, 255, 0), 2, 8)
    #             cv2.putText(frame, component['label'], (component['right']-20, component['top']-5),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    #
    #         t2 = time.time() - t1
    #         fps = 1 / t2
    #
    #         cv2.putText(frame, "FPS: {:.2f}".format(fps), (10,20),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 1, cv2.LINE_AA)
    #         cv2.imshow("FACE DETECTION USING FROZEN GRAPH", frame)
    #
    #         if myconfig.WRITE_VIDEO:
    #             video_output.write(frame)
    #             frame_index = frame_index + 1
    #
    #         k = cv2.waitKey(1) & 0xff
    #         if k == ord('q') or k == 27:
    #             break
    #
    #     cap.release()
    #     if myconfig.WRITE_VIDEO:
    #         video_output.release()
    #     cv2.destroyAllWindows()

