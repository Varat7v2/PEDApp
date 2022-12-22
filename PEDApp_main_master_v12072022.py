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

from scipy import stats

# from myTensorFlow.utils import label_map_util
from ecd.myFROZEN_GRAPH import FROZEN_GRAPH_INFERENCE
from line_detection.hough_line_transform_v9 import NODE_DETECTION
from line_detection.mlsd.myMLSD import MLSD
# from line_detection.mlsd.utils import pred_squares
from ecd.myPYTORCH_INFERENCE import PYTORCH_INFERENCE
from pyspice.myPySPICE_v2 import myPYSPICE
from line_detection.select_node import KMEANS_CLUSTERING

import config as myconfig

import matplotlib
matplotlib.use('TkAgg')

plt.rcParams.update({'figure.max_open_warning': 0})
# plt.rcParams.update({'font.size': 20})

# def load_labelmap():
#     # List of the strings that is used to add correct label for each box.
#     label_map = label_map_util.load_labelmap(myconfig.COMPONENTS_LABELS)
#     categories = label_map_util.convert_label_map_to_categories(label_map, 
#         max_num_classes=myconfig.COMPONENTS_CLASSES, use_display_name=True)
#     category_index = label_map_util.create_category_index(categories)

def remove_components(img, component):
    # TODO: Using Mode of the arrays find the background pixels of the image
    # print(stats.mode(img, axis=2))
    # sys.exit(0)
    # mask_2d = np.full((component['height'], component['width']), 255, dtype=np.uint8)
    mask_3d = np.full((component['height'], component['width'], 3), 240, dtype=np.uint8)
    img[component['top']:component['bottom'], component['left']:component['right']] = mask_3d

    return img

def euclidean_dist(component, node):
    return np.sqrt((node[0]-component[0])**2 + (node[1]-component[1])**2)

def simulation_plots(analysis, converter):
    dict_V = dict()
    dict_I = dict()
    
    for node in analysis.nodes.values():
        voltages = np.array(node)
        dict_V[str(node)] = voltages

        if (myconfig.PLOT_FLAG) and (str(node) in ['1','4','gate']):
            plt.figure()
            plt.plot(range(voltages.shape[0]), voltages, linewidth=4)
            # plt.title('Voltage at node {}'.format(str(node)))
            plt.xlabel('time (us)', fontsize=20, weight='bold')
            plt.ylabel('Voltages (V)', fontsize=20, weight='bold')
            plt.grid()
            plt.tight_layout()
            plt.xlim([0, myconfig.X_LIMIT])

            if myconfig.SAVE_PLOT:
                plt.savefig('plots/voltage/{}.png'.format(str(node)))
        else:
            continue
        plt.show()


    for node in analysis.branches.values():
        currents = np.array(node)
        dict_I[str(node)] = currents

        if (myconfig.PLOT_FLAG) and (str(node) in ['v1', 'vbranch_r', 'vbranch_l']):
            plt.figure()
            plt.plot(range(currents.shape[0]), currents, linewidth=4)
            # plt.title('Current at node {}'.format(str(node)))
            plt.xlabel('time (us)', fontsize=20, weight='bold')
            plt.ylabel('current (A)', fontsize=20, weight='bold')
            plt.grid()
            plt.tight_layout()
            plt.xlim([0, myconfig.X_LIMIT])

            if myconfig.SAVE_PLOT:
                plt.savefig('plots/current/{}.png'.format(str(node)))
        else:
            continue
        plt.show()

    dict_V['time'] = [i*1e-4 for i in range(len(voltages))]
    dict_I['time'] = [i*1e-4 for i in range(len(currents))]
    if myconfig.SAVE_CSV:
        dfV = pd.DataFrame.from_dict(dict_V)
        dfI = pd.DataFrame.from_dict(dict_I)
        dfV.to_csv('outputs/{}_voltage.csv'.format(converter), encoding='utf-8', index=False)
        dfI.to_csv('outputs/{}_current.csv'.format(converter), encoding='utf-8', index=False)
        print('[INFO]: CSV files are saved successfully!')

    if myconfig.SAVE_PLOT:
        print('[INFO]: All voltage and current waveforms saved successfully!')

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
    mlsd = MLSD(myconfig)

    if myconfig.IMG_FLAG:
        frame = cv2.imread(myconfig.SOURCE)
        frameDebug = frame.copy()
        frame_flt = frame.copy()
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
            frame_flt = remove_components(frame_flt, component)
            cv2.drawMarker(frame, component['center'], (255, 255, 0), markerType=cv2.MARKER_TILTED_CROSS,
                           markerSize=15, thickness=3, line_type=cv2.LINE_AA)
            cv2.rectangle(frame, (component['left'], component['top']),
                (component['right'], component['bottom']), (255, 0, 0), 2, 8)
            cv2.putText(frame, component['label'], (component['left'], component['top']-5),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        frame_bnw = np.where(frame_flt <= 128, 0, 255).astype(np.uint8)
        cv2.imwrite("outputs/black_n_white.jpg", frame_bnw)
        cv2.imwrite('outputs/component_removed.jpg', frame_flt)
        cv2.imshow('Filtered Image', cv2.resize(frame_flt, (640, 480)))

        ### GET NODES
        if myconfig.LSD_type == 'houghTransform':
            ### Horizontal and Vertical lines detectione
            _, horizontal_lines, vertical_lines = node_det.houghTransform(frame_bnw, im_width, im_height)

            ### Node detection using Point-of-Intersection method
            if len(horizontal_lines)>0 and len(vertical_lines)>0:
                frameDebug, nodes = node_det.findNodes(frame, horizontal_lines, vertical_lines)

        elif myconfig.LSD_type == 'MLSD':
            frameDebug, outputs = mlsd.pred_tflite(frame, frame_bnw)
            # frame = outputs['inter_points']



        # ### KMeans Clustering for separating out the cluster groups
        # k_nodes = kmeans.kmeans_clustering(myconfig.NODES_NUMBER, intersection_pts)
        # myNodes = list()
        # for idx, node in enumerate(k_nodes):
        #     node_num = idx+1
        #     cen_x, cen_y = int(node[0]), int(node[1])
        #     myNodes.append([cen_x, cen_y])
        #     if myconfig.DEBUG_FLAG and myconfig.NODE_SHOW_FLAG:
        #         # cv2.drawMarker(src, (cen_x, cen_y), (255,98,41), markerType=cv2.MARKER_TILTED_CROSS,
        #         #                markerSize=20, thickness=3, line_type=cv2.LINE_AA)
        #         cv2.circle(src, (cen_x, cen_y), 50, (0,0,0), 2)
        #         cv2.putText(src, str(node_num), (cen_x-30, cen_y-20), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 0, 0), 4, cv2.LINE_AA)




        # for idx, node in enumerate(nodes):
        #     node_num = idx+1
        #     # # Assigning node greater than 5 to 0
        #     # if idx >= 4:
        #     #     idx = 0
        #     cv2.circle(frame, (node[0], node[1]), 10, (0,255,0), -1)
        #     # cv2.putText(frame, str(node_num), (node[0]+5, node[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # netlist_dict = node_component_match(components, nodes)

        # if myconfig.DEBUG_PEDAPP:
        #     print('[INFO]: Netlist Generated: ')
        #     for key, item in netlist_dict.items():
        #         print('\t\t{} {}: {}'.format(key, item[0][0], item[1]))
        # # Identify the power converter
        # converter = identify_circuit(netlist_dict)
        # print('[INFO]: Converter type:', converter)

        # ### CIRCUIT SIMULATION
        # circuit, analysis = pyspice.simulate_ckt(netlist_dict, converter)
        # print('\n\n CIRCUIT NETLIST \n', circuit, '\n')
        
        # ### Plots Results
        # simulation_plots(analysis, converter)

        ### TODO: PCB DESIGN - PLACING COMPONENTS
        # df_BOM = pcb

        ### TODO: PCB DESIGN - AUTOROUTING
        ### TODO: SCHEMDRAW THE CIRCUIT

        if not myconfig.DEBUG_FLAG:
            cv2.imshow('Source', frame)
        else:
            cv2.imshow('Debug', cv2.resize(frameDebug, (640, 480)))
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

