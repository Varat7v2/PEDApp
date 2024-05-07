'''
M-LSD
Copyright 2021-present NAVER Corp.
Apache License v2.0

Updated by Varat7v2 on 09/15/2022 at the University of Houston
'''
# for mydemo
import os, sys
import cv2
import time
import argparse

# for tflite
import numpy as np
from PIL import Image
import tensorflow as tf

# for square detector

# from utils import pred_squares        ## Uncomment if want to run from local directory
from line_detection.mlsd.utils import pred_squares

import config_v2 as myconfig
import myUtils

# os.environ['CUDA_VISIBLE_DEVICES'] = '' # CPU mode

class MLSD:
    def __init__(self, args):
        self.interpreter, self.input_details, self.output_details = self.load_tflite(args.LS_MODEL_PATH)
        self.params = {'score': args.SCORE_THR,'outside_ratio': args.OUTSIDE_RATIO,'inside_ratio': args.INSIDE_RATIO,
                       'w_overlap': args.W_OVERLAP,'w_degree': args.W_DEGREE,'w_length': args.W_LENGTH,
                       'w_area': args.W_AREA,'w_center': args.W_CENTER}
        self.args = args

    def load_tflite(self, tflite_path):
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        return interpreter, input_details, output_details


    def pred_tflite(self, image_flt):
        segments, squares, score_array, inter_points = pred_squares(image_flt,
                                                                    self.interpreter, 
                                                                    self.input_details, 
                                                                    self.output_details, 
                                                                    [self.args.INPUT_SIZE,
                                                                     self.args.INPUT_SIZE],
                                                                    params=self.params)
        output = {}
        output['segments'] = segments
        output['squares'] = squares
        output['scores'] = score_array
        output['inter_points'] = inter_points

        return output
    
    def findNodes(self, img, horizontal_lines, vertical_lines):
        img_lined = img.copy()
        # if myconfig.DEBUG_FLAG:
        #     for hline, vline in zip(horizontal_lines, vertical_lines):
        #         cv2.line(img, hline[0], hline[1], (0,255,0), 1, cv2.LINE_AA)
        #         cv2.line(img, vline[0], vline[1], (240,16,255), 1, cv2.LINE_AA)

        ## My method (we can use here combination collections method)
        intersection_pts = list()
        for hline in horizontal_lines:
            # cv2.line(img_lined, hline[0], hline[1], (50,255,50), 1, cv2.LINE_AA)
            # horizontal line parameters
            a1 = hline[0][1] - hline[1][1]
            b1 = hline[1][0] - hline[0][0]
            c1 = hline[1][1] * hline[0][0] - hline[1][0] * hline[0][1]
            # m1 = - a1/b1

            for vline in vertical_lines:
                # cv2.line(img_lined, vline[0], vline[1], (255,165,255), 1, cv2.LINE_AA)
                # vertical line parameters
                a2 = vline[0][1] - vline[1][1]
                b2 = vline[1][0] - vline[0][0]
                c2 = vline[1][1] * vline[0][0] - vline[1][0] * vline[0][1]
                # m2 = - a2/b2

                ## Point of intersections - Calculation
                x, y = int((b1 * c2 - b2 * c1) / (a1 * b2 - a2 * b1)), int(((a2 * c1 - a1 * c2) / (a1 * b2 - a2 * b1)))
                intersection_pts.append((x,y))

                # if myconfig.DEBUG_FLAG and myconfig.myNODE_FLAG:
                #     cv2.circle(img, (x, y), 3, (255, 0, 255), -1)

        # # Perform weighted addition of the input image and the overlay
        # OPTIONAL: make the circles transparent (larger alpha --> more transparent)
        img = cv2.addWeighted(img_lined, myconfig.ALPHA, img, 1 - myconfig.ALPHA, 0)

        # print('[INFO]: No. of points of intersection: ', len(intersection_pts))

        return img, intersection_pts

    def intersection_nodes(self, image, output):
        fheight, fwidth = image.shape[:2]
        img_center = (fwidth//2, fheight//2)

        nodes = list()
        lines = output['segments']
        if lines is not None:
            horizontal_lines = list()
            vertical_lines = list()
            # print('[INFO]: Total no. of lines: {}'.format(len(lines)))
            for line in lines:
                x_start, y_start, x_end, y_end = [int(val) for val in line]
                # cv2.line(image, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)

                pt1 = (x_start, y_start)
                pt2 = (x_end, y_end)
                ### Separate out the perfect horizontal and vertical lines
                try:
                    # slope = abs(float("inf") if (x1 == x2) else (y2 - y1) / (x2 - x1))
                    if x_start == x_end:
                        slope = abs(float("inf"))
                    else:
                        slope = np.rad2deg(np.arctan(abs((y_end - y_start) / (x_end - x_start))))

                    if slope <= 0+myconfig.THETA:   # horizontal lines (strict constraint)
                        # print('horizontal slope=', slope)
                        # if True:
                        if (pt1[1]<(img_center[1]-10) or pt1[1]>(img_center[1]+10)) \
                                and (pt2[1]<(img_center[1]-10) or pt2[1]>(img_center[1]+10)):
                            if myconfig.STRETCH_LINE:   # Stretch horizontal line to the image width/height
                                horizontal_lines.append([(0, pt1[1]), (fwidth, pt2[1])])
                            else:
                                horizontal_lines.append([pt1, pt2])
                    elif slope >= 90-myconfig.THETA or slope == float('inf'): # vertical lines
                        # print('vertial slope=', slope)
                        if myconfig.STRETCH_LINE:
                            vertical_lines.append([(pt1[0], 0), (pt2[0], fheight)])
                        else:
                            vertical_lines.append([pt1, pt2])
                except Exception as e:
                    print(e)

        ### Node detection using Point-of-Intersection method
        if len(horizontal_lines)>0 and len(vertical_lines)>0:
            # displays vertical and horizontal lines
            image, nodes = self.findNodes(image, horizontal_lines, vertical_lines)
        else:
            print('[WARNING]: Horizontal and vertical lines are not detected!')

        # if nodes:
        #     for node in nodes:
        #         # cv2.drawMarker(frameDebug, (node[0], node[1]),(255,0,0), markerType=cv2.MARKER_TILTED_CROSS,
        #         #                markerSize=40, thickness=1, line_type=cv2.LINE_AA)
        #         cv2.circle(image, (node[0], node[1]), 10, (255,0,0), -1)
        #
        # ### Observe nodes using MLSD
        # if myconfig.mlsdNODE_FLAG:
        #     for pt in output['inter_points']:
        #         x, y = [int(val) for val in pt]
        #         cv2.circle(image, (x, y), 10, (0, 0, 255), -1)

            # if myconfig.SQUARES_FLAG:
            #     for square in output['squares']:
            #         cv2.polylines(image, [square.reshape([-1, 1, 2])], True, (0, 190, 246), 2)
            #
            #     for square in output['squares'][0:1]:
            #         cv2.polylines(image, [square.reshape([-1, 1, 2])], True, (0, 255, 255), 2)
            #         for pt in square:
            #             cv2.circle(image, (int(pt[0]), int(pt[1])), 10, (100, 100, 0), -1)

        return image, nodes

if __name__ == '__main__':
    mlsd = MLSD(myconfig)

    if myconfig.IMG_FLAG:
        image = cv2.imread(myconfig.IMAGE_PATH)

        ### Line Segment Detection with Mobile (M-LSD)
        output = mlsd.pred_tflite(image)

        ### Nodes detection using Point-of-Intersections
        image, nodes = mlsd.intersection_nodes(image, output)

        if nodes:
            ### Combine the nodes from both the methods (MLSD+PoI)
            all_nodes = list(tuple(x) for x in output['inter_points']) + nodes
            print('Total size of all_nodes = ', len(all_nodes))

            for node in all_nodes:
                cv2.circle(image, node, 5, (0, 0, 255), -1)
                
        cv2.imshow('Test image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
                
    else:
        cap = cv2.VideoCapture(myconfig.VIDEO_PATH)
        im_width = int(cap.get(3))
        im_height = int(cap.get(4))
        
        if myconfig.WRITE_VIDEO:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_output = cv2.VideoWriter(myconfig.VIDEO_OUTPUT, fourcc, 30, (im_width, im_height))
            frame_index = -1

        while True:
            t1 = time.time()
            ret, frame = cap.read()

            if ret == 0:
                print('[INFO]: No frames detected!')
                break

            ### Line Segment Detection with Mobile (M-LSD)
            output = mlsd.pred_tflite(frame)

            ### Nodes detection using Point-of-Intersections
            frame, nodes = mlsd.intersection_nodes(frame, output)

            if nodes:
                ### Combine the nodes from both the methods (MLSD+PoI)
                nodes = list(tuple(x) for x in output['inter_points']) + nodes
                print('Total size of all_nodes = ', len(nodes))

                for node in nodes:
                    cv2.circle(frame, node, 5, (0, 0, 255), -1)

            t2 = time.time() - t1
            fps = 1 / t2
            
            cv2.imshow("Webcam", frame)

            if myconfig.WRITE_VIDEO:
                video_output.write(frame)

            k = cv2.waitKey(1) & 0xff
            if k == ord('q') or k == 27:
                break

        cap.release()
        if myconfig.WRITE_VIDEO:
            video_output.release()
        cv2.destroyAllWindows()
