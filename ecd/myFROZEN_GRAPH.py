#!/usr/bin/env python3

import sys
import time
import numpy as np
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import cv2

import config as myconfig

from myTensorFlow.utils import label_map_util
from myTensorFlow.utils import visualization_utils_color as vis_util

class FROZEN_GRAPH_INFERENCE:
    
    def __init__(self, MODEL_PATH):
        """Tensorflow detector
        """
        self.inference_list = list()
        PATH_TO_CKPT = MODEL_PATH
        self.count = 0

        NUM_CLASSES = myconfig.COMPONENTS_CLASSES
        PATH_TO_LABELS = myconfig.COMPONENTS_LABELS

        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, 
            max_num_classes=NUM_CLASSES, use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        with self.detection_graph.as_default():
            config = tf.ConfigProto()
            # Allow growth: (more flexible)
            #config.gpu_options.allow_growth = False
            config.gpu_options.allow_growth = True
            #Allocate fixed memory
            #config.gpu_options.per_process_gpu_memory_fraction = 0.3
            self.sess = tf.Session(graph=self.detection_graph, config=config)
            self.windowNotSet = True

    def draw_bounding_box(self, image, scores, boxes, classes, im_width, im_height):
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes).astype(np.int32)

        objects = list()
        bboxs = list()
        idx = 0

        for score, box, name in zip(scores, boxes, classes):
            if score > 0.6:
                # ymin, xmin, ymax, xmax = box
                label = self.category_index[name]['name']
               
                left = int(box[1]*im_width)
                top = int(box[0]*im_height)
                right = int(box[3]*im_width)
                bottom = int(box[2]*im_height)
                bboxs.append([left, top, right, bottom])

                # cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2, 8)
                # cv2.putText(image, label, (right-20, top-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                width = right - left
                height = bottom - top
                center = (left + int((right-left)/2), top + int((bottom-top)/2))
                confidence = score
                # label = name

                mydict = {
                    "width": width,
                    "height": height,
                    "left": left,
                    "right": right,
                    "top": top,
                    "bottom": bottom,
                    "confidence": confidence,
                    "label": label,
                    "center": center,
                    "model_type": 'FROZEN_GRAPH',
                    "bboxs": bboxs
                    }
                objects.append(mydict)
                idx += 1

        return image, objects

    def run_frozen_graph(self, image, im_width, im_height):
        """image: bgr image
        return (boxes, scores, classes, num_detections)
        """
        image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
        # Actual detection.
        start_time = time.time()
        (boxes, scores, classes, num_detections) = self.sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
        elapsed_time = time.time() - start_time
        self.inference_list.append(elapsed_time)
        self.count = self.count + 1
        average_inference = sum(self.inference_list)/self.count
        # print('Average inference time: {}'.format(average_inference))

        # Visualization of the results of a detection.
        # vis_util.visualize_boxes_and_labels_on_image_array(
        #     #image_np,
        #     image,
        #     np.squeeze(boxes),
        #     np.squeeze(classes).astype(np.int32),
        #     np.squeeze(scores),
        #     self.category_index,
        #     use_normalized_coordinates=True,
        #     line_thickness=4)
        # return image

        # Draw bounding boxes on the image (MY)
        image, objects = self.draw_bounding_box(image, scores, boxes, classes, im_width, im_height)
        return image, objects
