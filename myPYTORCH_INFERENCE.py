import argparse
import os, sys
import io
import platform
import shutil
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from yolor.utils.google_utils import attempt_load
from yolor.utils.datasets import LoadStreams, LoadImages
from yolor.utils.general import (check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, strip_optimizer)
from yolor.utils.plots import plot_one_box
from yolor.utils.torch_utils import select_device, load_classifier, time_synchronized

from yolor.models.models import *
from yolor.utils.datasets import *
from yolor.utils.general import *

import PEDApp_config as myconfig

class PYTORCH_INFERENCE(object):
    def __init__(self, parser):
        self.opt = parser.parse_args()
        
    def load_classes(self, path):
        # Loads *.names file at 'path'
        with open(path, 'r') as f:
            names = f.read().split('\n')
        return list(filter(None, names))  # filter removes empty strings (such as last line)

    def detect(self, save_img=False):
        print('DETECTING ...')
        out, source, weights, view_img, save_txt, imgsz, cfg, names = \
            self.opt.output, self.opt.source, self.opt.weights, self.opt.view_img, self.opt.save_txt, self.opt.img_size, self.opt.cfg, self.opt.names
        webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

        # Initialize
        device = select_device(self.opt.device)
        # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        if os.path.exists(out):
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder
        half = device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        model = Darknet(cfg, imgsz).cuda()
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
        #model = attempt_load(weights, map_location=device)  # load FP32 model
        #imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
        model.to(device).eval()
        if half:
            model.half()  # to FP16

        # Second-stage classifier
        classify = False
        if classify:
            modelc = load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
            modelc.to(device).eval()

        # Set Dataloader
        vid_path, vid_writer = None, None
        if webcam:
            view_img = True
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz)
        else:
            save_img = True
            dataset = LoadImages(source, img_size=imgsz, auto_size=64)

        # Get names and colors
        names = self.load_classes(names)
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

        # Run inference
        t0 = time.time()
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
        _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=self.opt.augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes, agnostic=self.opt.agnostic_nms)
            t2 = time_synchronized()

            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
                else:
                    p, s, im0 = path, '', im0s

                save_path = str(Path(out) / Path(p).name)
                txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += '%g %ss, ' % (n, names[int(c)])  # add to string

                    # Write results
                    print('OUTPUTs:')
                    objects = list()
                    for *xyxy, conf, cls in det:
                        left = int(xyxy[0])
                        top = int(xyxy[1])
                        right = int(xyxy[2])
                        bottom = int(xyxy[3])
                        label = str(names[int(cls)])
                        confidence = float(conf)
                        width = right - left
                        height = bottom - top
                        center = (left + int((right-left)/2), top + int((bottom-top)/2))

                        ### CREATING COMPONENTS DICTIONARY
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
                                    "model_type": 'PYTORCH',
                                }
                        objects.append(mydict)
                        
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                        if save_img or view_img:  # Add bbox to image
                            label = '%s %.2f' % (names[int(cls)], conf)
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

                # Print time (inference + NMS)
                print('%sDone. (%.3fs)' % (s, t2 - t1))

                # Stream results
                if view_img:
                    cv2.imshow(p, im0)
                    if cv2.waitKey(1) == ord('q'):  # q to quit
                        raise StopIteration

                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'images':
                        cv2.imwrite(save_path, im0)
                    else:
                        if vid_path != save_path:  # new video
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()  # release previous video writer

                            fourcc = 'mp4v'  # output video codec
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                        vid_writer.write(im0)

        if save_txt or save_img:
            print('Results saved to %s' % Path(out))
            if platform == 'darwin' and not self.opt.update:  # MacOS
                os.system('open ' + save_path)

        print('Done. (%.3fs)' % (time.time() - t0))
        return objects


if __name__ == '__main__':
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
    
    detector = PYTORCH_INFERENCE(parser)
    
    if torch.cuda.is_available():
        print('INFO: Using CUDA device')
    else:
        print('WARNING: CUDA is not available!!!')

    with torch.no_grad():
        components = detector.detect()
        print(components)
        # if opt.update:  # update all models (to fix SourceChangeWarning)
        #     for opt.weights in ['']:
        #         detector.detect()
        #         strip_optimizer(opt.weights)
        # else:
        #     detector.detect()