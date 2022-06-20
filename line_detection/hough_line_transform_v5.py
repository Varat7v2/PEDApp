"""
@brief This program demonstrates line finding with the Hough transform
"""
import sys
import math
import cv2
import numpy as np
import config as myconfig


class NODE_DETECTION():
    def __init__(self):
        pass

    def houghTransform(self, src):
        dst = cv2.Canny(src, 50, 200, None, 3)

        # Copy edges to the images that will display the results in BGR
        cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
        cdstP = np.copy(cdst)

        lines = cv2.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)
        horizontal_lines = list()
        vertical_lines = list()

        if lines is not None:
            for i in range(0, len(lines)):
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1, y1 = int(x0 + 1000 * (-b)), int(y0 + 1000 * (a))
                x2, y2 = int(x0 - 1000 * (-b)), int(y0 - 1000 * (a))
                pt1 = (x1, y1)
                pt2 = (x2, y2)

                ## FOR MAKING HORIZONTAL AND VERTICAL LINES EXTENDED
                # horizontal_lines.append([(0, pt1[1]), (fwidth, pt2[1])])
                # verticle_lines.append([(pt1[0], 0), (pt2[0], fheight)])

                try:
                    slope = abs(float("inf") if (x1 == x2) else (y2 - y1) / (x2 - x1))
                    # print(slope)
                    if slope >= 0 and slope < 1:
                        horizontal_lines.append([pt1, pt2])
                    elif (slope > 1 and slope < 1e10) or slope == float('inf'):
                        vertical_lines.append([pt1, pt2])
                except Exception as e:
                    print(e)
        return horizontal_lines, vertical_lines

    def findNodes(self, src, fheight, fwidth, horizontal_lines, vertical_lines):
        hline_y = list()
        vline_x = list()
        for hline in horizontal_lines:
            hline_y += [hline[0][1], hline[1][1]]
            if myconfig.DEBUG_FLAG:
                cv2.line(src, hline[0], hline[1], (0,0,255), 3, cv2.LINE_AA)

        for vline in vertical_lines:
            vline_x += [vline[0][0], vline[1][0]]
            if myconfig.DEBUG_FLAG:
                cv2.line(src, vline[0], vline[1], (255,0,0), 3, cv2.LINE_AA)

        hline_ymin = min(hline_y)
        hline_ymax = max(hline_y)
        hline_yavg = int((hline_ymin + hline_ymax) / 2)

        vline_xmin = min(vline_x)
        vline_xmax = max(vline_x)
        vline_xavg = int((vline_xmin + vline_xmax) / 2)

        if myconfig.DEBUG_FLAG:
            cv2.line(src, (0, hline_yavg), (fwidth, hline_yavg), (255,255,0), 3, cv2.LINE_AA)
            cv2.line(src, (vline_xavg, 0), (vline_xavg, fheight), (255,255,0), 3, cv2.LINE_AA)

        # (vline_xavg --> x=k, hline_yavg --> y=l)
        # for i in range(len(horizontal_lines)-1):
        #     dist = np.sqrt((horizontal_lines[i+1] - horizontal_lines[i])**2
        #         + ())

        intersection_points = 0

        ### CATEGORIZING HORIZONTAL LINES
        hline_ref = [(0, hline_yavg), (fwidth, hline_yavg)]
        vline_ref = [(vline_xavg, 0), (vline_xavg, fheight)]
        # vertical line parameters
        a2 = vline_ref[0][1] - vline_ref[1][1]
        b2 = vline_ref[1][0] - vline_ref[0][0]
        c2 = vline_ref[1][1] * vline_ref[0][0] - vline_ref[1][0] * vline_ref[0][1]
        # m2 = - a2/b2
        intersect_pts = list()
        for idx, hline in enumerate(horizontal_lines):
            # horizontal line parameters
            a1 = hline[0][1] - hline[1][1]
            b1 = hline[1][0] - hline[0][0]
            c1 = hline[1][1] * hline[0][0] - hline[1][0] * hline[0][1]
            # m1 = - a1/b1

            ## POINT OF INTERSECTION
            x, y = int((b1 * c2 - b2 * c1) / (a1 * b2 - a2 * b1)), int(((a2 * c1 - a1 * c2) / (a1 * b2 - a2 * b1)))
            intersect_pts.append((x, y))
            # if myconfig.DEBUG_FLAG:
            #     cv2.circle(src, (x,y), 5, (0, 0, 0), -1)

        avg_dist = 0
        hgroup1, hgroup2 = list(), list()
        for idx, pt in enumerate(intersect_pts):
            dist = np.sqrt((pt[0] - intersect_pts[0][0]) ** 2 + (pt[1] - intersect_pts[0][1]) ** 2)
            # print(dist)
            if dist > 100:
                hgroup2.append(idx)
            else:
                hgroup1.append(idx)
        # print(hgroup1)
        # print(hgroup2)

        hslope = 9999
        myline1 = 9999
        for line in hgroup1:
            slope = abs(horizontal_lines[line][1][1] - horizontal_lines[line][0][1]) / (
                    horizontal_lines[line][1][0] - horizontal_lines[line][0][0])
            if slope < hslope:
                hslope = slope
                myline1 = line
        # if myconfig.DEBUG_FLAG:
        #     cv2.line(src, horizontal_lines[myline1][0], horizontal_lines[myline1][1], (0,0,255), 3, cv2.LINE_AA)

        hslope = 9999
        myline2 = 9999
        for line in hgroup2:
            slope = abs(horizontal_lines[line][1][1] - horizontal_lines[line][0][1]) / (
                    horizontal_lines[line][1][0] - horizontal_lines[line][0][0])
            if slope < hslope:
                hslope = slope
                myline2 = line
        # if myconfig.DEBUG_FLAG:
        #     cv2.line(src, horizontal_lines[myline2][0], horizontal_lines[myline2][1], (0,0,255), 3, cv2.LINE_AA)
        myhorizontal_lines = [(horizontal_lines[myline1][0], horizontal_lines[myline1][1]), (
            horizontal_lines[myline2][0], horizontal_lines[myline2][1])]

        ### CATEGORIZING VERTICLE LINES
        # horizontal line parameters
        a1 = hline_ref[0][1] - hline_ref[1][1]
        b1 = hline_ref[1][0] - hline_ref[0][0]
        c1 = hline_ref[1][1] * hline_ref[0][0] - hline_ref[1][0] * hline_ref[0][1]
        # m1 = - a1/b1
        intersect_pts = list()
        for idx, vline in enumerate(vertical_lines):
            # vertical line parameters
            a2 = vline[0][1] - vline[1][1]
            b2 = vline[1][0] - vline[0][0]
            c2 = vline[1][1] * vline[0][0] - vline[1][0] * vline[0][1]
            # m2 = - a2/b2

            ## POINT OF INTERSECTION
            x, y = int((b1 * c2 - b2 * c1) / (a1 * b2 - a2 * b1)), int(((a2 * c1 - a1 * c2) / (a1 * b2 - a2 * b1)))
            intersect_pts.append((x, y))
            # if myconfig.DEBUG_FLAG:
            #     cv2.circle(src, (x,y), 5, (0, 0, 0), -1)

        avg_dist = 0
        vgroup1, vgroup2, vgroup3, vgroup4 = list(), list(), list(), list()
        for idx, pt in enumerate(intersect_pts):
            dist = np.sqrt((pt[0] - intersect_pts[0][0]) ** 2 + (pt[1] - intersect_pts[0][1]) ** 2)
            # print(dist)
            if dist < 250:
                vgroup1.append(idx)
            elif dist > 250 and dist < 450:
                vgroup2.append(idx)
            elif dist > 450 and dist < 650:
                vgroup3.append(idx)
            elif dist > 650:
                vgroup4.append(idx)
            else:
                print('ERROR: ValueError in distance!')
        # print(vgroup1)
        # print(vgroup2)
        # print(vgroup3)
        # print(vgroup4)

        ### FOR VERTICLE LINE1
        vslope = 0
        myline1 = 9999
        for line in vgroup1:
            x1 = vertical_lines[line][0][0]
            x2 = vertical_lines[line][1][0]
            y1 = vertical_lines[line][0][1]
            y2 = vertical_lines[line][1][1]
            slope = abs(float("inf") if (x1 == x2) else (y2 - y1) / (x2 - x1))
            # print(slope)
            if slope == float("inf") and vslope is not float("inf"):
                vslope = slope
                myline1 = line
            elif (slope > vslope):
                vslope = slope
                myline1 = line
            else:
                continue
        # if myconfig.DEBUG_FLAG:
        #     cv2.line(src, vertical_lines[myline1][0], vertical_lines[myline1][1], (0,0,255), 3, cv2.LINE_AA)

        ### FOR VERTICLE LINE2
        vslope = 0
        myline2 = 9999
        for line in vgroup2:
            x1 = vertical_lines[line][0][0]
            x2 = vertical_lines[line][1][0]
            y1 = vertical_lines[line][0][1]
            y2 = vertical_lines[line][1][1]
            slope = abs(float("inf") if (x1 == x2) else (y2 - y1) / (x2 - x1))
            # print(slope)
            if slope == float("inf") and vslope is not float("inf"):
                vslope = slope
                myline2 = line
            elif (slope > vslope):
                vslope = slope
                myline2 = line
            else:
                continue
        # if myconfig.DEBUG_FLAG:
        #     cv2.line(src, vertical_lines[myline2][0], vertical_lines[myline2][1], (0,0,255), 3, cv2.LINE_AA)

        ### FOR VERTICLE LINE3
        vslope = 0
        myline3 = 9999
        for line in vgroup3:
            x1 = vertical_lines[line][0][0]
            x2 = vertical_lines[line][1][0]
            y1 = vertical_lines[line][0][1]
            y2 = vertical_lines[line][1][1]
            slope = abs(float("inf") if (x1 == x2) else (y2 - y1) / (x2 - x1))
            # print(slope)
            if slope == float("inf") and vslope is not float("inf"):
                vslope = slope
                myline3 = line
            elif (slope > vslope):
                vslope = slope
                myline3 = line
            else:
                continue
        # if myconfig.DEBUG_FLAG:
        #     cv2.line(src, vertical_lines[myline3][0], vertical_lines[myline3][1], (0,0,255), 3, cv2.LINE_AA)

        ### FOR VERTICLE LINE4
        vslope = 0
        myline4 = 9999
        for line in vgroup4:
            x1 = vertical_lines[line][0][0]
            x2 = vertical_lines[line][1][0]
            y1 = vertical_lines[line][0][1]
            y2 = vertical_lines[line][1][1]
            slope = abs(float("inf") if (x1 == x2) else (y2 - y1) / (x2 - x1))
            # print(slope)
            if slope == float("inf") and vslope is not float("inf"):
                vslope = slope
                myline4 = line
            elif (slope > vslope):
                vslope = slope
                myline4 = line
            else:
                continue
        # if myconfig.DEBUG_FLAG:
        #     cv2.line(src, vertical_lines[myline4][0], vertical_lines[myline4][1], (0,0,255), 3, cv2.LINE_AA)
        myvertical_lines = [(
            vertical_lines[myline1][0], vertical_lines[myline1][1]), (
            vertical_lines[myline2][0], vertical_lines[myline2][1]), (
            vertical_lines[myline3][0], vertical_lines[myline3][1]), (
            vertical_lines[myline4][0], vertical_lines[myline4][1])]

        ## My method (we can use here combination collections method)
        myNodes = list()
        if not myconfig.DEBUG_FLAG:
            for hline in myhorizontal_lines:
                # horizontal line parameters
                a1 = hline[0][1] - hline[1][1]
                b1 = hline[1][0] - hline[0][0]
                c1 = hline[1][1] * hline[0][0] - hline[1][0] * hline[0][1]
                # m1 = - a1/b1
                for vline in myvertical_lines:
                    # vertical line parameters
                    a2 = vline[0][1] - vline[1][1]
                    b2 = vline[1][0] - vline[0][0]
                    c2 = vline[1][1] * vline[0][0] - vline[1][0] * vline[0][1]
                    # m2 = - a2/b2

                    intersection_points += 1

                    ## POINT OF INTERSECTION
                    x, y = int((b1 * c2 - b2 * c1) / (a1 * b2 - a2 * b1)), int(((a2 * c1 - a1 * c2) / (a1 * b2 - a2 * b1)))
                    myNodes.append((x,y))
                    # if x >= min(
                    #     hline[0][0], hline[1][0], vline[0][0], vline[1][0]) and x <= max(
                    #     hline[0][0], hline[1][0], vline[0][0], vline[1][0]):
                    #     if y >= min(
                    #         hline[0][1], hline[1][1], vline[0][1], vline[1][1]) and y <= max(
                    #         hline[0][1], hline[1][1], vline[0][1], vline[1][1]):
                    cv2.circle(src, (x, y), 5, (0, 255, 0), -1)
        else:
            for hline in horizontal_lines:
                # horizontal line parameters
                a1 = hline[0][1] - hline[1][1]
                b1 = hline[1][0] - hline[0][0]
                c1 = hline[1][1] * hline[0][0] - hline[1][0] * hline[0][1]
                # m1 = - a1/b1
                for vline in vertical_lines:
                    # vertical line parameters
                    a2 = vline[0][1] - vline[1][1]
                    b2 = vline[1][0] - vline[0][0]
                    c2 = vline[1][1] * vline[0][0] - vline[1][0] * vline[0][1]
                    # m2 = - a2/b2

                    intersection_points += 1

                    ## POINT OF INTERSECTION
                    x, y = int((b1 * c2 - b2 * c1) / (a1 * b2 - a2 * b1)), int(
                        ((a2 * c1 - a1 * c2) / (a1 * b2 - a2 * b1)))
                    myNodes.append((x, y))
                    # if x >= min(
                    #     hline[0][0], hline[1][0], vline[0][0], vline[1][0]) and x <= max(
                    #     hline[0][0], hline[1][0], vline[0][0], vline[1][0]):
                    #     if y >= min(
                    #         hline[0][1], hline[1][1], vline[0][1], vline[1][1]) and y <= max(
                    #         hline[0][1], hline[1][1], vline[0][1], vline[1][1]):
                    if myconfig.DEBUG_FLAG:
                        cv2.circle(src, (x, y), 5, (0, 255, 0), -1)

        print('No. of lines: {}'.format(len(horizontal_lines)+len(vertical_lines)))
        print('No. of horizontal_lines: {}'.format(len(horizontal_lines)))
        print('No. of verticle_lines: {}'.format(len(vertical_lines)))
        print('No. of final intersections points: {}'.format(intersection_points))

        return src, myNodes, hline_yavg


def main(argv):
    node_det = NODE_DETECTION()
    default_file = myconfig.IMG_PATH
    filename = argv[0] if len(argv) > 0 else default_file
    # Loads an image
    frame = cv2.imread(cv2.samples.findFile(filename))
    frameDubug = frame.copy()
    original_frame = frame.copy()
    src = cv2.imread(cv2.samples.findFile(filename), cv2.IMREAD_GRAYSCALE)
    fheight, fwidth = src.shape[:2]
    print('Img_height={}, Img_widdth={}'.format(fheight, fwidth))
    # Check if image is loaded fine
    if src is None:
        print('Error opening image!')
        print('Usage: hough_lines.py [image_name -- default ' + default_file + '] \n')
        return -1

    horizontal_lines, vertical_lines = node_det.houghTransform(src)
    frameDebug, nodes, _ = node_det.findNodes(frameDubug, fheight, fwidth, horizontal_lines, vertical_lines)

    for node in nodes:
        cv2.circle(frame, (node[0], node[1]), 5, (0,255,0), -1)

    if not myconfig.DEBUG_FLAG:
        cv2.imshow('Source', frame)
    else:
        cv2.imshow('Debug', frameDebug)
    # cv.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
    # cv.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)

    cv2.waitKey()
    return 0


if __name__ == "__main__":
    main(sys.argv[1:])
