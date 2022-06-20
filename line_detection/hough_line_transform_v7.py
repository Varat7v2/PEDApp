"""
@brief This program demonstrates line finding with the Hough transform
"""
import sys
import math
import cv2
import numpy as np
from sklearn.cluster import KMeans

import config as myconfig

class NODE_DETECTION():
    def __init__(self):
        pass

    def houghTransform(self, src, fwidth, fheight):
        # Detect edges
        dst = cv2.Canny(src, 50, 200, None, 3)

        # Copy edges to the images that will display the results in BGR
        cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
        cdstP = np.copy(cdst)

        if myconfig.HOUGH_TRANSFORM_TYPE == 'Standard':
            # Standard Hough Line Transform
            lines = cv2.HoughLines(dst, rho=1, theta=np.pi/180, threshold=myconfig.LINES_THERESHOLD, lines=None, srn=0, stn=0)
        elif myconfig.HOUGH_TRANSFORM_TYPE == 'Probabilistic':
            # Probabilistic Hough Line Transform
            lines = cv2.HoughLinesP(dst, rho=1, theta=np.pi/180, threshold=myconfig.LINES_THERESHOLD, lines=None, minLineLength=50, maxLineGap=10)

        print('[INFO]: Total no. of lines: ', len(lines))
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

                ### Separate out the perfect horizontal and vertical lines
                try:
                    slope = abs(float("inf") if (x1 == x2) else (y2 - y1) / (x2 - x1))
                    if slope >= 0 and slope < 1:
                        if myconfig.STRETCH_LINE: # Stretch horizontal line to the image width/height
                            horizontal_lines.append([(0, pt1[1]), (fwidth, pt2[1])])
                        else:
                            horizontal_lines.append([pt1, pt2])
                    elif (slope > 1 and slope < 1e10) or slope == float('inf'):
                        if myconfig.STRETCH_LINE:
                            vertical_lines.append([(pt1[0], 0), (pt2[0], fheight)])
                        else:
                            vertical_lines.append([pt1, pt2])
                except Exception as e:
                    print(e)

        return src, horizontal_lines, vertical_lines

    def findNodes(self, src, horizontal_lines, vertical_lines):
        if myconfig.DEBUG_FLAG:
            for hline, vline in zip(horizontal_lines, vertical_lines):
                cv2.line(src, hline[0], hline[1], (0,255,0), 1, cv2.LINE_AA)
                cv2.line(src, vline[0], vline[1], (240,16,255), 1, cv2.LINE_AA)
            print('[INFO]: No. of horizontal lines: ', len(horizontal_lines))
            print('[INFO]: No. of vertical lines: ', len(vertical_lines))

        ## My method (we can use here combination collections method)
        intersection_pts = list()
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

                ## Point of intersections - Calculation
                x, y = int((b1 * c2 - b2 * c1) / (a1 * b2 - a2 * b1)), int(((a2 * c1 - a1 * c2) / (a1 * b2 - a2 * b1)))
                intersection_pts.append((x,y))
                if myconfig.DEBUG_FLAG:
                    cv2.circle(src, (x, y), 3, (0, 0, 255), -1)

        print('[INFO]: No. of points of intersection: ', len(intersection_pts))

        ### KMeans Clustering for separating out the cluster groups
        kmeans = KMeans(n_clusters=myconfig.NODES_NUMBER, random_state=0).fit(np.array(intersection_pts))
        cluster_pts = kmeans.cluster_centers_
        ceny_avg = np.mean(cluster_pts[:,1])
        hline_down, hline_top = list(), list()

        for pt in cluster_pts:
            if pt[1] < ceny_avg:
                hline_top.append(pt.tolist())
            else:
                hline_down.append(pt.tolist())

        # Update hline groups to get nodes
        nodes_top   = sorted(hline_top,  key=lambda x:x[0])
        nodes_down  = sorted(hline_down, key=lambda x:x[0])

        myNodes = list()
        for idx, node in enumerate(nodes_top+nodes_down):
            node_num = idx+1
            cen_x, cen_y = int(node[0]), int(node[1])
            myNodes.append([cen_x, cen_y])
            if myconfig.DEBUG_FLAG:
                # cv2.drawMarker(src, (cen_x, cen_y), (255,98,41), markerType=cv2.MARKER_TILTED_CROSS,
                #                markerSize=20, thickness=3, line_type=cv2.LINE_AA)
                cv2.circle(src, (cen_x, cen_y), 50, (255,255,1), 2)
                cv2.putText(src, str(node_num), (cen_x-30, cen_y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        return src, myNodes

def main():
    node_det = NODE_DETECTION()

    img = cv2.imread(myconfig.SOURCE)
    fheight, fwidth = img.shape[:2]
    print('[INFO]: Img height={}, Img width={}'.format(fheight, fwidth))

    # Check if image is loaded fine
    if img is None:
        print('Error opening image!')
        print('Usage: hough_lines.py [image_name -- default ' + img + '] \n')
        return -1

    ### Lines detection using HOUGHTRANSFORM
    frameDebug, horizontal_lines, vertical_lines = node_det.houghTransform(img, fwidth, fheight)

    ### Node detection using Point-of-Intersection method
    if len(horizontal_lines)>0 and len(vertical_lines)>0:
        frameDebug, nodes = node_det.findNodes(img, horizontal_lines, vertical_lines)
    else:
        print('[ERROR]: Horizontal and vertical lines are not detected!')
        sys.exit(0)

    for node in nodes:
        cv2.circle(frameDebug, (node[0], node[1]), 10, (0,255,0), -1)

    if not myconfig.DEBUG_FLAG:
        cv2.imshow('Source', frameDebug)
    else:
        cv2.imshow('Debug', frameDebug)

    cv2.waitKey()
    return 0


if __name__ == "__main__":
    main()
