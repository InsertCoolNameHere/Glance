import cv2
import numpy as np


def crop_irregular_polygon(points, filename):
    cnt = []
    for pnt in points:
        #print(pnt)
        cnt.append([[pnt[0], pnt[1]]])
    cnt = np.array(cnt)
    img = cv2.imread(filename)

    #print("shape of cnt: {}".format(cnt.shape))
    rect = cv2.minAreaRect(cnt)
    #print("rect: {}".format(rect))

    # the order of the box points: bottom left, top left, top right,
    # bottom right
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    #print("bounding box: {}".format(box))
    # cv2.drawContours(img, [box], 0, (0, 0, 255), 2)

    # get width and height of the detected rectangle
    width = int(rect[1][0])
    height = int(rect[1][1])

    src_pts = box.astype("float32")
    # coordinate of the points in box points after the rectangle has been
    # straightened
    dst_pts = np.array([[0, height - 1],
                        [0, 0],
                        [width - 1, 0],
                        [width - 1, height - 1]], dtype="float32")

    # the perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # directly warp the rotated rectangle to get the straightened rectangle
    warped = cv2.warpPerspective(img, M, (width, height))

    #cv2.imwrite("/s/chopin/e/proj/sustain/sapmitra/super_resolution/crop_img.jpg", warped)
    #cv2.waitKey(0)
    return warped


if __name__ == '__main__':

    img = cv2.imread("/s/chopin/e/proj/sustain/sapmitra/super_resolution/SRData_Quad/co_test/02310010132_20190604.tif")

    cnt = np.array([
        [[2, 43]], [[1490, 1]], [[1533, 1485]], [[42, 1526]]
    ])
    print("shape of cnt: {}".format(cnt.shape))
    rect = cv2.minAreaRect(cnt)
    print("rect: {}".format(rect))

    # the order of the box points: bottom left, top left, top right,
    # bottom right
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    print("bounding box: {}".format(box))
    #cv2.drawContours(img, [box], 0, (0, 0, 255), 2)

    # get width and height of the detected rectangle
    width = int(rect[1][0])
    height = int(rect[1][1])

    src_pts = box.astype("float32")
    # coordinate of the points in box points after the rectangle has been
    # straightened
    dst_pts = np.array([[0, height - 1],
                        [0, 0],
                        [width - 1, 0],
                        [width - 1, height - 1]], dtype="float32")

    # the perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # directly warp the rotated rectangle to get the straightened rectangle
    warped = cv2.warpPerspective(img, M, (width, height))

    cv2.imwrite("/s/chopin/e/proj/sustain/sapmitra/super_resolution/crop_img.jpg", warped)
    cv2.waitKey(0)
