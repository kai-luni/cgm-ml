import cv2
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, filename='pose_prediction.log',
                    format='Date-Time : %(asctime)s : Line No. : %(lineno)d - %(message)s', filemode='w')


def opencv_hull_demo(img_path):
    img1 = cv2.imread(img_path)
    contourtwo = np.array([[50, 50], [50, 80], [100, 50], [100, 80]])
    # hulltwo = contourtwo
    hulltwo = cv2.convexHull(contourtwo)
    cv2.fillPoly(img1, pts=[hulltwo], color=(255, 0, 0))
    cv2.imshow(" ", img1)
    cv2.waitKey()


def opencv_polyfill_demo():
    # contours = np.array( [ [50,50], [50,90], [80, 90], [80,50] ] )
    # contours = np.array( [ [50,50], [50,90], [80,50], [80, 90] ])
    contours = np.array([[[50, 50]], [[50, 90]], [[80, 50]], [[80, 90]]])
    img = np.zeros((200, 200))  # create a single channel 200x200 pixel black image
    cv2.fillPoly(img, pts=[contours], color=(255, 255, 255))
    cv2.imshow(" ", img)
    cv2.waitKey()


def opencv_convex_hull_demo(img_path):
    # Load the image
    img = cv2.imread(img_path)
    # Convert it to greyscale
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Threshold the image
    ret, thresh = cv2.threshold(img_grey, 50, 255, 0)
    # Find the contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # For each contour, find the convex hull and draw it
    # on the original image.
    logging.info("%s %s", "len of contours ", len(contours))
    contour = contours[0]
    # logging.info("%s %s", "len of contours ", len(contour))
    logging.info("%s %s", "contour", contour)
    logging.info("%s %s", "contour.shape ", contour.shape)

    hullone = cv2.convexHull(contour)
    # contourtwo = np.array([[[0, 0]], [[0, 269]], [[479, 269]], [[479, 0]]])
    # contourtwo = np.array([[[0, 0]], [[0, 269]], [[479, 0]], [[479, 269]]])
    # contourtwo = np.array([[[50, 50]], [[50, 80]], [[100, 50]], [[100, 80]]])
    contourtwo = np.array([[50, 50], [50, 80], [100, 50], [100, 80]])
    hulltwo = cv2.convexHull(contourtwo)

    logging.info("%s %s", "hullone shape ", hullone.shape)
    logging.info("%s %s", "hulltwo shape ", hulltwo.shape)

    logging.info("%s %s", "hullone ", hullone)
    logging.info("%s %s", "hulltwo ", hulltwo)

    # assert (hullone == hulltwo), 'Not Equal'
    # a = [[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.5]]
    # hull = cv2.convexHull(np.array(a,dtype='float32'))

    # logging.info("%s %s", "hull ", hull)
    # cv2.drawContours(img1, [hulltwo], -1, (255, 0, 0), 2)

    img = cv2.imread(img_path)
    cv2.fillPoly(img, pts=[hulltwo], color=(255, 0, 0))

    # Display the final convex hull image
    cv2.imshow('ConvexHull', img)
    cv2.waitKey(0)
