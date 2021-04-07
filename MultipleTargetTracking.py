import cv2
import numpy as np
from os import listdir
from os.path import join, isfile
import random
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

class Blob:
    def __init__(self, bound_rect):
        self.centroid = None
        self.KF = None
        self.prediction = None
        self.bound_rect = bound_rect
        self.diagSize = np.sqrt(bound_rect[2]**2 + bound_rect[3]**2)
        self.color = (100 + random.randint(0, 155), 100 + random.randint(0, 155), 100 + random.randint(0, 155))
        self.foundMatchorIsNew = False
        self.numberOfConsecutiveMiss = 0
        self.stillBeingTracked = True
        self.history = []

    def setup(self):
        self.calcCentroid()
        self.setupKF()

    def calcCentroid(self):
        self.centroid = ((self.bound_rect[2]/2) + self.bound_rect[0], (self.bound_rect[3]/2) + self.bound_rect[1])

    def setupKF(self):
        kf = KalmanFilter(dim_x=4, dim_z=2)
        kf.x = np.array([self.centroid[0], self.centroid[1], 0., 1.])
        kf.F = np.array([[1, 0, 1, 0],
                        [0, 1, 0, 1],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]], np.float32)
        kf.H = np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0]], np.float32)
        kf.P *= 1000
        kf.R = 5
        kf.Q = Q_discrete_white_noise(dim=4, dt=0.1, var=0.13)

        self.KF = kf

    def predict(self):
        self.KF.predict()
        self.prediction = (self.KF.x[0], self.KF.x[1])

    def update(self, centroid):
        self.KF.update(centroid)

    def getIntCentroid(self):
        return (int(self.centroid[0]), int(self.centroid[1]))

    def addHistory(self, centroid):
            self.history.append(centroid)
    def drawHistory(self, image):
        if len(self.history) >= 3:
            prev = self.history[1]
            prev_position = prev.getIntCentroid()
            cv2.circle(image, prev_position, 2, self.color, 2)

            for i in range(2, len(self.history)):
                curr = self.history[i]
                curr_pos = curr.getIntCentroid()
                prev_position = prev.getIntCentroid()
                cv2.line(image, prev_position, curr_pos, self.color, 2)

                prev = curr
            if len(self.history) >= 10:
                self.history = self.history[3:]

def adaptive_threshold(img):
    adaptive_threshold = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 17, -5)
    
    return adaptive_threshold

def prepare_image(img, type):

    if type == "bats":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (19,19),0)
        thresh_output = adaptive_threshold(blur)
        kernel = np.ones((10,10), np.uint8)
        dialated = cv2.dilate(thresh_output, kernel, iterations=1)
        dialated = cv2.dilate(dialated, kernel, iterations=1)
        erode = cv2.erode(dialated, kernel, iterations=1)
        
        cv2.namedWindow("Segmentation", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("Segmentation", erode)

        return erode
    elif type == "cells":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (9,9),0)
        thresh_output = adaptive_threshold(blur)
        
        w,h = thresh_output.shape[0:2]
        roi = np.zeros((w,h), dtype="uint8")
        circle_img = cv2.circle(roi, (int(thresh_output.shape[0]/2)+18,int(thresh_output.shape[1]/2) - 40), 470, (255,255,255), -1)
        masked_data = cv2.bitwise_and(thresh_output, thresh_output, mask=circle_img)
        
        kernel = np.ones((5,5), np.uint8)
        erode = cv2.erode(masked_data, kernel, iterations=1)
        kernel2 = np.ones((10,10), np.uint8)
        dialated = cv2.dilate(erode, kernel2, iterations=1)
        dialated = cv2.dilate(dialated, kernel2, iterations=1)

        cv2.namedWindow("Segmentation", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("Segmentation", dialated)

        return dialated

def distance(point1, point2):
    x_dist = abs(point1[0]-point2[0])
    y_dist = abs(point1[1]-point2[1])
    dist = np.sqrt(x_dist**2 + y_dist**2)
    return dist


def matchCurrentFrameBlobs(existingBlobs, currFrameBlobs):
    for existingBlob in existingBlobs:
        existingBlob.foundMatchorIsNew = False
        existingBlob.predict()
    
    # Compare and assign
    for currBlob in currFrameBlobs:
        idxLeastDist = 0
        leastDist = 100000
        for i in range(len(existingBlobs)):
            if existingBlobs[i].stillBeingTracked:
                dist = distance(currBlob.centroid, existingBlobs[i].centroid)
                if dist < leastDist:
                    idxLeastDist = i 
                    leastDist = dist
        if leastDist < currBlob.diagSize * 1.15:
            existingBlobs[idxLeastDist].centroid = currBlob.centroid
            existingBlobs[idxLeastDist].KF.update(currBlob.centroid)
            existingBlobs[idxLeastDist].bound_rect = currBlob.bound_rect
            existingBlobs[idxLeastDist].diagSize = currBlob.diagSize
            existingBlobs[idxLeastDist].foundMatchorIsNew = True
            existingBlobs[idxLeastDist].stillBeingTracked = True
            existingBlobs[idxLeastDist].addHistory(currBlob)
        else:
            currBlob.foundMatchorIsNew = True
            existingBlobs.append(currBlob)
    for i in range(len(existingBlobs)):
        if existingBlobs[i].foundMatchorIsNew == False:
            existingBlobs[i].centroid = (existingBlobs[i].KF.x[0], existingBlobs[i].KF.x[1])
            existingBlobs[i].KF.update(existingBlobs[i].centroid)
            existingBlobs[i].addHistory(existingBlobs[i])
            existingBlobs[i].numberOfConsecutiveMiss += 1
        if existingBlobs[i].numberOfConsecutiveMiss > 10:
            existingBlobs[i].stillBeingTracked = False

def countCells(existingBlobs):
    trackedBlob = []
    for blob in existingBlobs:
        if blob.stillBeingTracked == True:
            trackedBlob.append(blob)

    
    return len(trackedBlob)

def putInfoImg(img, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (50,80)
    fontScale = 3
    color = (0,0,255)
    thickness = 2

    cv2.putText(img, text, org, font, fontScale, color, thickness, cv2.LINE_AA)


def main(type):

    if type == "cells":
        img_path = './CS585-Cells'
    elif type == "bats":
        img_path = './CS585-BatImages/FalseColor'
    # Get all files in folder
    files = [join(img_path, f) for f in listdir(img_path) if isfile(join(img_path, f))]
    img = []
    for i in range(len(files)):
        img.append(cv2.imread(files[i]))

    kf = None

    img_idx = 1

    existingBlobs = []

    blob_hist = 0

    count = 0
    prev_count = 0

    while img_idx < len(files):
        currFrameBlobs = []

        curr = img[img_idx]

        img_prep = prepare_image(curr, type)

        contours, hierarchy = cv2.findContours(img_prep, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        
        contour_output = img[img_idx][:,:,:].copy()

        # Get current frame blobs
        if(len(contours) > 0):
            for i in range(len(contours)):
                bound_rect = cv2.boundingRect(contours[i])
                new_blob = Blob(bound_rect)
                new_blob.setup()
                currFrameBlobs.append(new_blob)
                
                # cv2.circle(contour_output, new_blob.getIntCentroid(), 2, (0,0,255), 2)
                
        # Initialize all existing blobs for first frame
        if img_idx == 1:
            for blob in currFrameBlobs:
                existingBlobs.append(blob)

        # Compare currFrameBlobs to existingBlobs
        else:
            matchCurrentFrameBlobs(existingBlobs,currFrameBlobs)

        for blob in existingBlobs:
            if blob.stillBeingTracked == True:
                blob.drawHistory(contour_output)
                cv2.rectangle(contour_output, blob.bound_rect, blob.color, 1, 8, 0)
        
        if type == "cells":
            if img_idx > 5:
                if blob_hist == 10:
                    count = countCells(existingBlobs)
                    if count > prev_count:
                        print("Cell Birth")

                    prev_count = count
                    blob_hist = 0
                blob_hist += 1
            putInfoImg(contour_output, str(count))

        cv2.namedWindow("Display", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("Display", contour_output)
        if cv2.waitKey(100)&0xFF == 27:
            break
        img_idx += 1
    cv2.destroyAllWindows()

main("bats")