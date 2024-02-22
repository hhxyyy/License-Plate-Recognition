import cv2
import numpy as np
from matplotlib import pyplot as plt


def getpos(event,x,y,flags,param):
    if event==cv2.EVENT_LBUTTONDOWN:
        print(HSV[y,x])


if __name__ == "__main__":
    image = cv2.imread('dataset/Traindata/1.png')
    HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    cv2.imshow("imageHSV", HSV)
    cv2.imshow('image', image)
    cv2.setMouseCallback("imageHSV", getpos)
    cv2.waitKey(0)