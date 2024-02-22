import cv2
import numpy as np
import matplotlib.pyplot as plt

def imread_photo(filename, flags=cv2.IMREAD_COLOR):
    return cv2.imread(filename, flags)

def resize_photo(imgArr,MAX_WIDTH = 1000):
    img = imgArr
    rows, cols= img.shape[:2]
    if cols >  MAX_WIDTH:
        change_rate = MAX_WIDTH / cols
        img = cv2.resize(img ,( MAX_WIDTH ,int(rows * change_rate) ), interpolation = cv2.INTER_AREA)
    return img


if __name__ == "__main__":
    img = imread_photo("dataset/Traindata/1.png")
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('img', img)
    #cv2.imshow('gray_img', gray_img)
    resized_img = resize_photo(gray_img)
    cv2.imshow('resized_img', resized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

