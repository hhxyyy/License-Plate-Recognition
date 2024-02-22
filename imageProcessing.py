import glob

import cv2
import numpy as np

import SuperSlice
import frameCapturing
import imageProcessingTwoPlates1
import recognize
import csv

from hypenAdd import hypenAdd


def imread_photo(filename, flags=cv2.IMREAD_COLOR):
    return cv2.imread(filename, flags)

def getLength(x,xx,y,yy):
    if x > y:
        x,xx,y,yy = y,yy,x,xx
    if xx < y:
        return 0
    else:
        return min(xx,yy) - y

def resize_keep_aspectratio(image_src, dst_size):
    src_h, src_w = image_src.shape[:2]
    dst_h, dst_w = dst_size

    h = dst_w * (float(src_h) / src_w)
    w = dst_h * (float(src_w) / src_h)

    h = int(h)
    w = int(w)

    if h <= dst_h:
        image_dst = cv2.resize(image_src, (dst_w, int(h)))
    else:
        image_dst = cv2.resize(image_src, (int(w), dst_h))

    h_, w_ = image_dst.shape[:2]

    return image_dst

def binarize(img):
    grayScaleImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurring = cv2.blur(grayScaleImg, (3, 3))
    res = cv2.equalizeHist(blurring)
    return res

def noiseCanceling(img):
    blured = cv2.blur(img, (3, 3))
    equalization = cv2.equalizeHist(blured)
    ret, thresh = cv2.threshold(equalization, 127.5, 255, cv2.THRESH_BINARY)
    return ret, thresh

def resize_photo(imgArr, MAX_WIDTH=1000):
    img = imgArr
    rows, cols = img.shape[:2]
    if cols > MAX_WIDTH:
        change_rate = MAX_WIDTH / cols
        img = cv2.resize(img, (MAX_WIDTH, int(rows * change_rate)), interpolation=cv2.INTER_AREA)
    return img

def adaptiveThresholding(img):
    res = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    res = 255 - res
    return res

def hsv_color_find(img):
    img_copy = img.copy()
    hsv = cv2.cvtColor(img_copy, cv2.COLOR_BGR2HSV)
    low_hsv = np.array([6, 120, 80])
    high_hsv = np.array([40, 235, 255])
    mask = cv2.inRange(hsv, lowerb=low_hsv, upperb=high_hsv)
    res = cv2.bitwise_and(img_copy, img_copy, mask=mask)
    return res

def findingEdge(threshold):
    kernel = np.zeros((3, 3), np.uint8)
    kernel[1, :] = 1
    kernel[:, 1] = 1
    res = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
    edges = cv2.Canny(res, 100, 200)
    return edges

def hough(edges):
    houghResult = cv2.HoughLinesP(edges, 1, np.pi / 180, 40, minLineLength=40, maxLineGap=10)
    return houghResult

def predict(imageArr):
    img_copy = imageArr.copy()
    img_copy = hsv_color_find(img_copy)
    gray_img = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    gray_img_ = cv2.GaussianBlur(gray_img, (5, 5), 0, 0, cv2.BORDER_DEFAULT)
    kernel = np.ones((23, 23), np.uint8)
    img_opening = cv2.morphologyEx(gray_img, cv2.MORPH_OPEN, kernel)
    img_opening = cv2.addWeighted(gray_img, 1, img_opening, -1, 0)
    ret, img_thresh = cv2.threshold(img_opening, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ret2, img_thresh2 = cv2.threshold(img_opening, 0, 255, cv2.THRESH_BINARY)
    img_edge = cv2.Canny(img_thresh, 100, 200)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    img_edge1 = cv2.morphologyEx(img_edge, cv2.MORPH_CLOSE, kernel)
    img_edge2 = cv2.morphologyEx(img_edge1, cv2.MORPH_OPEN, kernel)
    img_edge3 = cv2.morphologyEx(img_thresh2, cv2.MORPH_CLOSE, kernel)
    img_edge4 = cv2.morphologyEx(img_edge3, cv2.MORPH_CLOSE, kernel)
    contours, hierarchy = cv2.findContours(img_edge2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours2, hierarchy2 = cv2.findContours(img_edge4, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return gray_img_, contours, contours2

def rotate(img):
    binarized = binarize(img)
    threshold = adaptiveThresholding(binarized)
    edges = findingEdge(threshold)
    width, height = edges.shape
    lines = hough(edges)
    if lines is None:
        return None
    center = (height / 2, width / 2)
    angles = []
    #reference: https://www.cnblogs.com/luofeel/p/9150968.html
    for line in lines:
        x, y, xx, yy = line[0]
        if (xx - x) != 0:
            angle = np.arctan((yy - y) / (xx - x))
            if angle > -np.pi * 0.25 and angle < np.pi * 0.25:
                angles.append(angle)
    rotationMatrix = cv2.getRotationMatrix2D(center, np.mean(angles) * 180 / np.pi, 1.0)
    res = cv2.warpAffine(img, rotationMatrix, (height, width), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))

    return res

def draw_contours(img, contours):
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(img, [box], 0, (0, 255, 0), 3)

def findContours(threshold):
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    return contours

def findSoble(grayScaleImg, mask):
    blur = cv2.blur(grayScaleImg, (3, 3))
    sobelXAxis = cv2.convertScaleAbs(cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3))
    res = cv2.bitwise_and(sobelXAxis, mask)
    return res

def chose_licence_plate(contours, Min_Area=1250): #area = 2000
    temp_contours = []
    for contour in contours:
        if cv2.contourArea(contour) > Min_Area:
            temp_contours.append(contour)
    car_plate1 = []

    for temp_contour in temp_contours:
        rect_tupple = cv2.minAreaRect(temp_contour)
        rect_width, rect_height = rect_tupple[1]
        if rect_width < rect_height:
            rect_width, rect_height = rect_height, rect_width
        aspect_ratio = rect_width / rect_height
        if aspect_ratio > 3 and aspect_ratio < 6.65:
            car_plate1.append(temp_contour)
            rect_vertices = cv2.boxPoints(rect_tupple)
    return car_plate1


def cutOut(img, ratio):
    grayScaleImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = cv2.adaptiveThreshold(grayScaleImg, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 4)

    sobleX = findSoble(grayScaleImg, mask)

    _, threshold = noiseCanceling(sobleX)

    contours = findContours(threshold)

    #reference: https://huaweicloud.csdn.net/63807faedacf622b8df89120.html?spm=1001.2101.3001.6661.1&utm_medium=distribute.pc_relevant_t0.none-task-blog-2~default~BlogCommendFromBaidu~activity-1-78739790-blog-74322285.pc_relevant_vip_default&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-2~default~BlogCommendFromBaidu~activity-1-78739790-blog-74322285.pc_relevant_vip_default&utm_relevant_index=1
    boundingRect = None
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        summation = np.sum(threshold[y : y + height, x : x + width] == 255)
        limit = 0.4 * width * height
        if summation > limit:
            boundingRect = contour
            break
    x, y, width, height = cv2.boundingRect(boundingRect)
    if boundingRect is None:
        return None
    for contour in contours:
        xx, yy, newWidth, newHight = cv2.boundingRect(contour)
        if getLength(y, y + height, yy, yy + newHight) > ratio * newHight:
            xPrevious = x
            yPrevious = y
            x = min(x, xx)
            y = min(y, yy)
            width = max(xPrevious + width, xx + newWidth) - x
            height = max(yPrevious + height, yy + newHight) - y
    if width < 30 or height < 10:
        return None
    if width > 10 * height:
        return None
    cutOutImg = img[y : y + height, x : x + width]
    return cutOutImg

def license_segment(car_plates, out_path, img):
    i = 0
    if len(car_plates) <= 2:
        for car_plate in car_plates:
            row_min, col_min = np.min(car_plate[:, 0, :], axis=0)
            row_max, col_max = np.max(car_plate[:, 0, :], axis=0)

            card_img = img[col_min:col_max, row_min:row_max, :]
            rotated_img = rotate(card_img)
            if (rotated_img is None):
                cv2.imwrite(out_path + "/card_img" + str(i) + ".jpg", card_img)
                i += 1
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                return card_img
            else:
                final_img = cutOut(rotated_img, 0.8)
                # print(final_img.shape)
                if(final_img is None):
                    return None
                cv2.imwrite(out_path + "/card_img" + str(i) + ".jpg", final_img)
                i += 1
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                return final_img

def sliceAndRecognize(img):
    cut, mask = SuperSlice.slice(img)
    name = "newName"
    cnt = 0
    for x, y, w, h in cut:
        cv2.imwrite('./temp/' + name + "_" + str(cnt) + ".png", SuperSlice.shrink(mask[y:y + h, x:x + w]))
        cnt += 1

    list = []
    files = glob.glob("temp/*")
    length = len(files)
    for i in range(0, length - 1):
        path = "temp/newName_" + str(i) + ".png"
        img = cv2.imread(path)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (20, 40))
        list.append(gray_img)

    result = recognize.SVM(list)
    return result

def imageProcessing(path):
    import time
    import os
    time_start = time.time()
    FPS = frameCapturing.capture(path)
    results = []
    filePath = 'capture_image/'
    files = os.listdir(filePath)
    files.sort(key = lambda x: int(x[:-4]))
    counter = 0
    for fileName in files:
        img = imread_photo(filePath+fileName)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = resize_keep_aspectratio(img, [500, 500])

        gray_img = resize_keep_aspectratio(gray_img, [500, 500])

        gray_img_, contours, contours2 = predict(img)

        draw_contours(gray_img, contours2)

        car_plate = chose_licence_plate(contours2)

        if len(car_plate) == 0:
            print('Nothing recognized.')
        else:
            car_img = license_segment(car_plate, "temp", img)
        outpath = './temp/'
        img = cv2.imread("temp/card_img0.jpg")
        if (img is None):
            continue
        result = sliceAndRecognize(img)
        if (result is not None):
            prefix = fileName.split(".")[0]
            final = [result,  int(prefix) - 1, int(prefix) / FPS]
            results.append(final)
        counter += 1
    results_without_hypen = []
    for i in range(len(results)):
        str1 = results[i][0].replace('-', '')
        results_without_hypen.append(str1)
    #print("results", results_without_hypen)
    index = getDiffIndex(results_without_hypen)
    final_result = []
    #print("index",index)
    last = 0
    for ind in index:
        temp = results_without_hypen[last:ind+1]
        temp1 = results[last:ind+1]
        unique_elements, counts_elements = np.unique(temp, return_counts=True)
        chara = unique_elements[np.argmax(counts_elements)]
        chara = hypenAdd(chara)
        prefixs = []
        for i in range(len(temp1)):
            prefixs.append(int(temp1[i][1]))
        prefixLen = np.max(prefixs) - np.min(prefixs)
        prefixMax = np.max(prefixs)
        prefixMe = np.median(prefixs)
        prefix = (prefixMe + prefixLen/2.0 * 0.25)
        final_result.append([chara, prefix, prefix/FPS])
        last = ind
    #print(final_result)
    temp = ["License plate", "Frame no.", "Timestamp(seconds)"]
    with open("result.csv", "w", newline ='') as csvfile:
        writer = csv.writer(csvfile)

        writer.writerow(temp)

        writer.writerows(final_result)

    time_end = time.time()
    print('time cost', time_end - time_start, 's')

def difference(str1, str2):
    count = 0
    if(len(str1) < len(str2)):
        minlen = len(str1)
    else:
        minlen = len(str2)
    for i in range(minlen):
        if(str1[i] != str2[i]):
            count = count + 1
    if(count >= 3):
        return True
    return False

def getDiffIndex(results):
    index = []
    for i in range(1, len(results)):
        if(difference(results[i], results[i-1])):
            index.append(i-1)
    return index
