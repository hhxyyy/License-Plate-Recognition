import cv2
import numpy as np


def imread_photo(filename, flags=cv2.IMREAD_COLOR):

    return cv2.imread(filename, flags)


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


def resize_photo(imgArr, MAX_WIDTH=1000):
    img = imgArr
    rows, cols = img.shape[:2]
    if cols > MAX_WIDTH:
        change_rate = MAX_WIDTH / cols
        img = cv2.resize(img, (MAX_WIDTH, int(rows * change_rate)), interpolation=cv2.INTER_AREA)
    return img


def hsv_color_find(img):
    img_copy = img.copy()

    hsv = cv2.cvtColor(img_copy, cv2.COLOR_BGR2HSV)
    low_hsv = np.array([100, 80, 80])
    high_hsv = np.array([124, 255, 255])

    mask = cv2.inRange(hsv, lowerb=low_hsv, upperb=high_hsv)
    cv2.imshow("hsv_color_find", mask)

    res = cv2.bitwise_and(img_copy, img_copy, mask=mask)
    cv2.imshow("hsv_color_find2", res)


    return res



def predict(imageArr):

    img_copy = imageArr.copy()
    img_copy = hsv_color_find(img_copy)

    gray_img = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)

    gray_img_ = cv2.GaussianBlur(gray_img, (5, 5), 0, 0, cv2.BORDER_DEFAULT)
    kernel = np.ones((23, 23), np.uint8)

    img_opening = cv2.morphologyEx(gray_img, cv2.MORPH_OPEN, kernel)

    img_opening = cv2.addWeighted(gray_img, 1, img_opening, -1, 0)

    cv2.imshow("img_opening", img_opening)

    ret, img_thresh = cv2.threshold(img_opening, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ret2, img_thresh2 = cv2.threshold(img_opening, 0, 255, cv2.THRESH_BINARY)

    cv2.imshow("img_thresh", img_thresh)
    cv2.imshow("img_thresh2", img_thresh2)


    img_edge = cv2.Canny(img_thresh, 100, 200)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    img_edge1 = cv2.morphologyEx(img_edge, cv2.MORPH_CLOSE, kernel)
    img_edge2 = cv2.morphologyEx(img_edge1, cv2.MORPH_OPEN, kernel)
    img_edge3 = cv2.morphologyEx(img_thresh2, cv2.MORPH_CLOSE, kernel)
    img_edge4 = cv2.morphologyEx(img_edge3, cv2.MORPH_CLOSE, kernel)

    cv2.imshow("img_edge3", img_edge3)
    cv2.imshow("img_edge4", img_edge4)

    contours, hierarchy = cv2.findContours(img_edge2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours2, hierarchy2 = cv2.findContours(img_edge4, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    return gray_img_, contours, contours2


def draw_contours(img, contours):
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)


        rect = cv2.minAreaRect(c)

        box = cv2.boxPoints(rect)

        box = np.int0(box)

        cv2.drawContours(img, [box], 0, (0, 255, 0), 3)

    cv2.imshow("contours", img)


def chose_licence_plate(contours, Min_Area=2000):

    temp_contours = []
    for contour in contours:
        if cv2.contourArea(contour) > Min_Area:
            temp_contours.append(contour)
    car_plate1 = []
    car_plate2 = []
    car_plate3 = []
    for temp_contour in temp_contours:
        rect_tupple = cv2.minAreaRect(temp_contour)
        rect_width, rect_height = rect_tupple[1]
        if rect_width < rect_height:
            rect_width, rect_height = rect_height, rect_width
        aspect_ratio = rect_width / rect_height

        if aspect_ratio > 1.5 and aspect_ratio < 4.65:
            car_plate1.append(temp_contour)
            rect_vertices = cv2.boxPoints(rect_tupple)
            rect_vertices = np.int0(rect_vertices)

    if len(car_plate1) > 1:
        for temp_contour in car_plate1:
            rect_tupple = cv2.minAreaRect(temp_contour)
            rect_width, rect_height = rect_tupple[1]
            if rect_width < rect_height:
                rect_width, rect_height = rect_height, rect_width
            aspect_ratio = rect_width / rect_height

            if aspect_ratio > 1.6 and aspect_ratio < 4.15:
                car_plate2.append(temp_contour)
                rect_vertices = cv2.boxPoints(rect_tupple)
                rect_vertices = np.int0(rect_vertices)

    if len(car_plate2) > 1:
        for temp_contour in car_plate2:
            rect_tupple = cv2.minAreaRect(temp_contour)
            rect_width, rect_height = rect_tupple[1]
            if rect_width < rect_height:
                rect_width, rect_height = rect_height, rect_width
            aspect_ratio = rect_width / rect_height

            if aspect_ratio > 1.8 and aspect_ratio < 3.35:
                car_plate3.append(temp_contour)
                rect_vertices = cv2.boxPoints(rect_tupple)
                rect_vertices = np.int0(rect_vertices)


    if len(car_plate3) > 0:
        return car_plate3
    if len(car_plate2) > 0:
        return car_plate2
    return car_plate1


def license_segment(car_plates, out_path):

    i = 0
    if len(car_plates) == 1:
        for car_plate in car_plates:
            row_min, col_min = np.min(car_plate[:, 0, :], axis=0)
            row_max, col_max = np.max(car_plate[:, 0, :], axis=0)
            cv2.rectangle(img, (row_min, col_min), (row_max, col_max), (0, 255, 0), 2)
            card_img = img[col_min:col_max, row_min:row_max, :]
            cv2.imwrite(out_path + "/card_img" + str(i) + ".jpg", card_img)
            cv2.imshow("card_img" + str(i) + ".jpg", card_img)
            i += 1
            cv2.waitKey(0)
            cv2.destroyAllWindows()


    return out_path + "/card_img0.jpg"


def find_waves(threshold, histogram):
    up_point = -1
    is_peak = False
    if histogram[0] > threshold:
        up_point = 0
        is_peak = True
    wave_peaks = []
    for i, x in enumerate(histogram):
        if is_peak and x < threshold:
            if i - up_point > 2:
                is_peak = False
                wave_peaks.append((up_point, i))
        elif not is_peak and x >= threshold:
            is_peak = True
            up_point = i
    if is_peak and up_point != -1 and i - up_point > 4:
        wave_peaks.append((up_point, i))
    return wave_peaks


def remove_plate_upanddown_border(card_img):

    plate_Arr = cv2.imread(card_img)
    plate_gray_Arr = cv2.cvtColor(plate_Arr, cv2.COLOR_BGR2GRAY)
    ret, plate_binary_img = cv2.threshold(plate_gray_Arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    row_histogram = np.sum(plate_binary_img, axis=1)
    row_min = np.min(row_histogram)
    row_average = np.sum(row_histogram) / plate_binary_img.shape[0]
    row_threshold = (row_min + row_average) / 2
    wave_peaks = find_waves(row_threshold, row_histogram)

    wave_span = 0.0
    for wave_peak in wave_peaks:
        span = wave_peak[1] - wave_peak[0]
        if span > wave_span:
            wave_span = span
            selected_wave = wave_peak
    plate_binary_img = plate_binary_img[selected_wave[0]:selected_wave[1], :]
    cv2.imshow("plate_binary_img", plate_binary_img)

    return plate_binary_img
