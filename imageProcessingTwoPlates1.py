import cv2
import numpy as np

import imageProcessing

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
    low_hsv = np.array([10, 80, 80])
    high_hsv = np.array([90, 255, 255])

    mask = cv2.inRange(hsv, lowerb=low_hsv, upperb=high_hsv)

    res = cv2.bitwise_and(img_copy, img_copy, mask=mask)



    return res


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


def draw_contours(img, contours):
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        rect = cv2.minAreaRect(c)

        box = cv2.boxPoints(rect)

        box = np.int0(box)

        cv2.drawContours(img, [box], 0, (0, 255, 0), 3)




def chose_licence_plate(contours, Min_Area=500): #area = 2000
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
            rect_vertices = np.int0(rect_vertices)


    return car_plate1


def license_segment(car_plates, out_path, img):
    i = 0
    if len(car_plates) <= 2:
        for car_plate in car_plates:
            row_min, col_min = np.min(car_plate[:, 0, :], axis=0)
            row_max, col_max = np.max(car_plate[:, 0, :], axis=0)
            cv2.rectangle(img, (row_min, col_min), (row_max, col_max), (0, 255, 0), 2)
            card_img = img[col_min:col_max, row_min:row_max, :]


            rotated_img = imageProcessing.rotate(card_img)


            if(rotated_img is not None):
            # if (False):
                final_img = imageProcessing.cutOut(rotated_img, 0.8)
                # final_img = rotated_img
                w = (int) (final_img.shape[0]*1.5)
                h = (int) (final_img.shape[1]*1.1)
                final_img = cv2.resize(final_img, ( h, w))

                cv2.imwrite(out_path + "/card_img" + str(i) + ".jpg", final_img)

                i += 1
                cv2.waitKey(0)
                cv2.destroyAllWindows()


                return final_img
            else:
                cv2.imwrite(out_path + "/card_img" + str(i) + ".jpg", card_img)

                i += 1
                cv2.waitKey(0)
                cv2.destroyAllWindows()


                return card_img


def imgProTwoImages(img):

    w = (int) (img.shape[0]/2)
    h = (int) (img.shape[1]/2)
    img_half = img[:, :h]
    img_half2 = img[:, h:]

    car_list = []
    for img in [img_half, img_half2]:


        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv_img = hsv_color_find(img)
        hsv_img = resize_keep_aspectratio(hsv_img, [500, 500])

        img = resize_keep_aspectratio(img, [500, 500])

        gray_img = resize_keep_aspectratio(gray_img, [500, 500])

        gray_img_, contours, contours2 = predict(img)


        draw_contours(gray_img, contours2)

        car_plate = chose_licence_plate(contours2)

        if len(car_plate) == 0:
            print('Not recognized, end!')
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:

            car_img_path = license_segment(car_plate, "temp", img)
            car_list.append(car_img_path)

    return car_list