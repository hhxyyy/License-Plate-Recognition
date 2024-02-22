# import matplotlib for data visualisation
import matplotlib.pyplot as plt

# import NumPy for better matrix support
import numpy as np

# import Pickle for data serialisation
import pickle as pkl

# import cv2 and imutils for image processing functionality
import cv2

yellowMin = np.array([10, 100, 100])
yellowMax = np.array([30, 255, 255])

"""
In this file, you need to define plate_detection function.
To do:
	1. Localize the plates and crop the plates
	2. Adjust the cropped plate images
Inputs:(One)
	1. image: captured frame in CaptureFrame_Process.CaptureFrame_Process function
	type: Numpy array (imread by OpenCV package)
Outputs:(One)
	1. plate_imgs: cropped and adjusted plate images
	type: list, each element in 'plate_imgs' is the cropped image(Numpy array)
Hints:
	1. You may need to define other functions, such as crop and adjust function
	2. You may need to define two ways for localizing plates(yellow or other colors)
"""

def plate_detection(image):
    imgGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = colorSegmentation(image)
    plate, plateCoordinate, imgMo = cropping(image, mask)
    return plate, plateCoordinate, imgMo


def contrastImprovementHistogramEqualization(img, title=None):
    shape = img.shape
    # Create your own histogram array (a suggested function is np.bincount())
    histogram, bin_edges = np.histogram(img, bins=256, range=(0, 256))
    # Normalize your histogram
    histogram = histogram / np.sum(histogram)
    # Convert to cumulative sum histogram
    cumulativeHistogramArray = np.zeros(len(histogram))
    for i in range(len(histogram)):
        for j in range(i):
            cumulativeHistogramArray[i] += histogram[j]
    # Creating a mapping lookup table
    transformMap = np.floor(255 * cumulativeHistogramArray).astype(np.uint8)
    # Flatten image into 1D list
    imgList = img.reshape(-1)
    # Transform pixel values to equalize
    imgTransform = transformMap[imgList]
    # Write back into an image
    img = imgTransform.reshape(shape)
    return img


def colorSegmentation(image, colorMin=yellowMin, colorMax=yellowMax):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(img, colorMin, colorMax)
    return mask


def cropping(img, mask):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i, j] = img[i, j] * mask[i, j]
    # plotImage(img, "masked", "hsv")
    imgMorphological = morphological(img)
    #plotFunction.plotImageAndHistogram(img, "plate", 'hsv')
    # indices = np.where(img != 0)
    indices = np.where(imgMorphological != 0)
    if (len(indices[0]) == 0):
        indices = np.where(img != 0)
    xs = np.min(indices[0])
    xe = np.max(indices[0])
    ys = np.min(indices[1])
    ye = np.max(indices[1])
    # print(xs, xe, ys, ye)
    # Crop the plate
    plate = img[xs:xe, ys:ye]
    plateMorphological = imgMorphological[xs:xe, ys:ye]
    # Plot the cropped image
    # plotImage(plate, "plate")
    # plotFunction.plotImageAndHistogram(plateMorphological, "plate", 'gray')
    plateCoordinate = [xs, ys, xe, ys, xe, ye, xs, ye]
    return plate, plateCoordinate, plateMorphological


def morphological(img):
    imgGray = img[:, :, 2]
    # plotImage(imgGray, "Graymasked", "gray")
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    p1 = cv2.morphologyEx(imgGray, cv2.MORPH_OPEN, kernel)
    # plotFunction.plotImageAndHistogram(p1, 'OpeningHH', 'gray')
    binCount = np.bincount(p1.flatten())
    if (binCount[0] >= 20000):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        p1 = cv2.morphologyEx(p1, cv2.MORPH_OPEN, kernel)
        binCount = np.bincount(p1.flatten())

    return p1
