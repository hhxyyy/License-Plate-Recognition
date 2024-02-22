import cv2
import joblib
import numpy as np

import hypenAdd
import os
import glob

import sift


def cleanFolder(path):
    files = glob.glob(path)
    for f in files:
        os.remove(f)

def SVM(input):

    width = 20
    height = 40
    characters = np.zeros((len(input), width * height))
    for i in range(len(input)):
        character = cv2.resize(input[i], (width, height), interpolation=cv2.INTER_LINEAR)
        temp = character.reshape((1, width * height))[0]
        characters[i, :] = temp
    model = joblib.load("NewModel.m")
    if (len(characters) == 0):
        return None
    predictResult = model.predict(characters)
    allCharacters = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'B', 'D', 'F', 'G', 'H', 'J', 'K',
                    'L', 'M', 'N', 'P', 'R', 'S', 'T', 'V', 'X', 'Z']
    #print(predictResult.tolist())
    licensePlate = ''
    for k in range(len(predictResult.tolist())):
        licensePlate += allCharacters[predictResult.tolist()[k]]
    #print(licensePlate)
    if (len(licensePlate) != 6):
        cleanFolder("temp/*")
        return None
    res = hypenAdd.hypenAdd(licensePlate)
    if(res.count("-") != 2):
        siftResult = sift.runSift()
        cleanFolder("temp/*")
        if (siftResult is None):
            return res
        #print("SIFT:" + siftResult)
        return siftResult
    cleanFolder("temp/*")
    #print("SVM:" + res)
    return res

if __name__ == "__main__":
    list = []
    # siftResult = sift.runSift()
    # print("SIFT:" + siftResult)
    for  i in range(0,6):
        path = "temp/newName_" + str(i) + ".png"
        img = cv2.imread(path)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #cv2.imshow('img', img)
        img = cv2.resize(img, (20, 40))
        #plotFunction.plotImage(img, "ss")
        #cv2.imshow('gray_img', gray_img)
        list.append(gray_img)

    result = SVM(list)
    #print(result)
