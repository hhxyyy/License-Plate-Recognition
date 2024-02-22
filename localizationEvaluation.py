import pandas as pd
import argparse
import numpy as np
import math
import plotFunction
import cv2

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, default="dataset/")
    parser.add_argument('--ground_truth_path', type=str, default="dataset/Labels/")
    args = parser.parse_args()
    return args

def drawPoint(x, y, image, test):

    for p in np.arange(-5, 5):
        for q in np.arange(-5, 5):
            if (test):
                
                image[x+p][y+q] = (255, 0, 0)
            else:
                
                image[x+p][y+q] = (0, 255, 0)
                
    return image

#This is the evaluation function specially for testing the performance of the localiztion function, which is the 
#-main part of poster1.
if __name__ == '__main__':
    args = get_args()
    student_results = pd.read_csv(args.file_path + "TestLabels.csv")
    ground_truth = pd.read_csv(args.ground_truth_path+ "RefLabels.csv")
    totalInput = len(student_results['License plate'])
    totalPlates = len(ground_truth['License plate'])
    result = np.zeros((totalPlates, 4))
    data = np.zeros((totalInput, 8))
    ref = np.zeros((totalInput, 8))
    centralsData = np.zeros((totalInput, 2))
    centralsRef = np.zeros((totalInput, 2))
	

    # For each line in the input list
    for i in range(totalInput):
        licensePlateT = student_results['License plate'][i]
        data[i][0] = student_results['x1'][i]
        data[i][1] = student_results['y1'][i]
        data[i][2] = student_results['x2'][i]
        data[i][3] = student_results['y2'][i]
        data[i][4] = student_results['x3'][i]
        data[i][5] = student_results['y3'][i]
        data[i][6] = student_results['x4'][i]
        data[i][7] = student_results['y4'][i]
        centralsData[i][0] = (data[i][0] + data[i][2] + data[i][4] + data[i][6])/4
        print(centralsData[i][0])
        centralsData[i][1] = (data[i][1] + data[i][3] + data[i][5] + data[i][7])/4
        print(centralsData[i][1])
        #print(centralsData[i][0], centralsData[i][1])
        licensePlateR = student_results['License plate'][i]
        ref[i][0] = ground_truth['x1'][i]
        ref[i][1] = ground_truth['y1'][i]
        ref[i][2] = ground_truth['x2'][i]
        ref[i][3] = ground_truth['y2'][i]
        ref[i][4] = ground_truth['x3'][i]
        ref[i][5] = ground_truth['y3'][i]
        ref[i][6] = ground_truth['x4'][i]
        ref[i][7] = ground_truth['y4'][i]
        centralsRef[i][0] = (ref[i][0] + ref[i][2] + ref[i][4] + ref[i][6])/4
        
        centralsRef[i][1] = (ref[i][1] + ref[i][3] + ref[i][5] + ref[i][7])/4
        #print(centralsRef[i][0], centralsRef[i][1])

    #print(centralsData)
	# Initialize arrays to save the final results per category
    dists = np.zeros(totalInput)
    centralDists = np.zeros(totalInput)

    print('---------------------------------------------------------')
    print('%10s'%'License plate Localization Result')

    for i in np.arange(totalInput):
        img = cv2.imread("dataset/Testdata/" + str(i+1) + ".png")
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = drawPoint(int(centralsData[i][0]), int(centralsData[i][1]), image, True)
        image = drawPoint(int(centralsRef[i][1]), int(centralsRef[i][0]), image, False)
        print(image.shape)
        plotFunction.plotImage(image, str(i+1))
        dists[i] = math.dist((data[i][0], data[i][1]), (ref[i][0], ref[i][1])) + math.dist((data[i][2], data[i][3]), (ref[i][2], ref[i][3])) + math.dist((data[i][4], data[i][5]), (ref[i][4], ref[i][5])) + math.dist((data[i][6], data[i][7]), (ref[i][6], ref[i][7]))
        print(str(i+1) + ". The distance of all four corners are: " + str(dists[i]))
        centralDists[i] = math.dist((centralsData[i][0], centralsData[i][1]), (centralsRef[i][0], centralsRef[i][1]))
        print("   The distance of the central is: " + str(centralDists[i]))
        
    print("The average corner error is: " + str(np.mean(dists)))
    print("The average central error is: " + str(np.mean(centralDists)))
    #average level: ~200 good, could be recognized
    #               ~500 not bad, but can't be recognized with higher than 80% accuracy
    #               ~1000+ bad, can't pass the course