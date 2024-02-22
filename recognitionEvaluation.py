import cv2

import plotFunction
import recognize
import SuperSlice

if __name__ == '__main__':
    allPlates = []
    results = []
    names = []
    p=0
    wrong = 0
    with open("dataset/dummy_plates/names.txt", "r") as f:  # 打开文件
        for line in f.readlines():
            line = line.strip('\n')  # 去掉列表中每一个元素的换行符
            names.append(line)
    for i in range(1, 20):
        print(str(i))
        path = "dataset/dummy_plates/" + str(i) + ".png"
        #print(path)
        plate = cv2.imread(path)
        plotFunction.plotImage(plate, str(i))
        outpath = './temp/'
        #print(plate)
        cut, mask = slice.slice(plate)
        name = "newName"
        cnt = 0
        for x, y, w, h in cut:
            cv2.imwrite(outpath + name + '_' + str(cnt) + '.png', slice.shrink(mask[y:y + h, x:x + w]))
            cnt += 1
        list = []
        for i in range(0, 6):
            path = "temp/newName_" + str(i) + ".png"
            img = cv2.imread(path)
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cv2.imshow('img', img)
            img = cv2.resize(img, (20, 40))
            cv2.imshow('gray_img', gray_img)
            list.append(gray_img)

        result = recognize.SVM(list)
        results.append(result)
        #if(result != names[p]):
            #print(result, names[p])
            #wrong += 1
        p+=1


    correct = 0
    for i in range(len(results)):
        if(names[i] == results[i]):
            correct = correct + 1
        else:
            print(names[i], results[i])
    print("accuracy", correct/len(results))

