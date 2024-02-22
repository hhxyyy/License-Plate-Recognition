import cv2
import numpy as np
import os
import SuperSlice
#morphological gradient
def morph_gradient(img):
    kernel = np.ones((3,3),np.uint8)
    gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
    return gradient
def loadGT(path = "character/"):
    files = os.listdir(path)
    GT = []
    numerical = []
    letter = []
    for file in files:
        name = file.split(".")[0]
        img = cv2.imread(path + file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        GT.append((name, img, sift_descriptor(img)))
        if name[0].isdigit():
            numerical.append(GT[-1])
        else:
            letter.append(GT[-1])
    return GT,numerical,letter

def getVerticalDiff(img,pattern):
    imgcopy = img.copy()
    #resize the img copy so that it has the same high with pattern but retain the aspect ratio
    imgcopy = cv2.resize(imgcopy, (int(imgcopy.shape[1]*pattern.shape[0]/imgcopy.shape[0]), pattern.shape[0]),interpolation=cv2.INTER_NEAREST)
    return np.linalg.norm(descriptorV(imgcopy) - descriptorV(pattern))
def getHorizontalDiff(img,pattern):
    imgcopy = img.copy()
    #resize the img copy so that it has the same width with pattern but retain the aspect ratio
    imgcopy = cv2.resize(imgcopy, (pattern.shape[1], int(imgcopy.shape[0]*pattern.shape[1]/imgcopy.shape[1])),interpolation=cv2.INTER_NEAREST)
    return np.linalg.norm(descriptorH(imgcopy) - descriptorH(pattern))

def descriptorV(img):
    #map the number of none-zero pixel to the vertical line
    w,h = img.shape
    des = np.zeros((w,))
    for i in range(w):
        des[i] = (np.count_nonzero(img[i,:])/h)
    return des
def descriptorH(img):
    #map the number of none-zero pixel to the vertical line
    w,h = img.shape
    des = np.zeros((h,))
    for i in range(h):
        des[i] = (np.count_nonzero(img[:,i])/w)
    return des

def verify(img,pattern):
    a = getHorizontalDiff(img,pattern)
    b = getVerticalDiff(img,pattern)
    return np.sqrt(a**2+b**2)

def sift_descriptor(image):
    result = np.zeros(128)
    # Make sure the size of the image_interest_patch is divisable by 16*16
    image = cv2.resize(image, (4 * (image.shape[1] // 4), 4 * (image.shape[0] // 4)))
    grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    magnitude, direction = cv2.cartToPolar(grad_x, grad_y)
    # Iterate over every pixel
    cnt = np.zeros(16,)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):

            indexa = int((direction[i,j]/2/np.pi)*8)
            indexb = int(int((i/image.shape[0])*4)*4 + int(j/image.shape[1]*4))
            # Add the direction of the edge to the feature vector, scaled by its magnitude
            result[indexb*8+indexa] += magnitude[i][j]
            cnt[indexb] += 1
    for i in range(16):
        result[i*8:(i+1)*8] /= cnt[i]
    return result

def bitwise_descriptor(image,pattern):
    #resize pattern to image
    patternResized = cv2.resize(pattern, (image.shape[1], image.shape[0]))
    return (image != patternResized).sum()/image.size


def sift_match(image,GT):
    shrinked = SuperSlice.shrink(image)
    if shrinked.shape[0] <4 or shrinked.shape[1] <4:
        return "", 1e9
    score = -1
    str = ""
    for name, img, des in GT:
        sc1 = np.linalg.norm(sift_descriptor(shrinked) - des)
        sc2 = verify(shrinked,img)
        sc3 = bitwise_descriptor(shrinked,img)
        sc = sc1*sc2*sc3
        #print(name,sc)
        if score == -1 or sc < score:
            score = sc
            str = name
    return str, score

def identify(img,GT,numerical,letter):
    boxes,mask = SuperSlice.slice(img)
    if len(boxes) < 6:
        return None
    s = []
    numericalResult = []
    letterResult = []
    nomatch = 0
    for x,y,w,h in boxes:
        nm, score = sift_match(mask[y:y+h,x:x+w],GT)
        if nm == "":
            nomatch += 1
        numericalResult.append(sift_match(mask[y:y+h,x:x+w],numerical))
        letterResult.append(sift_match(mask[y:y+h,x:x+w],letter))
        s.append(score)
    if len(boxes) - nomatch < 6:
        return None
    #get indexes of the 6 lowest score
    indexes = np.argsort(s)[:6]
    ans = ""
    lines = []
    space = []
    seq = []
    for i in range(len(s)):
        if i in indexes:
            seq.append(i)
            x,y,w,h = boxes[i]
            if len(lines) > 0:
                space.append(x - lines[-1][1])
            lines.append((x, x + w))
    #print(space)
    if space[0] + space[3] > max(space[1] + space[3],space[1] + space[4]): #2-XXX-80
        num = numericalResult[seq[1]][1] + numericalResult[seq[2]][1] + numericalResult[seq[3]][1]
        let = letterResult[seq[1]][1] + letterResult[seq[2]][1] + letterResult[seq[3]][1]
        if num > let:
            ans = numericalResult[seq[0]][0] + letterResult[seq[1]][0] + letterResult[seq[2]][0] + letterResult[seq[3]][0] + numericalResult[seq[4]][0] + numericalResult[seq[5]][0]
        else:
            ans = letterResult[seq[0]][0] + numericalResult[seq[1]][0] + numericalResult[seq[2]][0] + numericalResult[seq[3]][0] + letterResult[seq[4]][0] + letterResult[seq[5]][0]
        ans = ans[0] + "-" + ans[1:4] + "-" + ans[4:]
    elif space[1] + space[3] > max(space[0] + space[3],space[1] + space[4]): #47-SX-GX
        if numericalResult[seq[0]][1] + numericalResult[seq[1]][1] > letterResult[seq[0]][1] + letterResult[seq[1]][1]:
            ans += letterResult[seq[0]][0] + letterResult[seq[1]][0]
        else:
            ans += numericalResult[seq[0]][0] + numericalResult[seq[1]][0]
        if numericalResult[seq[2]][1] + numericalResult[seq[3]][1] > letterResult[seq[2]][1] + letterResult[seq[3]][1]:
            ans += letterResult[seq[2]][0] + letterResult[seq[3]][0]
        else:
            ans += numericalResult[seq[2]][0] + numericalResult[seq[3]][0]
        if numericalResult[seq[4]][1] + numericalResult[seq[5]][1] > letterResult[seq[4]][1] + letterResult[seq[5]][1]:
            ans += letterResult[seq[4]][0] + letterResult[seq[5]][0]
        else:
            ans += numericalResult[seq[4]][0] + numericalResult[seq[5]][0]
        ans = ans[0:2] + "-" + ans[2:4] + "-" + ans[4:]
    else: #29-KTV-7
        num = numericalResult[seq[2]][1] + numericalResult[seq[3]][1] + numericalResult[seq[4]][1]
        let = letterResult[seq[2]][1] + letterResult[seq[3]][1] + letterResult[seq[4]][1]
        if num > let:
            ans = numericalResult[seq[0]][0] + numericalResult[seq[1]][0] + letterResult[seq[2]][0] + letterResult[seq[3]][0] + letterResult[seq[4]][0] + numericalResult[seq[5]][0]
        else:
            ans = letterResult[seq[0]][0] + letterResult[seq[1]][0] + numericalResult[seq[2]][0] + numericalResult[seq[3]][0] + numericalResult[seq[4]][0] + letterResult[seq[5]][0]
        ans = ans[0:2] + "-" + ans[2:5] + "-" + ans[5]

    return ans.upper()

def runSift():

    filename = "temp/card_img0.jpg"
    GT,numerical,letter = loadGT()
    img = cv2.imread(filename)
    name = identify(img, GT, numerical, letter)
    return name
