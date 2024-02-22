import cv2
import numpy as np

def floodFill(vis, mask, cnt, xx, yy):
    queue = [(xx,yy)]
    vis[yy,xx] = cnt
    while len(queue) > 0:
        x,y = queue.pop(0)
        if x > 0 and mask[y,x-1] > 0 and vis[y,x-1] == 0:
            vis[y,x-1] = cnt
            queue.append((x-1,y))
        if x < mask.shape[1]-1 and mask[y,x+1] > 0 and vis[y,x+1] == 0:
            vis[y,x+1] = cnt
            queue.append((x+1,y))
        if y > 0 and mask[y-1,x] > 0 and vis[y-1,x] == 0:
            vis[y-1,x] = cnt
            queue.append((x,y-1))
        if y < mask.shape[0]-1 and mask[y+1,x] > 0 and vis[y+1,x] == 0:
            vis[y+1,x] = cnt
            queue.append((x,y+1))

def find_connected_components(mask,image):
    vis = np.zeros(mask.shape, dtype=np.uint8)
    tot = 0
    box = []
    for i in range(mask.shape[1]):
        for j in range(mask.shape[0]):
            if vis[j,i] == 0 and mask[j,i] > 0:
                tot += 1
                floodFill(vis, mask, tot ,i,j)
                #print(vis.sum())
    ans = []
    rec = []
    for num in range(1,tot+1):
        mask2 = (vis == num)
        if mask2.sum() < 20:
            continue
        countour = cv2.findContours(mask2.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        if len(countour) == 0:
            continue
        countour = countour[0]
        x,y,w,h = cv2.boundingRect(countour)
        if w*6 > mask.shape[1] or h < mask.shape[0]*0.6 or w<mask.shape[1]*0.05:
            continue
        if mask2.sum() < w*h*0.3:
            continue
        if len(rec)> 15:
            return []
        rec.append((x,y,w,h))
    rec.sort(key=lambda x:x[0])
    for x,y,w,h in rec:
        ans.append(mask[y:y+h,x:x+w])
        box.append((x,y,w,h))
    return box

def shrink(img):
    mopho = img
    col = 0
    factor = 0.3
    while col < mopho.shape[1] - 1:
        diff = (mopho[:,col+1]  !=  mopho[:,col]).sum()
        if diff < mopho.shape[0]*factor:
            break
        col += 1
    mopho = mopho[:,col:]

    colend = mopho.shape[1] - 1
    while colend > 1:
        diff = (mopho[:,colend-1]  !=  mopho[:,colend]).sum()
        if diff < mopho.shape[0]*factor:
            break
        colend -= 1

    mopho = mopho[:,:colend+1]

    row = 0
    while row < mopho.shape[0] - 1:
        diff = (mopho[row+1,:]  !=  mopho[row,:]).sum()
        if diff < mopho.shape[1]*factor:
            break
        row += 1
    mopho = mopho[row:,:]
    rowend = mopho.shape[0] - 1
    while rowend > 1:
        diff = (mopho[rowend-1,:]  !=  mopho[rowend,:]).sum()
        if diff < mopho.shape[1]*factor:
            break
        rowend -= 1
    mopho = mopho[:rowend+1,:]
    return mopho

def slice(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 4)
    blured = cv2.GaussianBlur(gray,(5,5),0)
    bluredmask = cv2.adaptiveThreshold(blured, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 4)
    ans = find_connected_components(bluredmask,img)
    return ans,mask


if __name__ == '__main__':
    #inpath = "./test/cutOut/"
    outpath = './temp/'
    img = cv2.imread("temp/card_img0.jpg")
    cut, mask = slice(img)
    name = "newName"
    cnt = 0
    for x, y, w, h in cut:
        cv2.imwrite(outpath + name + "_" + str(cnt) + ".png", shrink(mask[y:y+h, x:x+w]))
        cnt += 1