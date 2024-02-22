import cv2
import os

def capture(video_path):
    timeF = 2
    images_path = "capture_image"
    if not os.path.exists(images_path):
        os.mkdir(images_path)

    vc = cv2.VideoCapture(video_path)
    FPS = vc.get(5)
    c = 1
    rat = 1
    if vc.isOpened():
        print('Reading successful! Capturing...')
        while rat:
            rat, frame = vc.read()
            if(c%timeF == 0 and rat == True):
                cv2.imwrite(images_path + '/' +  str(c) + '.jpg',frame)
            c = c + 1
        vc.release()
        print("Done!")
    else:
        print("Fail to read the video, please check the path given.")
    return FPS