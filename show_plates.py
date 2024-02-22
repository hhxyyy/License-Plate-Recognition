from Localization import plate_detection
import cv2 
import numpy as np

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, default='dummytestvideo.avi')
    args = parser.parse_args()
    return args



# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture('dummytestvideo.avi')

# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
    #dummy arguments for sample frequency and save_path should be changed
    detections = plate_detection(frame)
    # Display the resulting frame
    cv2.imshow('Frame',detections)

    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break

  # Break the loop
  else: 
    break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()



