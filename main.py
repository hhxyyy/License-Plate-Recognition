import argparse
import imageProcessing


# define the required arguments: video path(file_path), sample frequency(second), saving path for final result table
# for more information of 'argparse' module, see https://docs.python.org/3/library/argparse.html
def get_args():
	parser = argparse.ArgumentParser()
	#parser.add_argument('--file_path', type=str, default='TrainingVideo.avi')
	parser.add_argument('--file_path', type=str, default="dataset/Testdata/")
	parser.add_argument('--output_path', type=str, default=None)
	parser.add_argument('--sample_frequency', type=int, default=0.1)
	args = parser.parse_args()
	return args

# In this file, you need to pass three arguments into CaptureFrame_Process function.
if __name__ == '__main__':
	path = "dataset/trainingsvideo.avi"
	imageProcessing.imageProcessing(path)
