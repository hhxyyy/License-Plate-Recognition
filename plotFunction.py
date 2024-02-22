# Some plotting functions to make your work easier
import matplotlib.pyplot as plt

# Displays a given RGB image using matplotlib.pyplot
def plotImage(img, title, cmapType=None):
    # Display image
    if (cmapType):
        plt.imshow(img, cmap=cmapType, vmin=0, vmax=255)
    else:
        plt.imshow(img, vmin=0, vmax=255)
    plt.title(title)
    plt.show()

# ! Use [img] that has only one channel !
# Displays the given image and its histogram below it
def plotImageAndHistogram(img, title, cmapType=None):
    # Display image
    if (cmapType):
        plt.imshow(img, cmap=cmapType, vmin=0, vmax=255)
    else:
        plt.imshow(img, vmin=0, vmax=255)
    plt.title(title)
    plt.show()
    # Display histogram
    plt.hist(img.flatten(), bins=256)
    plt.title(title)
    plt.show()