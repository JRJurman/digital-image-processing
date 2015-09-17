# required only for plotting
import matplotlib.pyplot as plt

def plotHist(histogram):
    """
    Function to display the histogram using the matplotlib library
    """

    plt.plot(histogram)
    plt.show()

def plotImgHist(im):
    """
    Function to display an image as a histogram using the matplotlib library
    """

    channels = []
    if (len(im.shape) == 2):
        # gray-scale image, look at channels [0]
        channels = [0]
    elif (len(im.shape) == 3):
        # color image, channel [0] - blue, [1] - green, [2] - red
        channels = [0,1,2]

    hist = cv2.calcHist([im],channels,None,[255],[0,255])
    plotHist(hist)
