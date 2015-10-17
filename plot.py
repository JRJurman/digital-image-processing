# required only for plotting
import matplotlib.pyplot as plt
import cv2
import numpy as np

def plotHist(histogram, verts=[]):
    """
    Function to display the histogram using the matplotlib library
    """

    plt.plot(histogram)

    # verts is a list of vertical lines to display
    for i in verts:
        plt.axvline(i)

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

def plotLetters(histogram, letters):
    pos = np.arange(len(letters))
    width = 1.0     # gives histogram aspect to the bar diagram

    ax = plt.axes()
    ax.set_xticks(pos + (width / 2))
    ax.set_xticklabels(letters)

    plt.bar(pos, histogram, width, color='r')
    plt.show()
