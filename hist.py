"""
PYTHON METHOD DEFINITION
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt


def plotHist(histogram):
    """
    Function to display the histogram using the matplotlib library
    """

    plt.plot(histogram)
    plt.show()

def plotImgHist(im):
    channels = []
    if (len(im.shape) == 2):
        # gray-scale image, look at channels [0]
        channels = [0]
    elif (len(im.shape) == 3):
        # color image, channel [0] - blue, [1] - green, [2] - red
        channels = [0,1,2]

    hist = cv2.calcHist([im],channels,None,[255],[0,255])
    plotHist(hist)

def build_cdf(a, dcValues):
    """
    Function to get the cdf of an array

    Args:
        a (array): array to build the cdf from.
            If the shape is 1 dimensional, it is assumed to be a pdf
            If the shape is 2 dimensional, it is assumed to be a gray-scale img
            If the shape is 3 dimensional, it is assumed to be a color image
        dcValues (int): maximum value of any element in the array
            For images this will be 255

    Returns:
        a single-dimension array cdf
    """
    if (len(a.shape) == 1):
        # it's a pdf
        pdf = a
    else:
        # get histogram from the image
        # first check if image is gray-scale or not
        channels = []
        if (len(a.shape) == 2):
            # gray-scale image, look at channels [0]
            channels = [0]
        elif (len(a.shape) == 3):
            # color image, channel [0] - blue, [1] - green, [2] - red
            channels = [0,1,2]
        else:
            raise ValueError("Could not determine number of channels for the image")

        # images, channels, mask, histSize, ranges
        hist = cv2.calcHist([a],channels,None,[dcValues],[0,dcValues])

        # get PDF of histogram
        pdf = hist / np.prod(np.shape(a))

    # get CDF of histogram
    cdf = np.cumsum(pdf)

    return cdf


def build_linear_lookup_table(im, value, dcValues):
    """
    Function to build lookup table for a given image

    Args:
        im (array): image to build initial histogram from
        value (integer): percent of CDF to remove from the histogram
        dcValues (integer): total number of digital counts

    Returns:
        a lookup table to perform on a histogram

    Raises:
        ValueError: image shape is not 2 (gray-scale) or 3 (color)
    """

    # get the cdf
    cdf = build_cdf(im, dcValues-1)

    # remove value/2 from each side
    #   subtract by value/2, get the absolute value, find the argmin
    #   the value closest to 0 will be the best

    # get the lower end of the CDF
    lowerCDF = cdf - ((value/2)/100)
    absLowCDF = np.absolute(lowerCDF)
    minIndex = np.argmin(absLowCDF)

    # get the higher end of the CDF
    higherCDF = (-cdf+1) - ((value/2)/100)
    absHighCDF = np.absolute(higherCDF)
    maxIndex = np.argmin(absHighCDF)

    print("building slope")
    # the number of values we want to change, over how many values we'll cover
    # basically, rise over run
    slope = (dcValues-1) / (maxIndex - minIndex)
    intercept = slope*(minIndex)
    print('maxIndex ' + str(maxIndex))
    print('minIndex ' + str(minIndex))
    print('slope ' + str(slope))
    print('intercept ' + str(intercept))

    # create linear array
    linearArray = np.arange(dcValues)

    # build mask to clip lookup table
    # zero out elements under the minIndex and max elements over the maxIndex
    lowMask = linearArray < minIndex
    highMask = linearArray > maxIndex

    linearArray = (slope*linearArray) - intercept
    np.place(linearArray, lowMask, 0)
    np.place(linearArray, highMask, dcValues)

    return linearArray

def histogram_enhancement(im, etype='linear2', target=None, maxCount=255):
    """
    Function to run histogram enhancement and histogram matching for a given
    image. The function returns an image after the histogram matching or
    enhancement has been performed.

    Args:
        im (array): image to be modified based on the etype or target
        etype (optional[string]): type of enhancement to perform
            Can be 'linear2', 'linear3', etc..., to do a linear enhancement
            based on the CDF.
            Can be 'equalize' to do enhancement based on a flat histogram.
            Can be 'match', to do an image or histogram (PDF) based enhancement.
        target (optional[image, histogram]): image or histogram to match against
            Can be None, to do enhancement based on other etypes.
        maxCount (optional[int]): the maximum value for a digital count.
    Returns:
        the enhanced image
    Raises:
        TypeError: if image is not a numpy ndarray
        TypeError: if etype is not a string value (and no supplied target)

    """

    outputImage = np.zeros(np.shape(im))

    # get int type for the image (to return to that later)
    dtype = im.dtype

    # type checking, look at the above Raises section
    if (not isinstance(im, np.ndarray)):
        raise TypeError("image is not a numpy ndarray; use openCV's imread")

    # use the etype to determine type of transform
    if (not isinstance(etype, str)):
        raise TypeError("etype is not a string value");

    if (etype.find("linear") == 0):
        # linear was at index 0, so we're doing a linear transformation
        linearValue = etype.split("linear")[1]
        if(not linearValue.isdigit()):
            # check if "linear" was not followed by a digit
            raise ValueError("linear etype should contain a digit")

        # perform linear enhancement
        lut = build_linear_lookup_table(im, int(linearValue), maxCount + 1)
        print("lut for linear enhancement")
        plotHist(lut)
        outputImage = lut[im]

    elif (etype == "match"):
        if (not isinstance(im, np.ndarray)):
            raise TypeError("target is not a numpy ndarray")
        else:
            print("imgCDF")
            imgCDF = build_cdf(im, maxCount)
            plotHist(imgCDF)

            print("tarCDF")
            tarCDF = build_cdf(target, maxCount)
            plotHist(tarCDF)

            print("lut")
            # for every value, get the result from the imgCDF;
            # then, find the first index that exists for that value in
            # the tarCDF.
            lut = np.arange(maxCount)
            for i in range(maxCount):
                # sometimes our imgCDF is higher than the tarCDF ever is
                # lets put that at the maxCount
                if (imgCDF[i] > np.amax(tarCDF)):
                    lut[i] = maxCount
                else:
                    lut[i] = np.argmax(np.where( tarCDF >= imgCDF[i], 1, 0 ))

            print(lut)
            plotHist(lut)

            outputImage = lut[im]

    outputImage = np.array(outputImage, dtype)
    print("plot original image")
    plotImgHist(im)
    print("plot output image")
    plotImgHist(outputImage)
    return outputImage
"""
PYTHON TEST HARNESS
"""
if __name__ == '__main__':

    import cv2
    import ipcv
    import os.path
    import time
    # should include numpy import? ============================================#
    import numpy

    home = os.path.expanduser('~')
    filename = home + os.path.sep + 'src/python/examples/data/redhat.ppm'
    filename = home + os.path.sep + 'src/python/examples/data/crowd.jpg'
    filename = home + os.path.sep + 'src/python/examples/data/lenna.tif'
    filename = home + os.path.sep + 'src/python/examples/data/giza.jpg'
    # should be this? =========================================================#
    img = "giza.jpg"
    filename = os.path.join(home, 'src', 'python', 'examples', 'data', img)

    matchFilename = home + os.path.sep + 'src/python/examples/data/giza.jpg'
    matchFilename = home + os.path.sep + 'src/python/examples/data/lenna.tif'
    matchFilename = home + os.path.sep + 'src/python/examples/data/redhat.ppm'
    matchFilename = home + os.path.sep + 'src/python/examples/data/crowd.jpg'
    # should be this? =========================================================#
    img = "crowd.jpg"
    matchFilename = os.path.join(home, 'src', 'python', 'examples', 'data', img)

    im = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    print('Filename = {0}'.format(filename))
    print('Data type = {0}'.format(type(im)))
    print('Image shape = {0}'.format(im.shape))
    print('Image size = {0}'.format(im.size))

    cv2.namedWindow(filename, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(filename, im)

    print('Linear 2% ...')
    startTime = time.time()
    enhancedImage = ipcv.histogram_enhancement(im, etype='linear2')
    print('Elapsed time = {0} [s]'.format(time.time() - startTime))
    cv2.namedWindow(filename + ' (Linear 2%)', cv2.WINDOW_AUTOSIZE)
    cv2.imshow(filename + ' (Linear 2%)', enhancedImage)

    print('Linear 1% ...')
    startTime = time.time()
    enhancedImage = ipcv.histogram_enhancement(im, etype='linear1')
    print('Elapsed time = {0} [s]'.format(time.time() - startTime))
    cv2.namedWindow(filename + ' (Linear 1%)', cv2.WINDOW_AUTOSIZE)
    cv2.imshow(filename + ' (Linear 1%)', enhancedImage)

    print('Equalized ...')
    startTime = time.time()
    enhancedImage = ipcv.histogram_enhancement(im, etype='equalize')
    print('Elapsed time = {0} [s]'.format(time.time() - startTime))
    cv2.namedWindow(filename + ' (Equalized)', cv2.WINDOW_AUTOSIZE)
    cv2.imshow(filename + ' (Equalized)', enhancedImage)

    tgtIm = cv2.imread(matchFilename, cv2.IMREAD_UNCHANGED)
    print('Matched (Image) ...')
    startTime = time.time()
    enhancedImage = ipcv.histogram_enhancement(im, etype='match', target=tgtIm)
    print('Elapsed time = {0} [s]'.format(time.time() - startTime))
    cv2.namedWindow(filename + ' (Matched - Image)', cv2.WINDOW_AUTOSIZE)
    cv2.imshow(filename + ' (Matched - Image)', enhancedImage)

    tgtPDF = numpy.ones(256) / 256
    print('Matched (Distribution) ...')
    startTime = time.time()
    enhancedImage = ipcv.histogram_enhancement(im, etype='match', target=tgtPDF)
    print('Elapsed time = {0} [s]'.format(time.time() - startTime))
    cv2.namedWindow(filename + ' (Matched - Distribution)', cv2.WINDOW_AUTOSIZE)
    cv2.imshow(filename + ' (Matched - Distribution)', enhancedImage)

    action = ipcv.flush()
