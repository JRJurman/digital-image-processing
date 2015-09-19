"""
PYTHON METHOD DEFINITION
"""
import numpy as np
import cv2
import ipcv

def w(im, i, maxCount=255):
    """
    Function to evaluate probability of class occurance

    Args:
        im (array): image to convert to histogram and evaluate
        i (int): pixel level evaluating class occurance probability
        maxCount (option[int]): total number of pixel levels possible
            shortcut for total class variance squared

    Returns:
        probability of class occurance at i
    """

    # get histogram of probabilities
    hist = cv2.calcHist([im],[0],None,[255],[0,255])
    histProbs = hist / np.cumprod(np.shape(im))[-1]

    # get the cummulative sum of all elements in the slice
    return np.cumsum(histProbs)[i]

def u(im, i):
    """
    Function to evaulate the class mean levels

    Args:
        im (array): image to convert to histogram and evaluate
        i (int): pixel level to evaluating mean levels

    Returns:
        class mean level for i
    """

    # get histogram of probabilities
    hist = cv2.calcHist([im],[0],None,[255],[0,255])
    histProbs = hist / np.cumprod(np.shape(im))[-1]

    # build array of indicies
    indHist = np.array(hist.max())

    histMean = indHist * histProbs

    result = np.cumsum(histMean)[i] * w(im, i)
    return result

def class_variance_b_squared(im, i, maxCount=255):
    """
    Function to get the class variance b squared, as described in otsu's paper

    Args:
        im (array): image to convert to histogram and evaluate
        i (int): pixel level to evaluating class variance
        maxCount (option[int]): total number of pixel levels possible
            shortcut for total class variance squared

    Returns:
        class variance b (squared) for the pixel level
    """

    wk = w(im, i, maxCount)

    numerator = ((u(im, maxCount-1) * wk) - u(im, i))**2
    denominator = wk * (1 - wk)

    return numerator/denominator

def otsu_threshold(im, maxCount=255, verbose=False):
    """
    Function to turn an image into a binary split based on otus's method

    Args:
        im (array): image to threshold
        maxCount (int): maximum value of any element in the array
            For images this will be 255
        verbose (boolean): flag to visually show plot of histogram
            with threshold

    Returns:
        a tuple of a binary image (array), and the threshold (int)

    Raises:
        TypeError: if image is not a numpy ndarray
        ValueError: if image is not in the shape of a grayscale image
    """

    # type checking, look at the above Raises section
    if (not isinstance(im, np.ndarray)):
        raise TypeError("image is not a numpy ndarray; use openCV's imread")
    if (len(np.shape(im)) != 2):
        raise ValueError("Shape of image is not a grayscale image or histogram")

    # get last non-zero pixel level
    hist = cv2.calcHist([im],[0],None,[255],[0,255])
    # get the element pair, which contains an index and value (which is 0)
    firstPixelIndex = np.transpose(np.nonzero(hist))[0][0]
    lastPixelIndex = np.transpose(np.nonzero(hist))[-1][0]

    # evaluate sigma of b squared for all values from the first non zero pixel
    # level to the last non zero pixel level
    result = np.zeros(maxCount)
    for i in range(firstPixelIndex+1, lastPixelIndex):
        sigma = class_variance_b_squared(im, i, maxCount)
        result[i] = sigma

    thresh = np.argmax(result)
    if (verbose):
        ipcv.plotHist(hist, [np.argmax(result)])

    # apply threshold to image
    binaryIm = np.ones(np.shape(im))
    binaryIm = binaryIm*im
    np.place(binaryIm, im<thresh, 0)
    np.place(binaryIm, im>=thresh, 1)

    return (binaryIm, thresh)


"""
PYTHON TEST HARNESS
"""
if __name__ == '__main__':

    import cv2
    import ipcv
    import os.path
    import time

    home = os.path.expanduser('~')
    filename = home + os.path.sep + 'src/python/examples/data/giza.jpg'

    im = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    print('Filename = {0}'.format(filename))
    print('Data type = {0}'.format(type(im)))
    print('Image shape = {0}'.format(im.shape))
    print('Image size = {0}'.format(im.size))

    startTime = time.time()
    thresholdedImage, threshold = ipcv.otsu_threshold(im, verbose=True)
    print('Elapsed time = {0} [s]'.format(time.time() - startTime))

    print('Threshold = {0}'.format(threshold))

    cv2.namedWindow(filename, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(filename, im)
    cv2.namedWindow(filename + ' (Thresholded)', cv2.WINDOW_AUTOSIZE)
    cv2.imshow(filename + ' (Thresholded)', thresholdedImage * 255)

    action = ipcv.flush()
