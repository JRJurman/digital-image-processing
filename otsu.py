"""
PYTHON METHOD DEFINITION
"""
import numpy as np
import cv2
import ipcv

def w(im, k, maxCount=255):
    """
    Function to evaluate probability of class occurance

    Args:
        im (array): image to convert to histogram and evaluate
        k (int): pixel level evaluating class occurance probability
        maxCount (option[int]): total number of pixel levels possible
            shortcut for total class variance squared

    Returns:
        probability of class occurance at k
    """

    # get histogram of probabilities
    hist = cv2.calcHist([im],[0],None,[255+1],[0,255+1])
    histProbs = hist / np.cumprod(np.shape(im))[-1]

    # get the cummulative sum of all elements in the slice
    return np.cumsum(histProbs)[k]

def w0(im, k):
    return w(im, k)

def w1(im, k):
    return 1 - w(im, k)

def u(im, k):
    """
    Function to evaulate the class mean levels

    Args:
        im (array): image to convert to histogram and evaluate
        k (int): pixel level to evaluating mean levels

    Returns:
        class mean level for k
    """

    # get histogram of probabilities
    hist = cv2.calcHist([im],[0],None,[255+1],[0,255+1])
    histProbs = hist / np.cumprod(np.shape(im))[-1]

    # build array of indicies
    indHist = np.arange(256)

    histMean = indHist * histProbs

    ipcv.plotHist(histProbs)
    ipcv.plotHist(indHist)
    ipcv.plotHist(histMean)

    result = np.cumsum(histMean)[k]
    return result

def u0(im, k):
    return u(im, k)/w(im, k)

def u1(im, k):
    return ( u(im, 255) - u(im, k) )/( 1 - w(im, k) )

def class_variance_b_squared(im, k, maxCount=255):
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

    print("======= {} ======".format(k))
    print("w0", w0(im,k))
    print("w1", w1(im,k))
    print("u", u(im, k))
    print("u0", u0(im,k))
    print("u1", u1(im,k))
    return w0(im, k) * w1(im, k) * ( (u1(im, k) - u0(im, k))**2 )

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

    ipcv.plotHist(result)
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
