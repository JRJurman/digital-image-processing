"""
PYTHON METHOD DEFINITION
"""
import numpy as np

def p(im, i):
    """
    Function to get the probability of pixels at level i, for the entire image

    Args:
        im (array): image or histogram
        i (array or int): pixel level

    Returns:
        a probability of a pixel at level i
    """

    if (len(im.shape()) == 2):
        # an image, we need to get a histogram
        hist = cv2.calcHist([im],[0],None,[255],[0,255])
        size = np.cumprod(im.shape())
    elif (len(im.shape()) == 1):
        hist = im;
        size = np.cumsum(hist)
    else:
        raise ValueError("Shape of image is not a grayscale image or histogram")

    return (hist[i] / size)

def histSliceProbabilty(im, start, end):
    """
    Function to return a slice of a histogram of probabilities for each level

    Args:
        im (array): image to convert to histogram and evaluate
        start (int): first pixel level to start evaluating class occurance
            value is inclusive
        end (int): pixel level to stop evaluating class occurance probability
            value is exclusive

    Returns:
        probability of each level in the histogram
    """

    # build the histogram and slice it to our start and end range
    hist = cv2.calcHist([im],[0],None,[255],[0,255])
    histSlice = hist[start:end]

    # get the entire size of the image, and build the probability for ths hist
    size = np.cumprod(im.shape())
    result = histSlice / size

    return result

def w(im, start, end, maxCount=255):
    """
    Function to evaluate probability of class occurance

    Args:
        im (array): image to convert to histogram and evaluate
        start (int): first pixel level to start evaluating class occurance
            value is inclusive
        end (int): pixel level to stop evaluating class occurance probability
            value is exclusive
        maxCount (option[int]): total number of pixel levels possible
            shortcut for total class variance squared

    Returns:
        probability of class occurance in the range specified
    """

    if ((start == 0) and (end == maxCount)):
        return 1

    # get histogram of probabilities
    histSliceProbs = histSliceProbabilty(im, start, end)

    # get the cummulative sum of all elements in the slice
    return np.cumsum(histSliceProbs)

def u(im, start, end):
    """
    Function to evaulate the class mean levels in the range specified

    Args:
        im (array): image to convert to histogram and evaluate
        start (int): first pixel level to start evaluating mean levels
            value is inclusive
        end (int): pixel level to stop evaluating class mean levels
            value is exclusive

    Returns:
        class mean level for the range specified
    """

    # build the histogram and slice it to our start and end range
    iHist = np.arange(end)
    histSlice = iHist[start:end]

    # get histogram of probabilities
    histSliceProbs = histSliceProbabilty(im, start, end)

    histSliceMean = histSlice * histSliceProbs

    return np.cumsum(histSliceMean)

def class_variance_squared(im, start, end):
    """
    Function to evaulate the class variance (squared) in the range specified

    Args:
        im (array): image to convert to histogram and evaluate
        start (int): first pixel level to start evaluating class variance
            value is inclusive
        end (int): pixel level to stop evaluating class variance
            value is exclusive

    Returns:
        class variance (squared) for the range specified
    """

    # get mean level for range
    meanLevel = u(im, start, end)

    # build the histogram and slice it to our start and end range
    iHist = np.arange(end)
    histSlice = iHist[start:end]

    # build array of histogram probabilities
    histProbs = histSliceProbabilty(im, start, end)

    # difference of mean and the histLevel
    meanDiff = (histSlice - meanLevel)

    resultHist = (meanDiff*meanDiff) * (histProbs/w(im, start, end))

    return np.cumsum(resultHist)

def class_variance_b_squared(im, start, end, maxCount=255):
    """
    Function to get the class variance b squared, as described in otsu's paper

    Args:
        im (array): image to convert to histogram and evaluate
        start (int): first pixel level to start evaluating class variance
            value is inclusive
        end (int): pixel level to stop evaluating class variance
            value is exclusive
        maxCount (option[int]): total number of pixel levels possible
            shortcut for total class variance squared

    Returns:
        class variance b (squared) for the range specified
    """

    # evaluation of k for start to end
    wk = w(im, start, end)

    numerator = (u(im, 0, maxCount) * wk - u(im, start, end))**2
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
    if (len(im.shape()) != 2):
        raise ValueError("Shape of image is not a grayscale image or histogram")

    # evaluate sigma of b squared for all values from 0 to maxCount
    for i in range(maxCount):
        sigma = class_variance_b_squared(im, 0, i, 255)
        print("i={}: sigma={}".format(i, sigma))


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
