"""
PYTHON METHOD DEFINITION
Jesse Jurman (jrj2703)
"""

import numpy as np
import ipcv

def filter_bandpass(im, radialCenter, bandwidth, order=1, filterShape=ipcv.IPCV_IDEAL):
    """
    Function to generate a bandpass filter for the shape of a given image

    Args:
        im (array): source image, needed for the shape of the filter
        radialCenter (int): the frequency at which the filter stops
        brandwidth (int): the width of the band
        order (optional[int]): the order of the power for non-ideal filters
        filterShape (optional[int]): Frequency filter shapes,
            defined in constants.py, defaults to ideal filter

    Return:
        the filter for the image, as numpy.float64 data types
    """

    distanceArray = ipcv.filter_distance(im).astype(np.float64)
    filterFunction = {
        ipcv.IPCV_IDEAL : ipcv.ideal_bandreject,
        ipcv.IPCV_BUTTERWORTH : ipcv.butterworth_bandreject,
        ipcv.IPCV_GAUSSIAN : ipcv.gaussian_bandreject
    }

    return 1-filterFunction[filterShape](distanceArray, radialCenter, bandwidth, order)


"""
PYTHON TEST HARNESS
"""
if __name__ == '__main__':

    import cv2
    import ipcv
    import numpy
    import matplotlib.pyplot
    import matplotlib.cm
    import mpl_toolkits.mplot3d
    import os.path

    home = os.path.expanduser('~')
    filename = home + os.path.sep + 'src/python/examples/data/lenna.tif'
    im = cv2.imread(filename)

    frequencyFilter = ipcv.filter_bandpass(im,
                                       32,
                                       15,
                                       order=2,
                                       filterShape=ipcv.IPCV_BUTTERWORTH)
    frequencyFilter = ipcv.filter_bandpass(im,
                                       32,
                                       15,
                                       filterShape=ipcv.IPCV_GAUSSIAN)
    frequencyFilter = ipcv.filter_bandpass(im,
                                       32,
                                       15,
                                       filterShape=ipcv.IPCV_IDEAL)

    # Create a 3D plot and image visualization of the frequency domain filter
    rows = im.shape[0]
    columns = im.shape[1]
    u = numpy.arange(-columns/2, columns/2, 1)
    v = numpy.arange(-rows/2, rows/2, 1)
    u, v = numpy.meshgrid(u, v)

    figure = matplotlib.pyplot.figure('Frequency Domain Filter', (14, 6))
    p = figure.add_subplot(1, 2, 1, projection='3d')
    p.set_xlabel('u')
    p.set_xlim3d(-columns/2, columns/2)
    p.set_ylabel('v')
    p.set_ylim3d(-rows/2, rows/2)
    p.set_zlabel('Weight')
    p.set_zlim3d(0, 1)
    p.plot_surface(u, v, frequencyFilter)
    i = figure.add_subplot(1, 2, 2)
    i.imshow(frequencyFilter, cmap=matplotlib.cm.Greys_r)
    matplotlib.pyplot.show()
