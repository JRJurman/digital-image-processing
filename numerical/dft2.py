"""
PYTHON METHOD DEFINITION
by Jesse Jurman (jrj2703)
"""

import numpy as np
import numerical

def dft2(f, scale=True):
    """
    Function to evaluate the dft for a 2D array.

    Args:
        f (2d-array): array to calculate the 2 dimensional dft of
        scale (optional[boolean]): scales the result by dividing it by the
                                    number of array elements

    Returns:
        2 dimensional array of fourier transformation
    """

    # for every column, evaluate the dft
    columns = f.swapaxes(0,1)
    Columns = []
    for col in range(columns.shape[0]):
        Columns.append(numerical.dft(columns[col], False))

    newSpace = np.array(Columns)

    # for every row in the new space, evaluate the dft
    # swap the axes again to set it back to normal
    rows = newSpace.swapaxes(0,1)
    Rows = []
    for row in range(rows.shape[0]):
        Rows.append(numerical.dft(rows[row], False))

    finalSpace = np.array(Rows)

    M = f.shape[0]*f.shape[1]
    scaleFactor = 1
    if (scale):
        scaleFactor = (1/M)

    return finalSpace*scaleFactor


"""
PYTHON TEST HARNESS
"""
if __name__ == '__main__':
    import numerical
    import numpy
    import time

    M = 2**5
    N = 2**5
    f = numpy.ones((M,N), dtype=numpy.complex128)

    repeats = 10
    print('Repetitions = {0}'.format(repeats))

    startTime = time.clock()
    for repeat in range(repeats):
        F = numerical.dft2(f)
    string = 'Average time per transform = {0:.8f} [s] '
    string += '({1}x{2}-point DFT2)'
    print(string.format((time.clock() - startTime)/repeats, M, N))

    startTime = time.clock()
    for repeat in range(repeats):
        F = numpy.fft.fft2(f)
    string = 'Average time per transform = {0:.8f} [s] '
    string += '({1}x{2}-point FFT2)'
    print(string.format((time.clock() - startTime)/repeats, M, N))
