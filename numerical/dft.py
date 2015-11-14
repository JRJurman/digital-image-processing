"""
PYTHON METHOD DEFINITION
by Jesse Jurman (jrj2703)
"""

import numpy as np

def dft(f, scale=True):
    """
    Function to evaluate the dft of an array.

    Args:
        f (1d-array): array to calculate the dft of
        scale (optional[boolean]): scales the result by dividing it by the
                                    number of array elements

    Returns:
        Array of fourier transformation
    """

    M = f.shape[0]
    u = np.arange(M)
    rotU = u.reshape(u.shape[0], 1)

    theta = (2 * np.pi * u * rotU) / M
    phase = np.cos(theta) - (np.sin(theta)*(0+1j))

    scaleFactor = 1
    if (scale):
        scaleFactor = (1/M)

    transform = scaleFactor * np.sum((f*phase), axis=1)
    return transform

"""
PYTHON TEST HARNESS
"""
if __name__ == '__main__':
    import numerical
    import numpy
    import time

    N = 2**12
    f = numpy.ones(N, dtype=numpy.complex128)

    repeats = 10
    print('Repetitions = {0}'.format(repeats))

    startTime = time.clock()
    for repeat in range(repeats):
        F = numerical.dft(f)
    string = 'Average time per transform = {0:.8f} [s] ({1}-point DFT)'
    print(string.format((time.clock() - startTime)/repeats, len(f)))

    startTime = time.clock()
    for repeat in range(repeats):
        F = numpy.fft.fft(f)
    string = 'Average time per transform = {0:.8f} [s] ({1}-point FFT)'
    print(string.format((time.clock() - startTime)/repeats, len(f)))
