import numpy as np

def filter_distance(im):
    """
    Function to calculate distance from the center
    Used for filter functions

    returns an array of distances from the center
    """

    M = im.shape[0]
    N = im.shape[1]

    # we want to create two tables of just the x position, and just the y
    # position these are so we can calculate the values based
    # on u and v, and then square root the sum
    template = np.arange(M*N).reshape(M, N)
    templateX = template % N
    templateY = np.floor( template / N )

    # can't do exactly center, because when we do this math, we get two rows
    # that have the same distance, just different signs (depending if it is odd
    # or even). This creates two rows that have the same values with odd valued
    # heights, and what seems like a one-off-shift with even values
    resultX = (templateX - (N / 2) )**2
    resultY = (templateY - (M / 2) )**2

    return (resultX + resultY)**(1/2)
