"""
PYTHON METHOD DEFINITION
by Jesse Jurman (jrj2703)
"""
from operator import itemgetter
import numpy as np
import cv2
import ipcv

def invertImage(src):
    """
    Function to return inverted image
     This is a separate method to handle all the conversions required

    Args:
        src(array of ints): image to convert

    Returns:
        inverted (black -> white, white -> black) image
    """
    # convert src to negatable-source, something that can have negative values
    nsrc = np.array(src, np.int32)

    # subtract the max value, and get absolute
    psrc = np.abs(nsrc - nsrc.max())

    # return the image in the original type
    return np.array(psrc, src.dtype)



def character_recognition(src, templates, threshold, filterType='spatial', verbose=False):
    """
    Function to determine the number of characters (for each character) that
    exist in an image.

    Author:
        Jesse Jurman (jrj2703)

    Args:
        src (array of ints): image to read charcters from
        templates (array of arrays): images for each character to be read in
        threshold (float): how strong the response needs to be before it is
            considered a match
        filterType (string): type of filter to run with images.
            defaults to 'spatial', can be 'match'
        verbose (bool): plot character maps

    Returns:
        a list of the indicies for the templates, in the order that they appear
            on the source image
        a histogram (size of the array templates)

    Raises:
        ValueError: filterType of not spatial or matched
    """

    isrc = invertImage(src)
    results = []
    text = []

    # text map is a mapping of x,y coords to letters
    # this will be sorted to retrieve the final text
    textMap = {}


    if (filterType == 'spatial'):

        # for every template, build a 2D map of where the template matches
        for character in templates:

            if (verbose):
                cv2.namedWindow('isrc', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('isrc', isrc)

            if (verbose):
                cv2.namedWindow('character', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('character', character)

            # invert the character
            icharacter = invertImage(character)

            # normalize template
            icharacter = icharacter.astype(np.float64)
            if (icharacter.sum() != 0):
                icharacter /= icharacter.sum()

            if (verbose):
                cv2.namedWindow('icharacter', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('icharacter', icharacter)

            # find points using filter2D
            cloudMatch = cv2.filter2D(src, -1, icharacter)

            if (verbose):
                cv2.namedWindow('CloudMatch', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('CloudMatch', cloudMatch)

            # make single points using np.where
            pointMatch = np.where(cloudMatch >= threshold, 0, 255)
            pointMatch = np.array(pointMatch, cloudMatch.dtype)

            if (verbose):
                cv2.namedWindow('PointMatch', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('PointMatch', pointMatch)

            if (verbose):
                ipcv.flush()

            results.append((pointMatch/255).sum())

            coords = np.where(pointMatch >= threshold)
            for coordInd in range(len(coords[0])):
                coordTuple = (coords[0][coordInd], coords[1][coordInd])

                # we don't know what actual letter we're dealing with,
                # but we kinda know the index using the size of the results list

                # this will totally overwrite letters that sit in the same coord
                textMap[coordTuple] = len(results)-1


    elif (filterType == 'matched'):

        # you can set the loading bar width here
        # make sure it's smaller than your window, otherwise it'll
        # print a LOT of newlines
        loadingBarWidth = 50
        loading = ipcv.Bar(loadingBarWidth)
        load = 0
        # iterate through our templates
        for character in templates:

            # update loading bar
            load += 1
            loading.showBar(load/len(templates))

            # image of matches
            resImage = np.zeros(src.shape)

            # size of 2d slices
            s = templates[0].shape
            arrayWidth = s[0]*s[1]

            # invert character
            icharacter = invertImage(character)

            # flatten template
            rcharacter = icharacter.flatten()

            # invert source
            isrc = invertImage(src)

            row = 0
            for row in range(src.shape[0]-s[0]):
                for col in range(src.shape[1]-s[1]):

                    # cut out a portion of the image, based on the row and column
                    cut = isrc[row:row+s[0], col:col+s[1]]
                    wideCut = cut.flatten()

                    # manipulate the character
                    numerator = np.vdot(rcharacter, wideCut)
                    denominator = np.vdot(np.linalg.norm(rcharacter),
                                            np.linalg.norm(wideCut))
                    if (denominator != 0):
                        resImage[row][col] = numerator/denominator

                loading.showBar(load/len(templates))


            if (verbose):
                cv2.namedWindow('resImage', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('resImage', resImage)

            finImage = np.where(resImage >= threshold, 255, 0)
            finImage = np.array(finImage, src.dtype)

            if (verbose):
                cv2.namedWindow('finImage', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('finImage', finImage)

            if (verbose):
                ipcv.flush()

            results.append((finImage/255).sum())

            coords = np.where(finImage >= threshold)
            for coordInd in range(len(coords[0])):
                coordTuple = (coords[0][coordInd], coords[1][coordInd])

                # we don't know what actual letter we're dealing with,
                # but we kinda know the index using the size of the results list

                # this will totally overwrite letters that sit in the same coord
                textMap[coordTuple] = len(results)-1

    else:
        raise ValueError("only filter types supported are spatial and matched")

    # sort the text by it's x and y value
    # print("TEXTMAP.KEYS()", textMap.keys())
    sortedMapKeys = sorted(textMap.keys(), key=itemgetter(0,1))
    # print("SORTEDMAPKEYS",sortedMapKeys)
    for smk in sortedMapKeys:
        text.append(textMap[smk])

    return text, results
"""
PYTHON TEST HARNESS
"""
if __name__ == '__main__':

    import cv2
    import fnmatch
    import numpy
    import os
    import os.path
    import ipcv

    home = os.path.expanduser('~')
    baseDirectory = home + os.path.sep + 'src/python/examples/data'
    baseDirectory += os.path.sep + 'character_recognition'

    documentFilename = baseDirectory + '/notAntiAliased/alphabet.tif'
    documentFilename = baseDirectory + '/notAntiAliased/text.tif'
    charactersDirectory = baseDirectory + '/notAntiAliased/characters'

    document = cv2.imread(documentFilename, cv2.IMREAD_UNCHANGED)

    characterNames = []
    characterImages = []
    for root, dirnames, filenames in os.walk(charactersDirectory):
        for filename in sorted(filenames):
            characterNames.append(chr(int(filename.split('.')[0])))
            currentCharacter = cv2.imread(root + '/' + filename,
                                                        cv2.IMREAD_UNCHANGED)
            characterImages.append(currentCharacter)

    """
    # append the space character
    spaceCharacter = characterImages[0].copy()
    spaceCharacter.fill(0)
    characterNames.append(" ")
    characterImages.append(spaceCharacter)
    """

    characterImages = numpy.asarray(characterImages)

    # Define the filter threshold
    threshold = 1.0

    # run the spatial filter
    text, histogram = character_recognition(document, characterImages, threshold, filterType='spatial', verbose=False)

    # display the histogram in the console
    for n in range(len(characterNames)):
        print(str(characterNames[n]) +": "+ str(histogram[n]))

    # plot the histogram
    ipcv.plotLetters(histogram, numpy.array(characterNames))

    # build and show the final string
    realText = []
    for index in text:
        realText.append(characterNames[index])

    realTextString = "".join(realText)
    print("TEXT:\n", realTextString)

    # Define the filter threshold
    threshold = 0.97

    # run the matched filter
    text, histogram = character_recognition(document, characterImages, threshold, filterType='matched', verbose=False)

    # display the histogram in the console
    for n in range(len(characterNames)):
        print(str(characterNames[n]) +": "+ str(histogram[n]))

    # plot the histogram
    ipcv.plotLetters(histogram, numpy.array(characterNames))

    # build and show the final string
    realText = []
    for index in text:
        realText.append(characterNames[index])

    realTextString = "".join(realText)
    print("TEXT:\n", realTextString)
