# using numpy's floor, because we already are using numpy
from numpy import floor

# required only for showing a loading bar
class Bar:
    def __init__(self, width):
        self.symCount = 0
        self.width = width

    def showBar(self, progress):
        symbols = "|/-\\"
        fillCount = floor(self.width*progress)
        self.symCount += 1
        sym = symbols[self.symCount%len(symbols)]
        print("[" + ("="*fillCount) + sym + (" "*((self.width-1)-fillCount)) + "]\r", end="")
