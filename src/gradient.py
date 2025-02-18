

from numba.typed import List as Lst
from numpy import array
from colour import Color

"""
darkBlue = Color('#120b96')
blue = Color('#1542a3')
lightBlue = Color('#2aa3db')
cyan = Color('#2adbc4')
lightGreen = Color('#2adb6e')
"""
blue = Color('#2a2ddb')
red = Color('#ff0000')

gradientLen = 15
gradientColors = [(int(c.red * 255), int(c.green * 255), int(c.blue * 255)) for c in blue.range_to(red, gradientLen)]

vMax = 300
scaleFactor = (gradientLen) / vMax
