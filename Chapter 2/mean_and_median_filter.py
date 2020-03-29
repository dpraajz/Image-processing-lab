import numpy as np
from PIL import Image
from PIL.ImageFilter import BLUR, MedianFilter

im = Image.open('ckt_board_saltpep.tif')
out = im.convert("L")
out.show()

mean_filtered_image = out.filter(BLUR)
mean_filtered_image.show()

median_filtered_image = out.filter(MedianFilter(3))
median_filtered_image.show()