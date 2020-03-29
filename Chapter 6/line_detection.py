import numpy as np
import random
from PIL import Image
import matplotlib.pyplot as plt
from PIL.ImageOps import equalize

from spatial_filter import correlation_2d


im = Image.open('wirebond.tif')
# im.show()
out = im.convert("L")
img = np.array(im)


filter = np.array([[1,1,1],[1,-8,1],[1,1,1]])

print(filter)

filtered_image_array = correlation_2d(img,filter)
filtered_image = Image.fromarray(filtered_image_array)
filtered_image.show()
      
# +45 degree filter
filter = np.array([[2,-1,-1],[-1,2,-1],[-1,-1,2]])

print(filter)

filtered_image_array = correlation_2d(img,filter)
filtered_image = Image.fromarray(filtered_image_array)
filtered_image.show()

