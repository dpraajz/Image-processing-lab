import numpy as np
import random
from PIL import Image
import matplotlib.pyplot as plt
from PIL.ImageOps import equalize

from spatial_filter import correlation_2d


im = Image.open('turbine_blade_black_dot.tif')
# im.show()
out = im.convert("L")
img = np.array(im)


filter = np.array([[1,1,1],[1,-8,1],[1,1,1]])

print(filter)

filtered_image_array = correlation_2d(img,filter)
filtered_image = Image.fromarray(filtered_image_array)
filtered_image.show()
max_pixel = np.amax(filtered_image_array)
final_image_array = np.zeros((img.shape))

for count,row in enumerate(filtered_image_array):
    for num,each in enumerate(row):
        # print(each)
        if abs(each) >= (max_pixel*0.9):
            print(each)
            final_image_array[count,num] = 255

final_image = Image.fromarray(final_image_array)
final_image.show()        
