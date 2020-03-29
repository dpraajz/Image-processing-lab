import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from PIL.ImageOps import equalize

im = Image.open('beans.tif')
out = im.convert("L")
histogram_original = out.histogram()
final = equalize(out)
histogram_equalized = final.histogram()
out.show()
final.show()
# Original histogram

plt.figure(0)
for i in range(0, 256):

    plt.bar(i, histogram_original[i], alpha=0.3)

 

# Inversed histogram

plt.figure(1)

for i in range(0, 256):

    plt.bar(i, histogram_equalized[i], alpha=0.3)

plt.show()