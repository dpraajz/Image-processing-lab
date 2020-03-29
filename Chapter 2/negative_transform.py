import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

im = Image.open('breast_digital_Xray.tif')
out = im.convert("L")
histogram_original = out.histogram()
pic = np.array(out)
print(pic.shape)
# print(pic)

def inverse(i):
    return (255-i)


new_pic = np.zeros((pic.shape))

for count,row in enumerate(pic):
    for num,each in enumerate(row):
        new_value = inverse(each)
        new_pic[count,num] = new_value

final_image = Image.fromarray(new_pic)
histogram_inversed = final_image.histogram()

out.show()
final_image.show()

plt.figure(0)

# Original histogram

for i in range(0, 256):

    plt.bar(i, histogram_original[i], alpha=0.3)

 

# Inversed histogram

plt.figure(1)

for i in range(0, 256):

    plt.bar(i, histogram_inversed[i], alpha=0.3)

plt.show()