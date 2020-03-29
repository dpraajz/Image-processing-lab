import numpy as np
from PIL import Image
from PIL.ImageFilter import GaussianBlur, MedianFilter, UnsharpMask

im = Image.open('dipxe_text.tif')
out = im.convert("L")
out.show()

out_array = np.array(out)

blured_image = out.filter(GaussianBlur(0.5))
# blured_image.show()
blured_image_array = np.array(blured_image)
unsharp_mask = out_array - blured_image_array

unsharp_mask_show = Image.fromarray(unsharp_mask)
# unsharp_mask_show.show()

unsharp_mask_image = out_array + unsharp_mask
unsharp_masked_image = Image.fromarray(unsharp_mask_image)
unsharp_masked_image.show()

direct_unsharp_mask = out.filter(UnsharpMask(2,150,3))
direct_unsharp_mask.show()

# high_boost_filter_image = out_array + 10*unsharp_mask
# high_boost_filtered_image = Image.fromarray(high_boost_filter_image)
# high_boost_filtered_image.show()