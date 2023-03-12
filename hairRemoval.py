import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


path = "/Users/lenaschill/Desktop/Skin_Dataset/training/Cancer/77.jpg" #hairy image
image = cv2.imread(path)

image_resize = cv2.resize(image,(224,224))
# Convert the original image to grayscale
plt.subplot(1, 5, 1)
plt.imshow(cv2.cvtColor(image_resize, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('Original')

grayScale = cv2.cvtColor(image_resize, cv2.COLOR_RGB2GRAY)
plt.subplot(1, 5, 2)
plt.imshow(grayScale, cmap = plt.cm.gray)
plt.axis('off')
plt.title('GrayScale')

# Kernel for the morphological filtering
kernel = cv2.getStructuringElement(1,(17,17))

# Perform the blackHat filtering on the grayscale image to find the hair countours
blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
plt.subplot(1, 5, 3)
plt.imshow(blackhat)
plt.axis('off')
plt.title('Blackhat')


# intensify the hair countours in preparation for the inpainting 
ret,threshold = cv2.threshold(blackhat,10,255,cv2.THRESH_BINARY)
plt.subplot(1, 5, 4)
plt.imshow(threshold)
plt.axis('off')
plt.title('Threshold')

# inpaint the original image depending on the mask
final_image = cv2.inpaint(image_resize,threshold,1,cv2.INPAINT_TELEA)
plt.subplot(1, 5, 5)
plt.imshow(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('Final image')

