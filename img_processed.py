import cv2
from matplotlib import pyplot as plt
img_path='IMG_4727.jpg'

img =cv2.imread(img_path)

#inverted the color of the image
inverted_image=cv2.bitwise_not(img)
#write the inverted image to a file
# cv2.imwrite("temp/inverted.jpg",inverted_image)

def greyscale_convert(image):
    return cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

grey_img=greyscale_convert(img)

thresh,blackwhite_img = cv2.threshold(grey_img, 150, 230, cv2.THRESH_BINARY)
cv2.imwrite("temp/blackwhiteimg.jpg",blackwhite_img)

def noise_removal(image):
    import numpy as np 
    
def display(im_path):
    dpi=80
    im_data=plt.imread(im_path)
    height,width,depth=im_data.shape
    #find out which size the figure need to be in inches
    fig_size=width / float(dpi),height / float(dpi)

    #create figure
    fig=plt.figure(figsize=fig_size)
    ax = fig.add_axes([0,0,1,1])

    #hide spines, ticks,etc.
    ax.axis('off')

    ax.imshow(im_data,cmap='gray')
    plt.show()
# display('IMG_20231219_183637.jpg') 