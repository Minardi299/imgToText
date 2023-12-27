import pytesseract
import sys
from matplotlib import pyplot as plt
import cv2

import numpy as np



pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\Tesseract-OCR\\tesseract.exe'



pytesseract.image_to_pdf_or_hocr

#take in black and white image
def noise_removal(image: np.ndarray):
    kernel = np.ones((1,1),np.uint8)
    image=cv2.dilate(image,kernel,iterations=1)
    kernel=np.ones((1,1),np.uint8)
    image=cv2.erode(image,kernel,iterations=1)
    image=cv2.morphologyEx(image,cv2.MORPH_CLOSE,kernel)
    image=cv2.medianBlur(image,3)
    return (image)

def display(img_path):
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

def thining_font(img: np.ndarray)-> np.ndarray:
    img=cv2.bitwise_not(img)
    kernel=np.ones((2,2),np.uint8)
    img=cv2.erode(img,kernel,iterations=1)
    img=cv2.bitwise_not(img)
    return img



def thickening_font(img: np.ndarray) -> np.ndarray:
    img=cv2.bitwise_not(img)
    kernel=np.ones((2,2),np.uint8)
    img=cv2.dilate(img,kernel,iterations=1)
    img=cv2.bitwise_not(img)
    return img

# Calculate skew angle of an image
def getSkewAngle(cvImage) -> float:
    # Prep image, copy, convert to gray scale, blur, and threshold
    newImage = cvImage.copy()
    gray = cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Apply dilate to merge text into meaningful lines/paragraphs.
    # Use larger kernel on X axis to merge characters into single line, cancelling out any spaces.
    # But use smaller kernel on Y axis to separate between different blocks of text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilate = cv2.dilate(thresh, kernel, iterations=5)

    # Find all contours
    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)

    # Find largest contour and surround in min area box
    largestContour = contours[0]
    minAreaRect = cv2.minAreaRect(largestContour)

    # Determine the angle. Convert it to the value that was originally used to obtain skewed image
    angle = minAreaRect[-1]
    if angle < -45:
        angle = 90 + angle
    return -1.0 * angle

# Rotate the image around its center
def rotateImage(cvImage, angle: float):
    newImage = cvImage.copy()
    (h, w) = newImage.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    newImage = cv2.warpAffine(newImage, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return newImage

# Deskew image
def deskew(cvImage):
    angle = getSkewAngle(cvImage)
    return rotateImage(cvImage, -1.0 * angle)

def remove_borders(img):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    contours, heiarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cntsSorted = sorted(contours, key=lambda x:cv2.contourArea(x))
    cnt = cntsSorted[-1]
    x, y, w, h = cv2.boundingRect(cnt)
    crop = img[y:y+h, x:x+w]
    return (crop)

def process_img(img: np.ndarray) -> np.ndarray:
    inverted_image=cv2.bitwise_not(img)
    grey_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #tweak the 150 and 230 value for better result
    thresh,blackwhite_img = cv2.threshold(grey_img, 130, 230, cv2.THRESH_BINARY)
    no_noise=noise_removal(blackwhite_img)
    final= thickening_font(no_noise)

    return final
    
def img_to_string(img: np.ndarray) -> str:
    processed_img = process_img(img)
    text =pytesseract.image_to_string(processed_img)  
    return text

def img_to_stringpdf(img: np.ndarray) -> str:
    inverted_image=cv2.bitwise_not(img)
    grey_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #tweak the 150 and 230 value for better result
    thresh,blackwhite_img = cv2.threshold(grey_img, 130, 230, cv2.THRESH_BINARY)
    # final=thickening_font(blackwhite_img)
    text =pytesseract.image_to_string(blackwhite_img)  
    return text


# img=cv2.imread('IMG_4727.jpg')

# fixed = process_img(img)

# cv2.imwrite("temp/processed.jpg", fixed)
# print(pytesseract.image_to_string(fixed))