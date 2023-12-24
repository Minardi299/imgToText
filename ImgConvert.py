import pytesseract
import sys
from matplotlib import pyplot as plt
import cv2
from pdf_processing import *
import numpy as np



pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\Tesseract-OCR\\tesseract.exe'


def img_to_string(img: np.ndarray) -> str:
    text =pytesseract.image_to_string(img)  
    return text

def process_img(img: np.ndarray) -> np.ndarray:
    inverted_image=cv2.bitwise_not(img)
    grey_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #tweak the 150 and 230 value for better result
    thresh,blackwhite_img = cv2.threshold(grey_img, 150, 230, cv2.THRESH_BINARY)
    cv2.imwrite("temp/blackwhiteimg.jpg",blackwhite_img)

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

def get_skew_angle(img: np.ndarray) -> float:
    #prep image,copy,convert to gray scale, blur and theshold
    new_img=img.copy()
    gray=cv2.cvtColor(new_img,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(9,9),0)
    thresh=cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    #apply dilate to merge text into meaningful lines/paragraphs.
    #use larger kernel on X axis to merge characters into single line, cancelling out any spaces.
    #but use smaller kernel on Y axis to seperate between different blocks of text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(30,5))
    dilate = cv2.dilate(thresh,kernel,iterations=2)

    #find all contours
    contours, hierarchy = cv2.findContours(dilate,cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours= sorted(contours,key = cv2.contourArea, reverse = True)
    for c in contours:
        rect = cv2.boundingRect(c)
        x,y,w,h = rect
        cv2.rectangle(new_img,(x,y),(x+w,y+h),(9,255,0),2)

    #find largest contour and surround in min area box
    largestContur = contours[0]
    print(len(contours))
    minAreaRect= cv2.minAreaRect(largestContur)
    cv2.imwrite("temp/boxes.jpg",new_img)
    #Determine the angle. Convert it to the value that was originally used to obtain skewed image
    angle = minAreaRect[-1]
    if angle < -45:
        angle = 90 + angle
    return -1.0 * angle


# eroded_img=thickening_font(noise_removal(cv2.imread('temp/blackwhiteimg.jpg')))
# cv2.imwrite('temp/thick.jpg',eroded_img)
img=cv2.imread('test1.jpeg')
print(get_skew_angle(img))