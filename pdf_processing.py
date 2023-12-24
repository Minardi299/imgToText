import fitz
import numpy as np
import cv2
from ImgConvert import *

pdf_path='p3.pdf'


def extract_searchable(path: str) ->str:
    doc = fitz.open(path) # open a document
    text=''
    out =open(path,'wb')
    for page in doc:
        text=page.get_text().encode("utf8")
        out.write(text)
        out.write(bytes((12,)))
        out.close
    return text

def extract_scanned(path: str) :
    text=''
    doc=fitz.open(path)
    zoom = 1.2
    mat = fitz.Matrix(zoom, zoom)

    for page_index in range(len(doc)):
        page = doc[page_index] #number of page
        img = page.get_images()

        pix = page.get_pixmap(matrix = mat)
        img=pix.pil_tobytes("JPEG")
        cv2_image = cv2.imdecode(np.frombuffer(bytearray(img), dtype=np.uint8), cv2.IMREAD_COLOR)
        

    return text


    # for page_index in range(len(doc)): # iterate over pdf pages
    #     page = doc[page_index] # get the page
    #     image_list = page.get_images()

    #     # print the number of images found on the page
    #     if image_list:
    #         print(f"Found {len(image_list)} images on page {page_index}")
    #     else:
    #         print("No images found on page", page_index)

    #     for image_index, img in enumerate(image_list, start=1): # enumerate the image list
    #         xref = img[0] # get the XREF of the image
    #         pix = fitz.Pixmap(doc, xref) # create a Pixmap

    #         if pix.n - pix.alpha > 3: # CMYK: convert to RGB first
    #             pix = fitz.Pixmap(fitz.csRGB, pix)

    #         pix.save(f"page_{page_index}_image_{image_index}.png") # save the image as png
    #         pix = None


extract_scanned(pdf_path)