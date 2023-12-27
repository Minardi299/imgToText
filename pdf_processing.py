import fitz
import numpy as np
import cv2
from pathlib import Path
from ImgConvert import *


def is_searchable(path: Path) -> bool:
    doc= fitz.open(path)
    for page in doc:
        text = page.get_text("text")
        if len(text.strip()):
            doc.close()
            return True
    doc.close()
    return False

def extract_searchable(path: Path) ->str:
    doc = fitz.open(path) # open a document
    text=''
    for page in doc:
        text+=page.get_text()
        
    return text

def extract_scanned(path: Path) :
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
        text += img_to_stringpdf(cv2_image)

    return text

def extract_pdf(path: Path) ->str:
    text=''
    if is_searchable(path):
        text += extract_searchable(path)
    else:
        text += extract_scanned(path)
    return text


# print(extract_pdf(Path('Pere.pdf')))
# # print(is_searchable(Path('Pere.pdf')))
