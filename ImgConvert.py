import pytesseract
import cv2
import tkinter as tk
from tkinter import filedialog
root = tk.Tk()
root.withdraw()

file_path = filedialog.askopenfilename()
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\Tesseract-OCR\\tesseract.exe'
img =cv2.imread(file_path)

text =pytesseract.image_to_string(img)  # this will print the text present in the image
with open ("result.txt","a")as file:

    file.write('==============New Image==============\n')
    file.write(f'Image:{file_path}\n')
    file.write(text)
    
