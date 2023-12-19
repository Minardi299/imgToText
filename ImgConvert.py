# import pytesseract
# from langchain.llms import Ollama
from langchain.llms import Ollama
# import cv2
# import tkinter as tk
# from tkinter import filedialog
# root = tk.Tk()
# root.withdraw()

# file_path = filedialog.askopenfilename()
# pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\Tesseract-OCR\\tesseract.exe'
# img =cv2.imread(file_path)
print("hel;lo")
ollama = Ollama(base_url='http://localhost:11434',model = 'llama2')
print ("done")
print(ollama('why is the sky blue?'))
# text =pytesseract.image_to_string('test1.jpeg')  # this will print the text present in the image
# with open ("result.txt","a")as file:

#     file.write('==============New Image==============\n')
#     file.write(f'Image:\n')
#     file.write(text)
