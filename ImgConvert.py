import pytesseract
import sys
from pathlib import Path
from langchain.llms import ollama
import cv2
import tkinter as tk
import datetime
from date_time import get_current_time

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\Tesseract-OCR\\tesseract.exe'
to_check_folder =Path('TODO')
destination_folder=Path('Result')
def is_folder_empty(path: Path) -> bool:
    return not any(Path(path).iterdir())

def get_file_name(file_path: Path) -> str:
    return file_path.split('.')[0]

def img_to_string(file_path: Path) -> str:
    img =cv2.imread(str(file_path))
    text =pytesseract.image_to_string(img)  
    return text

def write_to_file(destination_path: Path,text: str) -> None:
    with open(destination_path, "w") as f:
        f.write(text)

def main(to_check: str, destination: str) -> None:
    if is_folder_empty(to_check):
       print(f"{to_check} Folder empty, please put image in the folder.")
       sys.exit()
    else:
        folder = Path(to_check)
        
        for file_path in folder.iterdir():
            text =img_to_string(file_path)
            destination_path= destination /f"{get_current_time()}.txt"
            with open(destination_path,'w') as file:
                file.write(text)
            file_path.unlink()
            




if __name__ == '__main__':
    main(to_check_folder, destination_folder)



