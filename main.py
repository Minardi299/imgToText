import pytesseract
import sys
from pathlib import Path
from langchain.llms import ollama
import cv2
from pdf_processing import *
from date_time import get_current_time
from ImgConvert import *

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\Tesseract-OCR\\tesseract.exe'
to_check_folder =Path('TODO')
destination_folder=Path('Result')

def is_folder_empty(path: Path) -> bool:
    return not any(Path(path).iterdir())

def get_file_name(file_path: Path) -> str:
    return file_path.stem

def write_to_file(destination_path: Path,text: str) -> None:
    with open(destination_path, "w") as f:
        f.write(text)

def get_file_extension(file_path: Path) -> str:
    return file_path.suffix


def main(to_check: str, destination: str) -> None:
    if is_folder_empty(to_check):
       print(f"{to_check} Folder empty, please put image in the folder.")
       sys.exit()
    else:
        folder = Path(to_check)
        
        for file_path in folder.iterdir():
            text=''
            file_extension=get_file_extension(file_path)
            if file_extension == '.pdf':
                text += extract_pdf(file_path)
            elif file_extension =='.jpeg' or file_extension =='.png' or file_extension == '.jpg':
                img=cv2.imread(str(file_path))
                text += img_to_string(img)

            destination_path = destination /f"{get_current_time()}_{get_file_name(file_path)}.txt"
            with open(destination_path,'w') as file:
                file.write(text)
            destination_file = destination / f"{get_file_name(file_path)}{get_file_extension(file_path)}" # New file path in the destination directory
            file_path.rename(destination_file)
            




if __name__ == '__main__':
    
    main(to_check_folder,destination_folder)

