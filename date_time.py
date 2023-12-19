import datetime


def get_current_time() -> str:
    current_datetime = datetime.datetime.now()
    current_date = current_datetime.date()
    current_time = current_datetime.time()
    return  datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3]

def hello():
    print(get_current_time())

#unsed codes- future use
# from tkinter import filedialog
# root = tk.Tk()
# root.withdraw()
# file_path = filedialog.askopenfilename()

