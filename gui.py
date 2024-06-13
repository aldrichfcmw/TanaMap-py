
# This file was generated by the Tkinter Designer by Parth Jadhav
# https://github.com/ParthJadhav/Tkinter-Designer


from pathlib import Path

# from tkinter import *
# Explicit imports to satisfy Flake8
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage, filedialog,END
import os
import  tkinter as tk
import requests
import time
import exifread
import json
# import threading

OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path("assets/")

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)


def select_path():
    global input_path
    filetypes = [
        ("JPG files", "*.jpg"),
        ("KML files", "*.kml"),
    ]
    jpg_file = filedialog.askopenfilename(title="Select JPG file", filetypes=[("JPG files", "*.jpg")])
    if jpg_file:
        log_message(f"Selected JPG file: {jpg_file}")
    
    kml_file = filedialog.askopenfilename(title="Select KML file", filetypes=[("KML files", "*.kml")])
    if kml_file:
        log_message(f"Selected KML file: {kml_file}")

    input_path = os.path.dirname(jpg_file)

    # input_path = tk.filedialog.askdirectory()
    path_entry.delete(0, tk.END)
    path_entry.insert(0, input_path)

def btn_predict():
    input_path = path_entry.get()
    input_path = input_path.strip()
    token = token_entry.get()
    if not input_path:
        tk.messagebox.showerror(
            title="Invalid Path!", message="Enter a valid output path.")
        return
    if not token:
        tk.messagebox.showerror(
            title="Empty Fields!", message="Please enter Token.")
        return
    log_message("Initializing...")
    #time.sleep(3)
    # log_message("Checking Folder Path....")
    #time.sleep(2)
    # if not check_folder(input_path):
    #     log_message("Stopping...")
    #     #time.sleep(1.5)
    #     log_message("Stopped! No Image Exist")
    #     return
    #time.sleep(1.5)
    log_message("Checking Server Status....")
    #time.sleep(2)
    if not check_server():
        log_message("Please Chek Your Connection!")
        return
    # log_message("Checking YoloV5...")
    #time.sleep(3)
    # check_yolo()
    #time.sleep(3)
    log_message("Checking YoloV5 Model...")
    #time.sleep(3)
    check_model()
    #time.sleep(3)
    # log_message("Processing Detection...")
    #time.sleep(3)
    # predict(input_path)
    #time.sleep(3)
    # log_message("Checking Data...")
    #time.sleep(3)
    # cek_data = check_data(input_path)
    # if not cek_data:
        # log_message("Stopping...")
        #time.sleep(1.5)
        # log_message("Stopped! No Image Exist")
        # return
    # log_message("Uploading Data To Server...")
    #time.sleep(3)
    # upload_data(token,cek_data)


def log_message(message, newline=True):
    # log_entry.config(state=tk.NORMAL)
    log_entry.insert(tk.END, message + ('\n' if newline else ''))
    # log_entry.config(state=tk.DISABLED)
    # log_entry.yview(tk.END)
    log_entry.see(tk.END)
    log_entry.update()

def check_folder(folder_path):
    image_files = [file for file in os.listdir(folder_path) if file.endswith(('.jpg', '.png','.JPG', '.PNG'))]
    if image_files:
        log_message(f"Found {len(image_files)} files JPG/PNG in {folder_path}")
        return True
    else:
        log_message(f"Not Found file JPG/PNG in {folder_path}!")
        return False

def check_server():
    global status_code, url
    
    url = "https://tanamap.drik.my.id"  # Ganti dengan URL server yang ingin diperiksa
    try:
        response = requests.get(url)
        status_code = response.status_code
        canvas.itemconfig(status_entry, text=status_code)
        if status_code == 200:
            canvas.itemconfig(image_6, state="hidden")
            canvas.itemconfig(image_5, state="normal")  # Menampilkan gambar 5
            canvas.itemconfig(image_4, state="hidden")  # Sembunyikan gambar 4
            log_message(f"Server {url} is up and running")
            return True
        else:
            canvas.itemconfig(status_entry, text=status_code)
            canvas.itemconfig(image_6, state="hidden")
            canvas.itemconfig(image_4, state="normal")  # Menampilkan gambar 4
            canvas.itemconfig(image_5, state="hidden")  # Sembunyikan gambar 5
            log_message(f"Server {url} is down. Status code: {response.status_code}")
            return False
    except requests.ConnectionError:
        canvas.itemconfig(status_entry, text="?")
        canvas.itemconfig(image_6, state="hidden")
        canvas.itemconfig(image_4, state="normal")  # Menampilkan gambar 4
        canvas.itemconfig(image_5, state="hidden")  # Sembunyikan gambar 5
        log_message("Failed to connect to the server")
        return False

def check_yolo():
    directory_path = os.path.dirname(__file__)
    folder_name = os.path.join(directory_path,"yolov5")
    if os.path.exists(folder_name):
        log_message(f"Yolov5 is exists, Updating...")
        command = ["git", "pull"]
    else:
        log_message(f"Yolov5 doesn't exist!")
        log_message(f"Downloading yolov5....")
        command = ["git", "clone", "https://github.com/ultralytics/yolov5.git"]
        sub_command(command,directory_path)
    log_message("Yolov5 berhasil diupdate")
    
def sub_command(command,directory_path):
    import subprocess
    log_entry.insert(tk.END, command)
    log_message("")
    process = subprocess.Popen(command, cwd=directory_path, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, universal_newlines=True)
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            log_message(output.strip())
            # print(output.strip())

def check_model():
    directory_path = os.path.dirname(__file__)
    destination_directory = os.path.join(directory_path, "yolov5", "runs", "models")
    
    if os.path.exists(destination_directory):
        log_message("Model directory exists")
    else:
        log_message("Model directory doesn't exist! Creating...")
        os.makedirs(destination_directory, exist_ok=True)

    save_path = os.path.join(destination_directory, 'best.pt')
    
    if os.path.exists(save_path):
        log_message("Model file exists")
    else:
        log_message("Model file doesn't exist. Downloading...")
        url = "http://tanamap.drik.my.id/model/best.pt"  # Ganti dengan URL yang sesuai
        download_model(url,save_path)
        # threading.Thread(target=download_model, args=(url, save_path)).start()

def download_model(url, save_path):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    chunk_size = 1024  # Ukuran chunk yang ingin Anda gunakan

    with open(save_path, 'wb') as file:
        downloaded = 0
        log_message("Downloading: [", newline=False)
        last_printed_progress = 0
        for data in response.iter_content(chunk_size=chunk_size):
            file.write(data)
            downloaded += len(data)
            percent_done = (downloaded / total_size) * 100
            print(f"Downloaded: {percent_done:.2f}%\r", end='', flush=True)
            if int(percent_done) // 5 > last_printed_progress:
                last_printed_progress = int(percent_done) // 5
                log_entry.insert(tk.END, "#")
                log_entry.see(tk.END)
                log_entry.update()
            # if(percent_done)
            # log_entry.insert(tk.END, "#")
        log_message("]", newline=False)
    log_message("\nFile successfully downloaded")

def predict(input_path):
    directory_path = os.path.dirname(__file__)
    folder_name = os.path.join(directory_path,"yolov5")
    command =[
        "python",
        "segment/predict.py",
        "--iou-thres", "0.45",
        "--line-thickness", "0",
        "--weights", "runs/models/best.pt",
        "--img", "640",
        "--conf-thres", "0.25",
        "--source", input_path
    ]
    sub_command(command,folder_name)
    log_message("Detection Complete")

def dms_to_decimal(dms):
    return dms[0].num / dms[0].den + dms[1].num / (dms[1].den * 60) + dms[2].num / (dms[2].den * 3600)

def read_gps_data(folder):
    gps_data = {}
    for filename in os.listdir(folder):
        if filename.endswith(('.jpg', '.png','.JPG', '.PNG')):
            with open(os.path.join(folder, filename), 'rb') as file:
                tags = exifread.process_file(file)
                if 'GPS GPSLatitude' in tags and 'GPS GPSLongitude' in tags:
                    lat_ref = str(tags['GPS GPSLatitudeRef'])
                    lon_ref = str(tags['GPS GPSLongitudeRef'])
                    lat = dms_to_decimal(tags['GPS GPSLatitude'].values)
                    lon = dms_to_decimal(tags['GPS GPSLongitude'].values)
                    if lat_ref == 'S':
                        lat *= -1
                    if lon_ref == 'W':
                        lon *= -1
                    gps_data[filename] = {
                        'latitude': str(lat),
                        'longitude': str(lon)
                    }
    return gps_data 

def check_data(input_path):
    directory_path = os.path.dirname(__file__)
    directory_path = os.path.join(directory_path,"yolov5","runs","predict-seg")
    folders = [folder for folder in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, folder))]
    folders.sort(key=lambda x: os.path.getctime(os.path.join(directory_path, x)), reverse=True)
    latest_folder = os.path.join(directory_path, folders[0])
    if not check_folder(latest_folder):
        log_message("Stopping...")
        #time.sleep(1.5)
        log_message("Stopped! No Image Exist")
        return
    image_list = [file for file in os.listdir(latest_folder) if file.endswith(('.jpg', '.png','.JPG', '.PNG'))]
    gps_data = read_gps_data(input_path)
    output_data = []
    for image_filename in image_list:
        if image_filename in gps_data:
            gps_info = gps_data[image_filename]
            # log_message(f"Nama Gambar: {image_filename}, Latitude {gps_info['latitude']}, Longitude {gps_info['longitude']}")
            latitude = str(gps_info['latitude'])
            longitude = str(gps_info['longitude'])
            output_data.append({
                "image": image_filename,
                "location": {
                    "latitude": gps_info['latitude'],
                    "longitude": gps_info['longitude']
                }
            })
    # print(json.dumps(output_data, indent=4))
    log_entry.insert(tk.END, json.dumps(output_data, indent=4))
    log_entry.see(tk.END)
    log_entry.update()
    log_message("")
    return output_data


def upload_data(token_bearer,output_data):
    url = 'http://tanamap.drik.my.id/api/upload-data-hpt'

    headers = {'Authorization': f'Bearer {token_bearer}', 'Content-Type': 'application/json'}

    response = requests.post(url, json=output_data, headers=headers)

    if response.status_code == 200:
        log_message("Data is successfully sent to the server")
    else:
        log_message(f"Failed to send data to the server. Status code:{response.status_code}" )
    return response.status_code

window = Tk()
window.title("Tanamap")
window.geometry("700x650")
window.configure(bg = "#F5F5F9")
window.iconbitmap(relative_to_assets("favicon.ico"))


canvas = Canvas(
    window,
    bg = "#F5F5F9",
    height = 650,
    width = 700,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge"
)

canvas.place(x = 0, y = 0)
image_image_1 = PhotoImage(
    file=relative_to_assets("image_1.png"))
image_1 = canvas.create_image(
    350.0,
    57.0,
    image=image_image_1
)

image_image_2 = PhotoImage(
    file=relative_to_assets("image_2.png"))
image_2 = canvas.create_image(
    350.0,
    352.0,
    image=image_image_2
)

image_image_3 = PhotoImage(
    file=relative_to_assets("image_3.png"))
image_3 = canvas.create_image(
    350.0,
    57.0,
    image=image_image_3
)

button_image_1 = PhotoImage(
    file=relative_to_assets("button_1.png"))
button_1 = Button(
    image=button_image_1,
    borderwidth=0,
    highlightthickness=0,
    command=check_server,
    relief="flat"
)
button_1.place(
    x=561.0,
    y=606.0,
    width=24.0,
    height=24.0
)

image_image_4 = PhotoImage(
    file=relative_to_assets("image_4.png"))
image_4 = canvas.create_image(
    674.0,
    618.0,
    image=image_image_4
)

image_image_5 = PhotoImage(
    file=relative_to_assets("image_5.png"))
image_5 = canvas.create_image(
    674.0,
    618.0,
    image=image_image_5
)

image_image_6 = PhotoImage(
    file=relative_to_assets("image_6.png"))
image_6 = canvas.create_image(
    674.0,
    618.0,
    image=image_image_6
)

canvas.create_text(
    592.0,
    611.0,
    anchor="nw",
    text="Status:",
    fill="#566A7F",
    font=("Public Sans", 12 * -1)
)

status_entry = canvas.create_text(
    632.0,
    611.0,
    anchor="nw",
    text=" ?",
    fill="#566A7F",
    font=("Public Sans", 12 * -1)
)

image_image_7 = PhotoImage(
    file=relative_to_assets("image_7.png"))
image_7 = canvas.create_image(
    350.0,
    442.0,
    image=image_image_7
)

entry_image_1 = PhotoImage(
    file=relative_to_assets("entry_1.png"))
entry_bg_1 = canvas.create_image(
    350.0,
    443.5,
    image=entry_image_1
)
log_entry = entry_1 = Text(
    bd=0,
    bg="#FFFFFF",
    fg="#000716",
    highlightthickness=0
)
entry_1.place(
    x=52.0,
    y=325.0,
    width=596.0,
    height=235.0
)

image_image_8 = PhotoImage(
    file=relative_to_assets("image_8.png"))
image_8 = canvas.create_image(
    288.0,
    277.0,
    image=image_image_8
)

entry_image_2 = PhotoImage(
    file=relative_to_assets("entry_2.png"))
entry_bg_2 = canvas.create_image(
    289.0,
    277.0,
    image=entry_image_2
)
token_entry = entry_2 = Entry(
    bd=0,
    bg="#FFFFFF",
    fg="#000716",
    highlightthickness=0
)
entry_2.place(
    x=50.0,
    y=259.0,
    width=478.0,
    height=34.0
)

canvas.create_text(
    40.0,
    224.0,
    anchor="nw",
    text="Token ID",
    fill="#566A7F",
    font=("Public Sans", 12 * -1)
)

image_image_9 = PhotoImage(
    file=relative_to_assets("image_9.png"))
image_9 = canvas.create_image(
    297.0,
    184.0,
    image=image_image_9
)

entry_image_3 = PhotoImage(
    file=relative_to_assets("entry_3.png"))
entry_bg_3 = canvas.create_image(
    298.0,
    184.0,
    image=entry_image_3
)
path_entry = entry_3 = Entry(
    bd=0,
    bg="#FFFFFF",
    fg="#000716",
    highlightthickness=0
)
entry_3.place(
    x=51.0,
    y=166.0,
    width=494.0,
    height=34.0
)

button_image_2 = PhotoImage(
    file=relative_to_assets("button_2.png"))
button_2 = Button(
    image=button_image_2,
    borderwidth=0,
    highlightthickness=0,
    command=select_path,
    relief="flat"
)
button_2.place(
    x=555.0,
    y=159.0,
    width=105.0,
    height=50.0
)

canvas.create_text(
    40.0,
    130.0,
    anchor="nw",
    text="Input File",
    fill="#566A7F",
    font=("Public Sans", 12 * -1)
)

button_image_3 = PhotoImage(
    file=relative_to_assets("button_3.png"))
button_3 = Button(
    image=button_image_3,
    borderwidth=0,
    highlightthickness=0,
    command=btn_predict,
    relief="flat"
)
button_3.place(
    x=546.0,
    y=253.0,
    width=114.0,
    height=48.0
)
window.resizable(False, False)
window.mainloop()
