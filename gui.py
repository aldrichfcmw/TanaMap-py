from pathlib import Path
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage, filedialog,END
import  tkinter as tk

import os, time, json, requests, cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from osgeo import gdal
from skimage import io
from xml.etree import ElementTree as ET

OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path("assets/")

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

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
    time.sleep(3)
    log_message("Checking Server Status....")
    time.sleep(2)
    server =  check_server()
    if not server:
        log_message("Please Chek Your Connection!")
    time.sleep(2)
    log_message("Checking YoloV5...")
    time.sleep(3)
    check_yolo()
    time.sleep(3)
    log_message("Checking YoloV5 Model...")
    time.sleep(3)
    check_model()
    time.sleep(3)
    log_message("Processing Detection...")
    time.sleep(3)
    predict(jpg_file)
    time.sleep(3)
    log_message("Checking Data...")
    time.sleep(3)
    get_latest_folder()
    time.sleep(3)
    log_message("Processing Image...")
    time.sleep(3)
    process_crop_images()
    time.sleep(3)
    log_message("Saving JSON...")
    time.sleep(3)
    save_results_to_json()
    time.sleep(3)
    log_message("Uploading Data To Server...")
    time.sleep(3)
    upload_data(token)

def log_message(message, newline=True):
    # log_entry.config(state=tk.NORMAL)
    log_entry.insert(tk.END, message + ('\n' if newline else ''))
    # log_entry.config(state=tk.DISABLED)
    # log_entry.yview(tk.END)
    log_entry.see(tk.END)
    log_entry.update()

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

def select_path():
    global input_path, jpg_file, kml_file
    filetypes = [
        ("JPG files", "*.jpg"),
        ("KML files", "*.kml"),
    ]
    jpg_file = filedialog.askopenfilename(title="Select JPG file", filetypes=[("JPG files", "*.jpg")])
    if jpg_file:
        log_message(f"Selected JPG file: {jpg_file}")
    
    input_path = os.path.dirname(jpg_file)
    path_entry.delete(0, tk.END)
    path_entry.insert(0, input_path)

    kml_file = filedialog.askopenfilename(title="Select KML file", filetypes=[("KML files", "*.kml")])
    if kml_file:
        log_message(f"Selected KML file: {kml_file}")   


def check_server():
    global status_code, url
    
    url = "https://tanamap.drik.my.id" 
    try:
        response = requests.get(url)
        status_code = response.status_code
        canvas.itemconfig(status_entry, text=status_code)
        if status_code == 200:
            canvas.itemconfig(image_6, state="hidden")
            canvas.itemconfig(image_5, state="normal")
            canvas.itemconfig(image_4, state="hidden")
            log_message(f"Server {url} is up and running")
            return True
        else:
            canvas.itemconfig(status_entry, text=status_code)
            canvas.itemconfig(image_6, state="hidden")
            canvas.itemconfig(image_4, state="normal")
            canvas.itemconfig(image_5, state="hidden")
            log_message(f"Server {url} is error. Status code: {response.status_code}")
            return False
    except requests.ConnectionError:
        canvas.itemconfig(status_entry, text="?")
        canvas.itemconfig(image_6, state="hidden")
        canvas.itemconfig(image_4, state="normal")
        canvas.itemconfig(image_5, state="hidden")
        log_message("Failed to connect to the server")
        return False

def check_yolo():
    global yolo_path
    yolo_path = os.path.join(OUTPUT_PATH,"yolov5")
    if os.path.exists(yolo_path):
        log_message("Yolov5 is exists, Updating...")
        command = ["git", "pull"]
        log_message("Already up to date.")
    else:
        log_message("Yolov5 doesn't exist!")
        log_message("Downloading Yolov5....")
        command = ["git", "clone", "https://github.com/ultralytics/yolov5.git"]
        sub_command(command,OUTPUT_PATH)
        log_message("Yolov5 Updated.")
    log_message("Installing requirements...")
    command = ["pip","install","-r","requirements.txt"]
    sub_command(command, yolo_path)
    log_message("All Required alredy installed.")

def check_model():
    destination_directory = os.path.join(yolo_path, "runs", "models")
    log_message("Checking Model...")
    time.sleep(2)
    if os.path.exists(destination_directory):
        log_message("Model directory exists")
    else:
        log_message("Model directory doesn't exist! Creating...")
        os.makedirs(destination_directory, exist_ok=True)

    model_path = os.path.join(destination_directory, 'best.pt')
    
    if os.path.exists(model_path):
        log_message("Model file exists")
    else:
        log_message("Model file doesn't exist. Downloading...")
        url = "http://tanamap.drik.my.id/models/disease-best.pt"
        download_file(url,model_path)
        log_message("\nSuccessfully downloaded model")

def download_file(url, save_path):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    chunk_size = 1024 

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
        log_message("]  100%", newline=False)
    log_message("\nSuccessfully downloaded")

def predict(input_path):
    log_message("Starting Detection...")
    folder_name = os.path.join(OUTPUT_PATH,"yolov5")
    command =[
        "python",
        "segment/predict.py",
        "--iou-thres", "0.45",
        "--line-thickness", "0",
        "--weights", "runs/models/best.pt",
        "--img", "640",
        "--conf-thres", "0.5",
        "--source", input_path, 
        "--save-crop"
    ]
    sub_command(command,folder_name)
    log_message("Detection Complete.")

def get_latest_folder():
    parent_folder = os.path.join(OUTPUT_PATH, "yolov5", "runs","predict-seg" ) 
    folders = [os.path.join(parent_folder,folder ) for folder in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, folder))]
    global latest_folder 
    latest_folder = max(folders, key=os.path.getmtime)
    log_message(latest_folder)

# Membaca file KML
def read_kml(kml_path):
    tree = ET.parse(kml_path)
    root = tree.getroot()
    namespace = {'kml': 'http://www.opengis.net/kml/2.2'}

    north = float(root.find('.//kml:LatLonBox/kml:north', namespace).text)
    south = float(root.find('.//kml:LatLonBox/kml:south', namespace).text)
    east = float(root.find('.//kml:LatLonBox/kml:east', namespace).text)
    west = float(root.find('.//kml:LatLonBox/kml:west', namespace).text)

    return north, south, east, west

# Mendapatkan ukuran gambar orthomosaic
def get_image_size(image_path):
    dataset = gdal.Open(image_path)
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    return width, height

# Hitung geo-transform dari informasi KML
def calculate_geo_transform(north, south, east, west, image_width, image_height):
    pixel_width = (east - west) / image_width
    pixel_height = (north - south) / image_height
    return (west, pixel_width, 0, north, 0, -pixel_height)

# Fungsi untuk mengonversi koordinat piksel ke koordinat geografis
def pixel_to_geo(pixel_x, pixel_y, geo_transform):
    geo_x = geo_transform[0] + pixel_x * geo_transform[1] + pixel_y * geo_transform[2]
    geo_y = geo_transform[3] + pixel_x * geo_transform[4] + pixel_y * geo_transform[5]
    return geo_y, geo_x

# Fungsi untuk menemukan posisi crop dalam orthomosaic menggunakan OpenCV
def find_crop_position(orthomosaic_path, crop_path):
    orthomosaic = cv2.imread(orthomosaic_path)
    crop = cv2.imread(crop_path)

    result = cv2.matchTemplate(orthomosaic, crop, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(result)

    return max_loc

# Fungsi untuk menghitung Green Leaf Index (GLI)
def calculate_gli(image):
    R = image[:,:,0]
    G = image[:,:,1]
    B = image[:,:,2]
    gli = (2 * G - R - B / 2 * G + R + B)+1e-6
    return gli

# Fungsi untuk menganalisis kesehatan tanaman berdasarkan GLI
def analyze_health(gli_image):
    gli_in_range = (gli_image >= -20000) & (gli_image <= -4000)
    unhealthy_area = np.sum(gli_in_range)
    total_area = gli_image.size
    unhealthy_percentage = unhealthy_area / total_area * 100
    return unhealthy_area, total_area, unhealthy_percentage

# Fungsi utama untuk memproses gambar crop
def process_crop_images():
    global results_json
    kml_path = kml_file
    orthomosaic_path = jpg_file
    crops_folder_path = os.path.join(latest_folder,"crops","rice-fields")
    save_path = os.path.join(latest_folder,"crops","rice-fields-gli")
    os.makedirs(save_path, exist_ok=True)

    north, south, east, west = read_kml(kml_path)
    # print("North:", north)
    # print("South:", south)
    # print("East:", east)
    # print("West:", west)
    log_message(f"North: {north}, South: {south},\nEast: {east}, West: {west}")

    # Langkah 2: Dapatkan ukuran sebenarnya dari gambar orthomosaic
    image_width, image_height = get_image_size(orthomosaic_path)
    # print("Ukuran gambar orthomosaic: width =", image_width, ", height =", image_height)
    log_message(f"Ukuran gambar orthomosaic: width={image_width}, height={image_height}")

    # Langkah 3: Hitung geo-transform dari informasi KML dan ukuran gambar
    geo_transform = calculate_geo_transform(north, south, east, west, image_width, image_height)
    # print("GeoTransform:", geo_transform)
    log_message(f"GeoTransform: {geo_transform}")

    # Langkah 4: Proses setiap file crop dalam folder
    crop_files = [f for f in os.listdir(crops_folder_path) if f.endswith(('.jpg', '.png'))]

    results_json = []

    for crop_file in crop_files:
        crop_path = os.path.join(crops_folder_path, crop_file)

        # Temukan posisi crop dalam orthomosaic
        crop_position = find_crop_position(orthomosaic_path, crop_path)
        # print(f"Posisi crop {crop_file} dalam piksel:", crop_position)
        log_message(f"\nPosisi crop {crop_file} dalam piksel: {crop_position}" )

        # Konversi posisi crop ke koordinat geografis
        crop_geo_coord = pixel_to_geo(crop_position[1], crop_position[0], geo_transform)
        # print(f"Koordinat geografis dari crop {crop_file}:", crop_geo_coord)
        log_message(f"Koordinat geografis dari crop {crop_file}: {crop_geo_coord}")

        # Muat citra crop
        crop_image = io.imread(crop_path) # Konversi dari BGR ke RGB

        R = crop_image[:,:,0]
        G = crop_image[:,:,1]
        B = crop_image[:,:,2]

        # Hitung Green Leaf Index (GLI)
        # gli_image = (2 * G - R - B / 2 * G + R + B)+1e-6
        gli_image = calculate_gli(crop_image)
        # print(gli_image)
        gli_in_range = (gli_image >= -20000) & (gli_image <= -4000)
        reverse_colormap = plt.cm.RdYlGn.reversed()
        plt.figure(figsize=(5,5))
        plt.imshow(gli_in_range, cmap=reverse_colormap)
        plt.axis('off')
        save_file = os.path.join(save_path, os.path.splitext(crop_file)[0] + ".png")
        plt.savefig(save_file)
        plt.close()
        gli_file = os.path.splitext(crop_file)[0] + ".png"

        # Analisis kesehatan tanaman berdasarkan GLI
        unhealthy_area, total_area, unhealthy_percentage = analyze_health(gli_image)
        log_message(f"Area Tanaman Tidak Sehat: {unhealthy_area} pixels")
        log_message(f"Total Area: {total_area} pixels")
        log_message(f"Persentase Area Tanaman Tidak Sehat: {unhealthy_percentage:.2f}%")

        healthy_area = total_area - unhealthy_area
        healthy_percentage = 100 - unhealthy_percentage
        log_message(f"Area Tanaman Sehat: {healthy_area} pixels")
        log_message(f"Total Area: {total_area} pixels")
        log_message(f"Persentase Area Tanaman Sehat: {healthy_percentage:.2f}%")

        health_status = ""
        if unhealthy_percentage  < 25:
            health_statusInt = 0
            health_status = "Kebanyakan tanaman dalam kondisi sehat."
        else:
            health_statusInt = 1
            health_status = "Banyak tanaman dalam kondisi sehat, namun ada beberapa area yang perlu diperhatikan."
        log_message(f"Health Status:{health_status}")

        # Buat link untuk membuka koordinat di Google Maps
        # google_maps_link = f"https://www.google.com/maps?q={crop_geo_coord[0]},{crop_geo_coord[1]}"
        # print(f"Link Google Maps {crop_file}:", google_maps_link)
        
        results_json.append({
            "crop_file": gli_file,
            # "pixel_position": [int(crop_position[0]), int(crop_position[1])],  # Convert to Python int
            "latitude":float(crop_geo_coord[0]),
            "longitude":float(crop_geo_coord[1]),
            # "geo_coordinates": [float(crop_geo_coord[0]), float(crop_geo_coord[1])],  # Convert to Python float
            # "google_maps_link": google_maps_link,
            "healthy_area": int(healthy_area),
            "total_area": int(total_area),
            "healthy_percentage": float(healthy_percentage),
            "health_status": health_statusInt,
            "status": health_status
        })
    log_entry.insert(tk.END, json.dumps(results_json, indent=4))
    log_entry.see(tk.END)
    log_entry.update()

# Fungsi untuk menyimpan hasil ke file JSON
def save_results_to_json():
    json_path = os.path.join(latest_folder,"results.json")
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=4)
    # print(f"Hasil telah disimpan di {json_path}")
    log_message(f"Hasil telah disimpan di {json_path}")

def upload_data(token_bearer):
    url = 'https://tanamap.drik.my.id/api/disease-data'

    headers = {'Authorization': f'Bearer {token_bearer}'}

    # response = requests.post(url, json=output_data, headers=headers)

    # Untuk setiap data, kirim gambar dalam format base64
    for item in results_json:
        file_path = os.path.join(latest_folder,'crops','rice-fields-gli', item['crop_file'])
        files = {
            'crop_image': open(file_path, 'rb')
        }
        payload = {
            "latitude": item['latitude'],
            "longitude": item['longitude'],
            "healthy_area": item['healthy_area'],
            "total_area": item['total_area'],
            "healthy_percentage": item['healthy_percentage'],
            "health_status": item['health_status'],
            "status": item['status']
        }
        
        response = requests.post(url, headers=headers, files=files, data=payload)
        if response.status_code == 200:
            # print('Data stored successfully:', response.json())
            log_message(f'Data stored successfully:{response.json()}')
        else:
            # print('Failed to store data:', response.text) 
            log_message(f'Failed to store data:{response.json()}')
        with open('test.txt', 'w', encoding='utf-8') as f:
            f.write(response.text)

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
    56.0,
    image=image_image_1
)

image_image_2 = PhotoImage(
    file=relative_to_assets("image_2.png"))
image_2 = canvas.create_image(
    350.0,
    354.0,
    image=image_image_2
)

image_image_3 = PhotoImage(
    file=relative_to_assets("image_3.png"))
image_3 = canvas.create_image(
    350.0,
    56.0,
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
    y=612.0,
    width=25.0,
    height=25.0
)

image_image_4 = PhotoImage(
    file=relative_to_assets("image_4.png"))
image_4 = canvas.create_image(
    674.0,
    624.0,
    image=image_image_4
)

image_image_5 = PhotoImage(
    file=relative_to_assets("image_5.png"))
image_5 = canvas.create_image(
    674.0,
    624.0,
    image=image_image_5
)

image_image_6 = PhotoImage(
    file=relative_to_assets("image_6.png"))
image_6 = canvas.create_image(
    674.0,
    624.0,
    image=image_image_6
)

canvas.create_text(
    590.0,
    617.0,
    anchor="nw",
    text="Status:",
    fill="#566A7F",
    font=("Public Sans", 12 * -1)
)

status_entry = canvas.create_text(
    632.0,
    617.0,
    anchor="nw",
    text=" ?",
    fill="#566A7F",
    font=("Public Sans", 12 * -1)
)

image_image_7 = PhotoImage(
    file=relative_to_assets("image_7.png"))
image_7 = canvas.create_image(
    350.0,
    448.0,
    image=image_image_7
)

entry_image_1 = PhotoImage(
    file=relative_to_assets("entry_1.png"))
entry_bg_1 = canvas.create_image(
    350.0,
    449.5,
    image=entry_image_1
)
log_entry = entry_1 = Text(
    bd=0,
    bg="#FFFFFF",
    fg="#000716",
    highlightthickness=0
)
entry_1.place(
    x=46.0,
    y=331.0,
    width=608.0,
    height=235.0
)

image_image_8 = PhotoImage(
    file=relative_to_assets("image_8.png"))
image_8 = canvas.create_image(
    285.0,
    291.0,
    image=image_image_8
)

entry_image_2 = PhotoImage(
    file=relative_to_assets("entry_2.png"))
entry_bg_2 = canvas.create_image(
    285.0,
    291.0,
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
    y=276.0,
    width=470.0,
    height=28.0
)

canvas.create_text(
    40.0,
    241.0,
    anchor="nw",
    text="Token ID",
    fill="#566A7F",
    font=("Public Sans", 17 * -1)
)

canvas.create_text(
    50.0,
    211.0,
    anchor="nw",
    text="*Mohon input file dalam format JPG dan KML",
    fill="#566A7F",
    font=("Public Sans", 12 * -1)
)

image_image_9 = PhotoImage(
    file=relative_to_assets("image_9.png"))
image_9 = canvas.create_image(
    297.0,
    185.0,
    image=image_image_9
)

entry_image_3 = PhotoImage(
    file=relative_to_assets("entry_3.png"))
entry_bg_3 = canvas.create_image(
    297.5,
    185.0,
    image=entry_image_3
)
path_entry = entry_3 = Entry(
    bd=0,
    bg="#FFFFFF",
    fg="#000716",
    highlightthickness=0
)
entry_3.place(
    x=50.0,
    y=170.0,
    width=495.0,
    height=28.0
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
    y=163.0,
    width=105.0,
    height=44.0
)

canvas.create_text(
    40.0,
    135.0,
    anchor="nw",
    text="Input File",
    fill="#566A7F",
    font=("Public Sans", 17 * -1)
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
    y=269.0,
    width=114.0,
    height=44.0
)
window.resizable(False, False)
window.mainloop()
