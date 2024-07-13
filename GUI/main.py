import sys
import tkinter as tk
from tkinter import filedialog, messagebox
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import cv2
from PIL import Image, ImageTk
from datetime import datetime
from options import DenoiseOptions, DeblurOptions
from util.video_to_numpy_array import get_video_frames

# Default directory for data
default_directory = "../data"
fixed_size = (640, 480)  # Fixed size for displayed images

# Color mapping for log levels
log_colors = {
    "info": "white",
    "warning": "yellow",
    "danger": "red",
    "comment": "blue",
}

# Function to upload video or folder of images
def upload_source():
    file_path = filedialog.askopenfilename(initialdir=default_directory, filetypes=[("Video files", "*.mp4;*.avi"), ("All files", "*.*")])
    if file_path:
        global source_path, displayed_image, source_data
        source_path = file_path
        source_data = get_video_frames(file_path)
        cap = cv2.VideoCapture(file_path)
        ret, frame = cap.read()
        if ret:
            displayed_image = frame  # Store the original image for further processing
            show_image(frame)
        cap.release()

# Function to display an image in the GUI
def show_image(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, fixed_size)
    img = Image.fromarray(frame_resized)
    imgtk = ImageTk.PhotoImage(image=img)
    image_canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
    image_canvas.image = imgtk

# Function to perform enhancement
def perform_enhancement():
    if not source_path:
        log_message("Please upload a source file before performing enhancements.", "danger")
        return

    selected_denoise = [option for var, option in zip(denoise_vars, denoise_options_list) if var.get()]
    selected_deblur = [option for var, option in zip(deblur_vars, deblur_options_list) if var.get()]

    if selected_denoise or selected_deblur:
        log_message(f"Enhancement selected: Denoise - {selected_denoise}, Deblur - {selected_deblur}, Source Path - {source_path}", "info")
    else:
        log_message("Please select at least one denoise or deblur method.", "warning")

    for method in selected_denoise:
        if method == DenoiseOptions.SCC:
            denoise_options.perform_SCC(source_path, source_data)
        if method == DenoiseOptions.ECC:
            denoise_options.perform_ECC(source_path, source_data)
        if method == DenoiseOptions.FEATURE_BASED:
            denoise_options.perform_FEATURE_BASED(source_path, source_data)
        if method == DenoiseOptions.OPTICAL_FLOW:
            denoise_options.perform_OPTICAL_FLOW(source_path, source_data)
        if method == DenoiseOptions.DENOISE_COT:
            denoise_options.perform_DENOISE_COT(source_path, source_data)
        if method == DenoiseOptions.RV_CLASSIC:
            denoise_options.perform_RV_CLASSIC(source_path, source_data)
        if method == DenoiseOptions.DENOISE_YOV:
            denoise_options.perform_DENOISE_YOV(source_path, source_data)

    for method in selected_deblur:
        if method == DeblurOptions.RV_OM:
            deblur_options.perform_RV_OM(source_path, source_data)
        if method == DeblurOptions.NAFNET:
            deblur_options.perform_NAFNET(source_path, source_data)
        if method == DeblurOptions.NUBKE:
            deblur_options.perform_NUBKE(source_path, source_data)
        if method == DeblurOptions.NUMBKE2WIN:
            deblur_options.perform_NUMBKE2WIN(source_path, source_data)
        if method == DeblurOptions.UNSUPERWIN:
            deblur_options.perform_UNSUPERWIN(source_path, source_data)

# Function to log messages with different levels
def log_message(message, level="info"):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    color = log_colors.get(level, "white")
    log_output.insert(tk.END, f"{timestamp} - {message}\n", level)
    log_output.tag_config(level, foreground=color)
    log_output.see(tk.END)

# Placeholder function for Edit Video
def edit_video():
    log_message("Edit Video function called.", "info")

# Function to save log
def save_log():
    log_message("Saving log...", "info")
    file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
    if file_path:
        with open(file_path, 'w') as file:
            log_text = log_output.get("1.0", tk.END)
            file.write(log_text)
        log_message("Log saved successfully.", "info")

# Initialize main window with ttkbootstrap theme
root = ttk.Window(themename="darkly")
root.title("Video Enhancement Tool")
root.geometry("1200x800")

# Create a Notebook for tabs
notebook = ttk.Notebook(root, bootstyle="info")
notebook.pack(fill=tk.BOTH, expand=True)

# Create frames for each tab
main_tab = ttk.Frame(notebook, padding=10)
log_tab = ttk.Frame(notebook, padding=10)

# Add tabs to notebook
notebook.add(main_tab, text="Main")
notebook.add(log_tab, text="Log")

# Variables
source_path = ""
source_data = []
displayed_image = None

# Create and place widgets in the main tab
upload_btn = ttk.Button(main_tab, text="Browse", command=upload_source, bootstyle="primary")
upload_btn.grid(row=0, column=0, padx=10, pady=10)

edit_video_btn = ttk.Button(main_tab, text="Video Editor", command=edit_video, bootstyle="primary")
edit_video_btn.grid(row=1, column=0, padx=10, pady=10)

# Frame for denoise methods
denoise_frame = ttk.Frame(main_tab, padding=10)
denoise_frame.grid(row=0, column=1, padx=10, pady=10, sticky='nw')

denoise_label = ttk.Label(denoise_frame, text="Select Denoise Methods:", font=('Arial', 12, 'bold'))
denoise_label.pack(anchor='w')

denoise_vars = []
denoise_options_list = [DenoiseOptions.SCC, DenoiseOptions.ECC, DenoiseOptions.FEATURE_BASED,
                        DenoiseOptions.OPTICAL_FLOW, DenoiseOptions.DENOISE_COT, DenoiseOptions.RV_CLASSIC, DenoiseOptions.DENOISE_YOV]
for option in denoise_options_list:
    var = tk.IntVar(value=0)
    chk = ttk.Checkbutton(denoise_frame, text=option, variable=var, bootstyle="success-round-toggle")
    chk.pack(anchor='w', padx=10, pady=2)
    denoise_vars.append(var)

# Frame for deblur methods
deblur_frame = ttk.Frame(main_tab, padding=10)
deblur_frame.grid(row=0, column=2, padx=10, pady=10, sticky='nw')

deblur_label = ttk.Label(deblur_frame, text="Select Deblur Methods:", font=('Arial', 12, 'bold'))
deblur_label.pack(anchor='w')

deblur_vars = []
deblur_options_list = [DeblurOptions.RV_OM, DeblurOptions.NAFNET, DeblurOptions.NUBKE,
                       DeblurOptions.NUMBKE2WIN, DeblurOptions.UNSUPERWIN]
for option in deblur_options_list:
    var = tk.IntVar(value=0)
    chk = ttk.Checkbutton(deblur_frame, text=option, variable=var, bootstyle="success-round-toggle")
    chk.pack(anchor='w', padx=10, pady=2)
    deblur_vars.append(var)

enhance_btn = ttk.Button(main_tab, text="Perform enhancement", command=perform_enhancement, bootstyle="primary")
enhance_btn.grid(row=1, column=1, columnspan=2, pady=20)

image_frame = ttk.Frame(main_tab, padding=10, bootstyle="info")
image_frame.grid(row=2, column=0, columnspan=3, pady=10)
image_canvas = tk.Canvas(image_frame, width=fixed_size[0], height=fixed_size[1], bg='#1C1C1C', highlightthickness=0)
image_canvas.pack(fill=tk.BOTH, expand=True)

# Create and place widgets in the log tab
log_frame = ttk.Frame(log_tab, padding=10, bootstyle="info")
log_frame.pack(fill=tk.BOTH, expand=True)
log_output = tk.Text(log_frame, height=30, width=95, bg='#1C1C1C', fg='white', font=('Arial', 10))
log_output.pack(fill=tk.BOTH, expand=True)

save_log_btn = ttk.Button(log_tab, text="Save log", command=save_log, bootstyle="primary")
save_log_btn.pack(pady=10)

# Create a menubar
menubar = tk.Menu(root)
root.config(menu=menubar)

# Create a settings menu
settings_menu = tk.Menu(menubar, tearoff=0)
menubar.add_cascade(label="Settings", menu=settings_menu)

# Add mode options to settings menu
mode_menu = tk.Menu(settings_menu, tearoff=0)
settings_menu.add_cascade(label="Mode", menu=mode_menu)
mode_menu.add_command(label="Dark", command=lambda: root.style.theme_use("darkly"))
mode_menu.add_command(label="Light", command=lambda: root.style.theme_use("flatly"))

# Initialize the denoise and deblur options with the log_message function
denoise_options = DenoiseOptions(log_message)
deblur_options = DeblurOptions(log_message)

# Run the GUI main loop
root.mainloop()
