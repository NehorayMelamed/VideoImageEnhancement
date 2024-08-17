from RapidBase.import_all import *

import cv2
import numpy as np
import os

import os
import glob

def add_gaussian_noise(frame, std_dev):
    """
    Adds Gaussian noise to a video frame or image.

    :param frame: Input video frame or image (numpy array)
    :param std_dev: Standard deviation of the Gaussian noise
    :return: Noisy frame or image
    """
    noise = np.random.normal(0, std_dev, frame.shape).astype(np.float32)
    noisy_frame = frame.astype(np.float32) + noise
    noisy_frame = np.clip(noisy_frame, 0, 255).astype(np.uint8)
    return noisy_frame

def process_video(input_video_path, output_video_path, std_dev):
    # Check if the video file exists
    if not os.path.exists(input_video_path):
        print(f"Video file '{input_video_path}' not found.")
        return

    # Capture the video
    cap = cv2.VideoCapture(input_video_path)

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    codec = int(cap.get(cv2.CAP_PROP_FOURCC))

    # Create VideoWriter object to write the noisy video
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    # Process each frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Add Gaussian noise to the frame
        noisy_frame = add_gaussian_noise(frame, std_dev)

        # Write the noisy frame to the output video
        out.write(noisy_frame)

    # Release the resources
    cap.release()
    out.release()

    print(f"Noisy video saved as '{output_video_path}'")

def process_image_folder(input_folder_path, output_video_path, std_dev, fps=30):
    # Get list of image files in the folder
    image_files = sorted(glob.glob(os.path.join(input_folder_path, "*.png")) +
                         glob.glob(os.path.join(input_folder_path, "*.jpg")) +
                         glob.glob(os.path.join(input_folder_path, "*.jpeg")))

    if not image_files:
        print(f"No image files found in '{input_folder_path}'.")
        return

    # Read the first image to get the size
    first_image = cv2.imread(image_files[0])
    frame_height, frame_width, _ = first_image.shape

    # Create VideoWriter object to write the noisy video
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    # Process each image
    for img_path in image_files:
        frame = cv2.imread(img_path)

        # Add Gaussian noise to the image
        noisy_frame = add_gaussian_noise(frame, std_dev)

        # Write the noisy frame to the output video
        out.write(noisy_frame)

    # Release the VideoWriter object
    out.release()

    print(f"Noisy video created from images and saved as '{output_video_path}'")


input_type = "folder"  # Can be "video" or "folder"
std_dev = 40  # Standard deviation of the Gaussian noise

if input_type == "video":
    input_video_path = "path/to/your/input_video.mp4"
    output_video_path = "path/to/your/output_noisy_video.mp4"
    process_video(input_video_path, output_video_path, std_dev)
elif input_type == "folder":
    input_folder_path = r"C:\Users\orior\PycharmProjects\VideoImageEnhancement\data\images\videos\REDS\test_blur\004"
    output_video_path = r"C:\Users\orior\PycharmProjects\VideoImageEnhancement\data\images\videos\REDS/output_noisy_video_" + str(int(std_dev)) + '.mp4'
    process_image_folder(input_folder_path, output_video_path, std_dev)


