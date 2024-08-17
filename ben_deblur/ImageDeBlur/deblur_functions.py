import torch
import sys
import PARAMETER

from ben_deblur.ImageDeBlur.basicsr.models import create_model
from ben_deblur.ImageDeBlur.basicsr.utils import img2tensor as _img2tensor, tensor2img, imwrite
import numpy as np
import cv2
import time
import os

from checkpoints.BenDeblur.NAFNet_width64 import nafnet_config as nafnet_config

# Ensure the BASE_PROJECT is correctly added to the path for imports
# sys.path.append(f"{PARAMETER.BASE_PROJECT}/ben_deblur/ImageDeBlur")

class Config:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                value = Config(value)
            setattr(self, key, value)
        self._config_dict = config_dict  # Store the original dictionary

    def __getitem__(self, item):
        return self._config_dict[item]

def imread(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def img2tensor(img, bgr2rgb=False, float32=True):
    img = img.astype(np.float32) / 255.
    return _img2tensor(img, bgr2rgb=bgr2rgb, float32=float32)

def process_images(model, images_tensor):
    model.feed_data(data={'lq': images_tensor})
    model.test()
    visuals = model.get_current_visuals()
    out_imgs = []
    for out_tensor in visuals['result']:
        out_img = tensor2img(out_tensor)
        final_img = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
        out_imgs.append(final_img)
    return out_imgs

def process_video(model, video_path, output_folder, batch_size, progress_callback=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return [], "Error: Could not open video."

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    base_filename = os.path.basename(video_path)
    output_filename = os.path.splitext(base_filename)[0] + "_deblurred.mp4"
    output_path = os.path.join(output_folder, output_filename)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    batch_frames = []
    processed_frames = 0
    start_time = time.time()
    total_proc_time = 0
    all_processed_frames = []  # List to store all processed frames

    print("Started processing video")
    while True:
        torch.cuda.empty_cache()
        ret, frame = cap.read()
        if not ret:
            break

        frame_tensor = torch.unsqueeze(img2tensor(frame), 0)
        batch_frames.append(frame_tensor)

        if len(batch_frames) == batch_size:
            frame_proc_start = time.time()
            batch_tensor = torch.cat(batch_frames, dim=0)
            processed_images = process_images(model, batch_tensor)
            batch_proc_time = time.time() - frame_proc_start
            total_proc_time += batch_proc_time

            del batch_tensor
            for img in processed_images:
                out.write(img)
                all_processed_frames.append(img)
            processed_frames += len(batch_frames)
            if progress_callback:
                progress_callback(processed_frames, total_frames)

            average_fps = processed_frames / total_proc_time if total_proc_time > 0 else float('inf')
            elapsed_time = time.time() - start_time
            estimated_total_time = elapsed_time / processed_frames * total_frames
            eta = estimated_total_time - elapsed_time
            progress = processed_frames / total_frames * 100
            sys.stdout.write(f"\rProgress: {progress:.2f}% | ETA: {eta:.2f}s | Average FPS: {average_fps:.2f}")
            sys.stdout.flush()
            batch_frames = []

    if batch_frames:
        frame_proc_start = time.time()
        batch_tensor = torch.cat(batch_frames, dim=0)
        processed_images = process_images(model, batch_tensor)
        batch_proc_time = time.time() - frame_proc_start
        total_proc_time += batch_proc_time

        for img in processed_images:
            out.write(img)
            all_processed_frames.append(img)
        processed_frames += len(batch_frames)
        if progress_callback:
            progress_callback(processed_frames, total_frames)

    average_fps = processed_frames / total_proc_time if total_proc_time > 0 else float('inf')

    cap.release()
    out.release()
    print(f"\nVideo processing complete. Final Average Processing FPS: {average_fps:.2f}")

    return all_processed_frames, output_path

def img2tensor2(img, bgr2rgb=False, float32=True):
    img = img.astype(np.float32) / 255.
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # (C, H, W) and add batch dimension

def process_images2(model, images_tensor):
    model.feed_data(data={'lq': images_tensor})
    model.test()
    visuals = model.get_current_visuals()
    out_imgs = []
    for out_tensor in visuals['result']:
        out_img = tensor2img(out_tensor)
        final_img = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
        out_imgs.append(final_img)
    return out_imgs

def process_list_of_frames(model, frame_list, batch_size, progress_callback=None):
    total_frames = len(frame_list)
    batch_frames = []
    processed_frames = 0
    start_time = time.time()
    total_proc_time = 0
    all_processed_frames = []  # List to store all processed frames

    print("Started processing frames")
    for frame in frame_list:
        torch.cuda.empty_cache()

        frame_tensor = img2tensor2(frame)
        batch_frames.append(frame_tensor)

        if len(batch_frames) == batch_size:
            frame_proc_start = time.time()
            batch_tensor = torch.cat(batch_frames, dim=0)
            processed_images = process_images2(model, batch_tensor)
            batch_proc_time = time.time() - frame_proc_start
            total_proc_time += batch_proc_time

            del batch_tensor
            all_processed_frames.extend(processed_images)
            processed_frames += len(batch_frames)
            if progress_callback:
                progress_callback(processed_frames, total_frames)

            average_fps = processed_frames / total_proc_time if total_proc_time > 0 else float('inf')
            elapsed_time = time.time() - start_time
            estimated_total_time = elapsed_time / processed_frames * total_frames
            eta = estimated_total_time - elapsed_time
            progress = processed_frames / total_frames * 100
            sys.stdout.write(f"\rProgress: {progress:.2f}% | ETA: {eta:.2f}s | Average FPS: {average_fps:.2f}")
            sys.stdout.flush()
            batch_frames = []

    if batch_frames:
        frame_proc_start = time.time()
        batch_tensor = torch.cat(batch_frames, dim=0)
        processed_images = process_images(model, batch_tensor)
        batch_proc_time = time.time() - frame_proc_start
        total_proc_time += batch_proc_time

        all_processed_frames.extend(processed_images)
        processed_frames += len(batch_frames)
        if progress_callback:
            progress_callback(processed_frames, total_frames)

    average_fps = processed_frames / total_proc_time if total_proc_time > 0 else float('inf')

    print(f"\nFrame processing complete. Final Average Processing FPS: {average_fps:.2f}")

    return all_processed_frames

def main_nafnet_list_of_frames(frame_list):
    opt = nafnet_config
    opt['dist'] = False
    model = create_model(Config(opt))

    return process_list_of_frames(model=model,
                                  frame_list=frame_list,
                                  batch_size=1)


def main_nafnet_list_of_frames_dict(input_dict: dict):
    frame_list = input_dict[PARAMETER.AlignClassDictInput.frames]
    return main_nafnet_list_of_frames(frame_list)


def main_nafnet_deblur(video_path, output_folder):
    opt = nafnet_config
    opt['dist'] = False
    model = create_model(Config(opt))

    return process_video(model=model,
                         video_path=video_path,
                         output_folder=output_folder,
                         batch_size=1)

if __name__ == '__main__':
    # opt = nafnet_config
    # opt['dist'] = False
    # model = create_model(Config(opt))
    #
    # process_video(model=model,
    #               video_path=r"C:\Users\orior\PycharmProjects\VideoImageEnhancement\data\images\videos\Car_Going_Down\scene_0_resized_compressed.mp4",
    #               output_folder=r"C:\Users\orior\PycharmProjects\VideoImageEnhancement",
    #               batch_size=8)




    # Example list of image paths
    image_paths = [
        r'C:\Users\orior\PycharmProjects\VideoImageEnhancement\data\images\Shaback\litle_amount_of_images_resized\00057774.png',
        r'C:\Users\orior\PycharmProjects\VideoImageEnhancement\data\images\Shaback\litle_amount_of_images_resized\00057775.png',

        # Add more image paths as needed
    ]
    image_list = [cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) for img_path in
                  image_paths]  # Read and convert images
    frames = image_list

    main_nafnet_list_of_frames(frames)