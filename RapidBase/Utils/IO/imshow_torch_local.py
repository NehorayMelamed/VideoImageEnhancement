import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
import torch
import cv2
import matplotlib.pylab as pylab
from matplotlib.pylab import figure, colorbar, title
import os
import time

def imshow_torch(image, flag_colorbar=True, title_str=''):
    fig = figure()
    # plt.cla()
    plt.clf()
    # plt.close()

    if len(image.shape) == 4:
        pylab.imshow(np.transpose(image[0].detach().cpu().numpy(), (1, 2, 0)).squeeze())  # transpose to turn from [C,H,W] to numpy regular [H,W,C]
    elif len(image.shape) == 3:
        pylab.imshow(np.transpose(image.detach().cpu().numpy(),(1,2,0)).squeeze()) #transpose to turn from [C,H,W] to numpy regular [H,W,C]
    elif len(image.shape) == 2:
        pylab.imshow(image.detach().cpu().numpy())

    if flag_colorbar:
        colorbar()  #TODO: fix the bug of having multiple colorbars when calling this multiple times
    title(title_str)
    return fig

def torch_to_numpy(input_tensor):
    if type(input_tensor) == torch.Tensor:
        (B,T,C,H,W), shape_len, shape_vec = get_full_shape_torch(input_tensor)
        input_tensor = input_tensor.cpu().data.numpy()
        if shape_len == 2:
            #[H,W]
            return input_tensor
        elif shape_len == 3:
            #[C,H,W] -> [H,W,C]
            return np.transpose(input_tensor, [1,2,0])
        elif shape_len == 4:
            #[T,C,H,W] -> [T,H,W,C]
            return np.transpose(input_tensor, [0,2,3,1])
        elif shape_len == 5:
            #[B,T,C,H,W] -> [B,T,H,W,C]
            return np.transpose(input_tensor, [0,1,3,4,2])
    return input_tensor

def get_full_shape_torch(input_tensor):
    if len(input_tensor.shape) == 1:
        W = input_tensor.shape
        H = 1
        C = 1
        T = 1
        B = 1
        shape_len = 1
        shape_vec = (W)
    elif len(input_tensor.shape) == 2:
        H, W = input_tensor.shape
        C = 1
        T = 1
        B = 1
        shape_len = 2
        shape_vec = (H,W)
    elif len(input_tensor.shape) == 3:
        C, H, W = input_tensor.shape
        T = 1
        B = 1
        shape_len = 3
        shape_vec = (C,H,W)
    elif len(input_tensor.shape) == 4:
        T, C, H, W = input_tensor.shape
        B = 1
        shape_len = 4
        shape_vec = (T,C,H,W)
    elif len(input_tensor.shape) == 5:
        B, T, C, H, W = input_tensor.shape
        shape_len = 5
        shape_vec = (B,T,C,H,W)
    shape_vec = np.array(shape_vec)
    return (B,T,C,H,W), shape_len, shape_vec

def imshow_torch_video(input_tensor, number_of_frames=None, FPS=3, flag_BGR2RGB=True, frame_stride=1,
                       flag_colorbar=False, video_title='', video_title_list=None,
                        close_after_next_imshow = False):
    #TODO: fix colorbar
    def get_correct_form(input_tensor, i):
        if shape_len == 4:
            #(T,C,H,W)
            if flag_BGR2RGB and C==3:
                output_tensor = cv2.cvtColor(torch_to_numpy(input_tensor[i]), cv2.COLOR_BGR2RGB)
            else:
                output_tensor = torch_to_numpy(input_tensor[i])
        elif shape_len == 3:
            #(T,H,W)
            output_tensor = torch_to_numpy(input_tensor[i])
        elif shape_len == 5:
            #(B,T,C   ,H,W)
            if flag_BGR2RGB and C==3:
                output_tensor = cv2.cvtColor(torch_to_numpy(input_tensor[0, i]), cv2.COLOR_BGR2RGB)
            else:
                output_tensor = torch_to_numpy(input_tensor[0, i])
        return output_tensor

    ### Get Parameters: ###
    (B, T, C, H, W), shape_len, shape_vec = get_full_shape_torch(input_tensor)
    if number_of_frames is not None:
        number_of_frames_to_show = min(number_of_frames, T)
    else:
        number_of_frames_to_show = T

    cf = figure()
    # plt.ion() #TODO: what's this?
    output_tensor = get_correct_form(input_tensor, 0)
    im = plt.imshow(output_tensor)
    if flag_colorbar:
        cbar = plt.colorbar(im)
    mtime = os.path.getmtime(temp_path)
    for i in np.arange(0,number_of_frames_to_show,frame_stride):
        output_tensor = get_correct_form(input_tensor, i)
        im.set_array(output_tensor)
        plt.show(block=False)
        if video_title_list is not None:
            current_title = str(video_title_list[i])
        else:
            current_title = video_title
        plt.title(current_title + '  ' + str(i))
        plt.pause(1/FPS)
        new_mtime = os.path.getmtime(temp_path)
        if new_mtime - mtime > 0:
            mtime = new_mtime
            plt.close()
            return
        if flag_colorbar:
            cbar = plt.colorbar(im)
            plt.draw()
    mtime = os.path.getmtime(temp_path)
    while 1:
        plt.pause(1)  # <-------
        new_mtime = os.path.getmtime(temp_path)
        if new_mtime - mtime > 0:
            if close_after_next_imshow:
                plt.close()
            return



input_path = 'C:/Users/temp/a.pt'
text_path = 'C:/Users/temp/b.txt'
temp_path = 'C:/Users/temp/temp.txt'

def imshow_torch_seamless():

    img = torch.load(input_path)
    with open(text_path) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    flag_colorbar = (lines[0]=='True')
    close_after_next_imshow = (lines[2]=='True')
    imshow_torch(img, flag_colorbar, lines[1])
    plt.show(block=False)
    mtime = os.path.getmtime(temp_path)
    while 1:
        # print(2)
        plt.pause(1)  # <-------
        new_mtime = os.path.getmtime(temp_path)
        if new_mtime - mtime > 0:
            if close_after_next_imshow:
                plt.close()
            return

def imshow_torch_video_seamless():

    video = torch.load(input_path)
    with open(text_path) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    number_of_frames = int(lines[0])
    FPS = int(lines[1])
    flag_BGR2RGB = (lines[2] == True)
    frame_stride = int(lines[3])
    close_after_next_imshow = (lines[4] == 'True')
    imshow_torch_video(video, number_of_frames, FPS, flag_BGR2RGB, frame_stride,
                       close_after_next_imshow = close_after_next_imshow)


if __name__ == '__main__':
    mtime = 0

    while 1:
        new_mtime = os.path.getmtime(temp_path)
        if new_mtime - mtime > 0:
            with open(text_path) as f:
                lines = f.readlines()
                lines = [line.rstrip() for line in lines]
            mtime = new_mtime
            if lines[-1] == 'image':
                imshow_torch_seamless()
            else:
                imshow_torch_video_seamless()

