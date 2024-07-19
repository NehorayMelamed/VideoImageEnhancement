import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import sys
import json

# from pathlib import Path
# IMPORT_PATH = Path('/home/simteam-j/Desktop/RDND_proper').parent
# sys.path.append('/home/simteam-j/Desktop/RDND_proper')
# sys.path.append(str(IMPORT_PATH))
from RapidBase.import_all import *
# Opening JSON file
f = open('/home/eylon/Downloads/check/IRX_0027/annotations/instances_default.json')
 
# returns JSON object as 
# a dictionary
a = json.load(f)

bla =0 
# directory1 = "/media/simteam-j/Datasets1/Object_Deteciton_datasets/BGS/p16/BGS/BGS_001/"
# directory1 = "/home/simteam-j/Desktop/RDND_proper/yolov5/runs/detect/test13"
# directory1 = "/media/simteam-j/Datasets1/Object_Deteciton_datasets/crooped_shoval_vids/DJI_video_1/6_bgs/"
directory1 = "/media/eylon/Datasets/Object_Deteciton_datasets/test/numpy_images_with_BB"
# directory1 = "/home/simteam-j/datasets/pal_numpy/train/images/"
list_of_files1 = os.listdir(directory1)
list_of_files1.sort()
# directory2 = "/home/simteam-j/bin_vids"
# list_of_files2 = os.listdir(directory2)


image_list = []
for i in range(3):
# for  i in range(1):
    # random = 1
    # image = np.load("/home/simteam-j/datasets/bgs_new/train/images/000025.npy")
    image = np.load(f"{directory1}/{list_of_files1[i]}", allow_pickle=True).squeeze()
    # image2 = np.load(f"{directory2}/{list_of_files2[random]}").squeeze()
    # image2 = np.load("/home/simteam-j/0.npy")
    # seprator = np.ones((640, 10)) * 255
    # full_image= np.hstack((image, seprator, image2))
    # plt.imshow(image2)
    # image = np.asarray(image[1000: image.shape[0]- 1000, ])
    # print_info(image)
    # print(np.argmax(image))
    # print(np.argmin(image))
    # print_info(image[300:, 4100:4400])
    # image_list = np.concatenate(image_list, image)
    image_list.append(image)
    # plt.figure()
    # plt.imshow(image)
    # path_to_save = "/home/simteam-j/0.npy"
    # np.save(path_to_save, image[:, 3780:4420])
    # plt.imshow(image[:, 3780:4420])
    # plt.show()
    # video = np.zeros((len(list_of_files1),1 ,image.shape[0], 700))
    # video[i] = image[:, 3500:4200]
    print(i)
# video = np.asarray(image_list)
# video = numpy_to_torch(video)
# video = video.permute(0,2,3,1)
image_list = np.asarray(image_list)
image_list = torch.tensor(image_list)
image_list = image_list.unsqueeze(1)
save_path = os.path.join(directory1, "movie.pt")
torch.save(image_list, save_path)
# image_list = torch.load(save_path)
h, w = int((image_list[0].shape[1]/ 2) - 100), int((image_list[0].shape[2]/ 2) - 100)
imshow_torch_video(image_list[:,:, h: h + 200, w:] , FPS=10, frame_stride=1)

# # imshow_torch_video(video, colormap="Greys")
# imshow_torch_video(video, FPS=10, frame_stride=1)
