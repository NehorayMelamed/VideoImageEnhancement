import os.path

import numpy
import numpy as np

BASE_PROJECT = os.path.dirname(os.path.abspath(__file__))
RDND_BASE_PATH = os.path.join(BASE_PROJECT, "RDND_proper")
GROUNDED_SEGMENT_ANYTHING_BASE_PATH = os.path.join(BASE_PROJECT, "Grounded_Segment_Anything")
OUTPUT = "tweets_per_accounts"
CAR_LICENSE_PLATE_RECOGNITION = "CAR_LICENSE_PLATE_RECOGNITION"

cwd = os.getcwd()
BASE_NAME_RESTORE = os.path.dirname(os.path.dirname(cwd))  # Get the parent of the parent directory

CHECKPOINTS = os.path.join(BASE_PROJECT, "checkpoints")

DEVICE = 'cuda'


### denoise

# path_restore_ckpt_denoise_flow_former = os.path.join(RDND_BASE_PATH, "models", "FlowFormer", "check_points", "sintel.pth")
path_restore_ckpt_denoise_flow_former = f"{CHECKPOINTS}/FlowFormer/check_points/sintel.pth"
# path_checkpoint_latest_things = os.path.join(RDND_BASE_PATH, "models", "irr", "checkpoints", "checkpoint_latest_things.ckpt")
path_checkpoint_latest_things = f"{CHECKPOINTS}/irr/checkpoints/checkpoint_latest_things.ckpt"

### deblur

RVRT_deblur_shoval_train_py_blur20_TEST1_Step60000 = os.path.join(BASE_PROJECT, "Omer", "to_neo",
                                                                  "RVRT_deblur_shoval_train.py_blur20_TEST1_Step60000.tar")

### Directory path to save result for the detedtion process

DETECTION_DIRECTORY_PATH = os.path.join(BASE_PROJECT, OUTPUT, CAR_LICENSE_PLATE_RECOGNITION)

#### NAFNet-ben
base_dir_ben_code = os.path.join(BASE_PROJECT, "ben_deblur", "ImageDeBlur")

### checkpoints
Base_dir_NAFNet = os.path.join(CHECKPOINTS, "BenDeblur")
NEFNet_width64 = os.path.join(Base_dir_NAFNet, "NAFNet_width64.py")
NEFNet_model_path = os.path.join(Base_dir_NAFNet, "models","NAFNet-REDS-width64.pth")

#### Model_New_Checkpoints
Model_New_Checkpoints = os.path.join(BASE_PROJECT, "Model_New_Checkpoints")

#### Blur_kernel_NubKe_model
Blur_kernel_NubKe_model = os.path.join(CHECKPOINTS, "NubKe", "TwoHeads.pkl")

### Depth map

DEPTH_ESTIMATION_CHECKPOINT_PATH = os.path.join(BASE_PROJECT, "Model_New_Checkpoints", )

#### SAM_CHECKPOINTS
SAM_CHECKPOINTS = os.path.join(CHECKPOINTS, "sam", "sam_vit_h_4b8939.pth")

#### yolo world
yolo_world_checkpoint = os.path.join(CHECKPOINTS, "yolo_world", "yolov8x-worldv2.pt")

#### grounding_diano
grounding_dino_checkpoint = os.path.join(CHECKPOINTS, "grounding_dino", "groundingdino_swint_ogc.pth")
grounding_dino_config_SwinT_OGC = os.path.join(CHECKPOINTS, "grounding_dino", "config", "GroundingDINO_SwinT_OGC.py")


ImageRestoration_base_dir = os.path.join(BASE_PROJECT, "ImageRestoration")
FMA_Net_model_path_D = os.path.join(CHECKPOINTS, "FMA_Net", "model_D_best.pt")
FMA_Net_model_path_R = os.path.join(CHECKPOINTS, "FMA_Net", "model_R_best.pt")

#C\LakDNet\
LakDNet_base_checkpoint = os.path.join(CHECKPOINTS, "LakDNet")

#C:\Users\orior\PycharmProjects\VideoImageEnhancement\checkpoints\UFPDeblur\train_on_GoPro\net_g_latest.pth

UFPD_train_go_pro = os.path.join(CHECKPOINTS, "UFPDeblur", "train_on_GoPro", "net_g_latest.pth")



### SAM2
SAM_PROJECT = os.path.join(BASE_PROJECT, "Segmentation", "SAM2")
#models
SAN_2_MODEL_HIERA_BASE_PLUS = os.path.join(SAM_PROJECT, "checkpoints", "sam2_hiera_base_plus.pt")
SAN_2_MODEL_HIERA_LARGE = os.path.join(SAM_PROJECT, "checkpoints", "sam2_hiera_large.pt")
#configs
SAN_2_CONFIG_HIERA_LARGE = os.path.join(SAM_PROJECT, "sam2_configs", "sam2_hiera_l.yaml")


class InputMethodForServices:
    segmentation    = "segmentation"
    BB              = "BB_XYWH"
    polygon         = "polygon"


class DictInput:
    frames: list[np.ndarray] = "frames" # List of frames (numpy arrays) to align.
    reference_frame: np.ndarray = "reference_frame" # (np.ndarray, optional): Reference frame to align against.
    input_method: InputMethodForServices = "input_method"
    user_input: dict = "user_input"
    params: dict = "params"

# class DictOutput:
#     frames: list[np.ndarray] = "frames"
