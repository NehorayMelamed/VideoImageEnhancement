import os.path

BASE_PROJECT = os.path.dirname(os.path.abspath(__file__))
RDND_BASE_PATH = os.path.join(BASE_PROJECT, "RDND_proper")
GROUNDED_SEGMENT_ANYTHING_BASE_PATH = os.path.join(BASE_PROJECT, "Grounded_Segment_Anything")
OUTPUT = "tweets_per_accounts"
CAR_LICENSE_PLATE_RECOGNITION = "CAR_LICENSE_PLATE_RECOGNITION"




CHECKPOINTS = os.path.join(BASE_PROJECT, "checkpoints")



### denoise

# path_restore_ckpt_denoise_flow_former = os.path.join(RDND_BASE_PATH, "models", "FlowFormer", "check_points", "sintel.pth")
path_restore_ckpt_denoise_flow_former = f"{CHECKPOINTS}/FlowFormer/check_points/sintel.pth"
# path_checkpoint_latest_things = os.path.join(RDND_BASE_PATH, "models", "irr", "checkpoints", "checkpoint_latest_things.ckpt")
path_checkpoint_latest_things = f"{CHECKPOINTS}/irr/checkpoints/checkpoint_latest_things.ckpt"

### deblur

RVRT_deblur_shoval_train_py_blur20_TEST1_Step60000 = os.path.join(BASE_PROJECT, "Omer", "to_neo", "RVRT_deblur_shoval_train.py_blur20_TEST1_Step60000.tar")


### Directory path to save result for the detedtion process

DETECTION_DIRECTORY_PATH = os.path.join(BASE_PROJECT, OUTPUT, CAR_LICENSE_PLATE_RECOGNITION)


#### NAFNet-width64.yml
NEFNet_width64 = os.path.join(CHECKPOINTS, "BenDeblur", "NAFNet-width64.yml")


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
grounding_dino_config_SwinT_OGC = os.path.join(CHECKPOINTS, "config", "GroundingDINO_SwinT_OGC.py")