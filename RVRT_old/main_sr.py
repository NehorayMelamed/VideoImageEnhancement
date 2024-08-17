import os
import glob
import torch
import yaml
from PIL import Image
from collections import OrderedDict
from utils import utils_image as util


def load_yaml(yaml_path):
    with open(yaml_path, 'r') as file:
        return yaml.safe_load(file)


def load_model(model_path, config, device):
    from models.network_rvrt import RVRT
    model = RVRT(**config)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def process_super_resolution(input_dir, output_dir, model_path, config_path, device='cuda'):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load configuration
    config = load_yaml(config_path)

    # Load the model
    model = load_model(model_path, config, device)

    # Process each image in the input directory
    for img_path in glob.glob(os.path.join(input_dir, '*')):
        img_name = os.path.basename(img_path)
        output_path = os.path.join(output_dir, img_name)

        # Load and preprocess image
        img = util.imread_uint(img_path, n_channels=3)
        img = util.uint2tensor4(img).to(device)

        # Perform super-resolution
        with torch.no_grad():
            sr_img = model(img)

        # Post-process and save the output image
        sr_img = util.tensor2uint(sr_img)
        util.imsave(sr_img, output_path)

    print(f"Super-resolution processing complete. Output saved in {output_dir}")


# Example usage
input_dir = 'path_to_input_images'
output_dir = 'path_to_output_images'
model_path = 'path_to_pretrained_model'
config_path = 'path_to_config_yaml'
process_super_resolution(input_dir, output_dir, model_path, config_path)
