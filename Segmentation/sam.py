import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
from Segmentation.segment_anything.segment_anything import sam_model_registry, SamPredictor


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

def get_mask_from_bbox(image_path, checkpoint_path, bbox=None, interactive=False, model_type='vit_h', display_mask=False):
    """
    Get the mask from a bounding box using the Segment Anything model.

    Parameters:
    - image_path (str): Path to the input image.
    - checkpoint_path (str): Path to the model checkpoint.
    - bbox (tuple, optional): Bounding box coordinates (x_min, y_min, x_max, y_max). Required if interactive is False.
    - interactive (bool): If True, allows user to draw the bounding box. Default is False.
    - model_type (str): Type of the model. Can be 'vit_h', 'vit_l', or 'vit_b'.
    - display_mask (bool): If True, displays the image with the mask overlayed.

    Returns:
    - mask (numpy.ndarray): The mask corresponding to the bounding box.
    """

    def draw_bbox(event, x, y, flags, param):
        global start_point, end_point, drawing, current_bbox
        if event == cv2.EVENT_LBUTTONDOWN:
            start_point = (x, y)
            drawing = True
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                end_point = (x, y)
                current_bbox = (start_point, end_point)
        elif event == cv2.EVENT_LBUTTONUP:
            end_point = (x, y)
            drawing = False
            current_bbox = (start_point, end_point)

    if interactive:
        global start_point, end_point, drawing, current_bbox
        drawing = False
        start_point = ()
        end_point = ()
        current_bbox = None

        # Load the image
        image = cv2.imread(image_path)
        clone = image.copy()

        cv2.namedWindow("Image")
        cv2.setMouseCallback("Image", draw_bbox)

        while True:
            display_image = clone.copy()
            if start_point and end_point:
                cv2.rectangle(display_image, start_point, end_point, (0, 255, 0), 2)
            cv2.imshow("Image", display_image)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("r"):
                start_point, end_point = (), ()
                current_bbox = None
                clone = image.copy()
            elif key == ord("c"):
                break

        cv2.destroyAllWindows()

        if current_bbox is None:
            raise ValueError("Bounding box not drawn correctly.")

        x_max, y_min = current_bbox[0]
        x_min, y_max = current_bbox[1]
        bbox = (x_min, y_min, x_max, y_max)

    elif bbox is None:
        raise ValueError("Bounding box must be provided if interactive mode is not used.")

    if isinstance(image_path, str):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif isinstance(image_path, np.ndarray):
        if image_path.shape[2] == 3:
            image = cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB)
        else:
            image = image_path
    else:
        raise ValueError("Invalid image input. Must be a file path or numpy array.")

    # Load the model
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"SAM - Using device: {device}")
    sam.to(device=device)

    predictor = SamPredictor(sam)
    predictor.set_image(image)

    # Predict the mask for the bounding box
    x_min, y_min, x_max, y_max = bbox
    input_box = np.array([x_min, y_min, x_max, y_max])
    masks, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box[None, :],
        multimask_output=False,
    )

    mask = masks[0].astype(bool)

    # plt.figure(figsize=(10, 10))
    # plt.imshow(image)
    # show_mask(masks[0], plt.gca())
    # show_box(input_box, plt.gca())
    # plt.axis('off')
    # plt.show()

    if display_mask:
        # Overlay the mask on the image
        mask_image = image.copy()
        mask = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(mask_image, contours, -1, (0, 255, 0), 2)
        mask_image = cv2.cvtColor(mask_image, cv2.COLOR_RGB2BGR)
        cv2.imshow("Image with Mask", mask_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return mask

def display_mask_bw(mask):
    """
    Display the mask in black and white.

    Parameters:
    - mask (numpy.ndarray): The mask to be displayed.
    """
    mask_bw = mask.astype(np.uint8) * 255  # Convert mask to binary (0 or 255)
    cv2.imshow("Mask in Black and White", mask_bw)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = r'C:\Users\dudyk\PycharmProjects\NehorayWorkSpace\Shaback\models\img.png'
    bbox = (425, 600, 700, 875)  # Example bounding box coordinates
    checkpoint_path = r'C:/Users/dudyk/PycharmProjects/NehorayWorkSpace/Shaback/models/sam_vit_h_4b8939.pth'

    mask = get_mask_from_bbox(image_path, checkpoint_path, interactive=True)
    display_mask_bw(mask)
    print(mask)