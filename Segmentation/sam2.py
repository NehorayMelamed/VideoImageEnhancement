import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import SAM

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

def interactive_bbox_drawing(image_path):
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
    return [x_min, y_min, x_max, y_max]

def get_mask_from_bbox_ultralytics(image_path, model_path, bbox, display_mask=False):
    """
    Get the mask from a bounding box using the Ultralytics SAM model.

    Parameters:
    - image_path (str): Path to the input image.
    - model_path (str): Path to the model file.
    - bbox (list): Bounding box coordinates [x_min, y_min, x_max, y_max].
    - display_mask (bool): If True, displays the image with the mask overlayed.

    Returns:
    - mask (numpy.ndarray): The mask corresponding to the bounding box.
    """

    # Load the model
    model = SAM(model_path)

    # Run inference with the bounding box prompt
    result = model(image_path, bboxes=[bbox])

    # Extract the mask from the result
    if isinstance(result, list) and len(result) > 0 and hasattr(result[0], 'masks'):
        mask = result[0].masks.data[0].cpu().numpy()
    else:
        raise ValueError("Unexpected result format or missing masks in the result.")

    if display_mask:
        display_image_with_mask(image_path, mask)

    return mask

def display_image_with_mask(image_path, mask):
    """
    Display the image with the mask overlayed using matplotlib.

    Parameters:
    - image_path (str): Path to the input image.
    - mask (numpy.ndarray): The mask to be overlayed.
    """
    image = cv2.imread(image_path)
    mask_image = image.copy()
    mask = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask_image, contours, -1, (0, 255, 0), 2)

    # Convert to RGB for matplotlib
    mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB)

    plt.figure()
    plt.imshow(mask_image)
    plt.title("Image with Mask")
    plt.axis('off')
    plt.show()

def display_mask_bw(mask):
    """
    Display the mask in black and white using matplotlib.

    Parameters:
    - mask (numpy.ndarray): The mask to be displayed.
    """
    mask_bw = mask.astype(np.uint8) * 255  # Convert mask to binary (0 or 255)
    plt.figure()
    plt.imshow(mask_bw, cmap='gray')
    plt.title("Mask in Black and White")
    plt.axis('off')
    plt.show()


# Example usage
image_path = 'C:/Users/dudyk/PycharmProjects/NehorayWorkSpace/Shaback/models/img.png'
bbox = [50, 50, 150, 150]  # Example bounding box coordinates
model_path = 'C:/Users/dudyk/PycharmProjects/NehorayWorkSpace/Shaback/models/sam_l.pt'
bbox = interactive_bbox_drawing(image_path)
mask = get_mask_from_bbox_ultralytics(image_path, model_path, bbox, display_mask=True)
display_mask_bw(mask)
print(mask)
