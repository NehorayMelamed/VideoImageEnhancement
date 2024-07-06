import cv2
import numpy as np

def select_and_crop_image(image_path, save_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found.")
        return

    # Initialize list for storing cropping points
    cropping_points = []

    # Mouse callback function to get cropping points
    def mouse_crop(event, x, y, flags, param):
        # Record starting point on left button click
        if event == cv2.EVENT_LBUTTONDOWN:
            cropping_points.append((x, y))

        # Record ending point on left button release and crop the image
        elif event == cv2.EVENT_LBUTTONUP:
            cropping_points.append((x, y))
            cv2.rectangle(image, cropping_points[0], cropping_points[1], (0, 255, 0), 2)
            cv2.imshow("image", image)

    # Display the image and set the mouse callback function
    cv2.imshow("image", image)
    cv2.setMouseCallback("image", mouse_crop)

    print("Select the crop area and press 'c' to crop and save the image.")

    # Wait until 'c' key is pressed
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            break

    # Ensure two points have been selected
    if len(cropping_points) == 2:
        # Get the coordinates of the crop area
        x1, y1 = cropping_points[0]
        x2, y2 = cropping_points[1]

        # Ensure coordinates are within the image dimensions
        x1, x2 = sorted([max(0, x1), min(image.shape[1], x2)])
        y1, y2 = sorted([max(0, y1), min(image.shape[0], y2)])

        print(f"Cropping coordinates: ({x1}, {y1}), ({x2}, {y2})")
        print(f"Image dimensions: {image.shape}")

        # Crop the image
        cropped_image = image[y1:y2, x1:x2]

        if cropped_image.size == 0:
            print("Error: Cropped image is empty. Check the selected area.")
        else:
            # Save the cropped image
            cv2.imwrite(save_path, cropped_image)
            print(f"Cropped image saved as {save_path}")
    else:
        print("Error: Two points were not selected for cropping.")

    # Close all OpenCV windows
    cv2.destroyAllWindows()



select_and_crop_image("/home/nehoray/PycharmProjects/VideoImageEnhancement/data/dgx_data/blur_cars/white_car/00057780.png",
                      "/home/nehoray/PycharmProjects/VideoImageEnhancement/data/data_croped_for_deblur_or_kernel_blur/00057780.png")

