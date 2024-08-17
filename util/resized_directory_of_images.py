import cv2
import os

def resize_or_crop_images(input_dir, output_dir, resize_dims=None):
    """
    Resize or crop all images in the input directory and save to the output directory.

    Parameters:
    - input_dir: Path to the input directory containing images.
    - output_dir: Path to the output directory to save resized or cropped images.
    - resize_dims: Tuple specifying the dimensions to resize images (width, height).
                   If None, the function will use ROI selection to crop images.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print("No images found in the specified directory.")
        return

    first_image_path = os.path.join(input_dir, image_files[0])
    first_image = cv2.imread(first_image_path)

    if resize_dims:
        # Resize all images
        for img_file in image_files:
            img_path = os.path.join(input_dir, img_file)
            img = cv2.imread(img_path)
            resized_img = cv2.resize(img, resize_dims)
            output_path = os.path.join(output_dir, img_file)
            cv2.imwrite(output_path, resized_img)
            print(f"Resized and saved: {output_path}")
    else:
        # Crop using ROI selected on the first image
        r = cv2.selectROI("Select ROI", first_image, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select ROI")

        x, y, w, h = r
        for img_file in image_files:
            img_path = os.path.join(input_dir, img_file)
            img = cv2.imread(img_path)
            cropped_img = img[y:y+h, x:x+w]
            output_path = os.path.join(output_dir, img_file)
            cv2.imwrite(output_path, cropped_img)
            print(f"Cropped and saved: {output_path}")

# Usage example
if __name__ == '__main__':
    # resize_or_crop_images("/home/nehoray/PycharmProjects/VideoImageEnhancement/data/dgx_data/blur_cars/litle_amount_of_images",
    #                       "/home/nehoray/PycharmProjects/VideoImageEnhancement/data/dgx_data/blur_cars/litle_amount_of_images_resized",
    #                       resize_dims=(800, 600))

    resize_or_crop_images(r"C:\Users\orior\PycharmProjects\VideoImageEnhancement\data\images\Shaback\litle_amount_of_images_resized",
                          r"C:\Users\orior\PycharmProjects\VideoImageEnhancement\data\images\Shaback\litle_amount_of_images_resized_2",
                          resize_dims=(224,224))
