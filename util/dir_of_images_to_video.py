import cv2
import os
import numpy as np

def images_to_video_and_frames(image_dir, output_video_path, fps=25):
    """
    Convert a directory of images to a video and return a list of frames as numpy arrays.

    Args:
        image_dir (str): Path to the directory containing images.
        output_video_path (str): Path to save the output video.
        fps (int): Frames per second for the output video.

    Returns:
        List[np.ndarray]: List of frames as numpy arrays.
    """
    # Get list of image files in the directory
    image_files = sorted([f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f)) and f.endswith(('.png', '.jpg', '.jpeg'))])

    if not image_files:
        raise ValueError("No images found in the directory.")

    # Read the first image to get the frame size
    first_image_path = os.path.join(image_dir, image_files[0])
    first_image = cv2.imread(first_image_path)
    if first_image is None:
        raise ValueError(f"Failed to read the first image: {first_image_path}")

    height, width, layers = first_image.shape

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for mp4
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    if not video_writer.isOpened():
        raise ValueError("Failed to initialize the video writer.")

    frames = []

    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Warning: Skipping image {image_path} because it could not be read.")
            continue

        # Append frame to the list
        frames.append(frame)

        # Write frame to the video
        video_writer.write(frame)

    # Release the video writer
    video_writer.release()

    return frames

if __name__ == '__main__':
    # Example usage
    image_directory = '/home/nehoray/PycharmProjects/VideoImageEnhancement/data/directory_video_images/video_blur_directory_1/sub_video_2'
    output_video = '/home/nehoray/PycharmProjects/VideoImageEnhancement/data/videos/omer_deblur/video_deblur_2.mp4'
    try:
        frames = images_to_video_and_frames(image_directory, output_video)
        print(f"Video saved successfully as {output_video}")
    except Exception as e:
        print(f"Error: {e}")
