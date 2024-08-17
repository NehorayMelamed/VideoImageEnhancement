import cv2
import numpy


def get_video_frames(video_path) -> list[numpy.ndarray]:
    """
    Extracts frames from a video file.

    Parameters:
    video_path (str): The path to the video file.

    Returns:
    list: A list of frames extracted from the video.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []

    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return frames

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames


if __name__ == '__main__':
    video_frames = get_video_frames("path_to_your_video_file.mp4")
    print(f"Extracted {len(video_frames)} frames from the video.")
