import cv2

def get_video_fps(video_path):
    """
    Returns the FPS (frames per second) of a video file.

    Args:
        video_path (str): Path to the video file.

    Returns:
        float: FPS of the video.
    """
    # Open the video file
    video_capture = cv2.VideoCapture(video_path)

    # Check if the video was opened successfully
    if not video_capture.isOpened():
        raise ValueError(f"Error opening video file: {video_path}")

    # Get the FPS of the video
    fps = video_capture.get(cv2.CAP_PROP_FPS)

    # Release the video capture object
    video_capture.release()

    return int(fps)

# Example usage
# video_path = 'path/to/your/video.mp4'
# fps = get_video_fps(video_path)
# print(f"FPS of the video: {fps}")
