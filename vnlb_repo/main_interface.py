import vnlb
import numpy as np
from Utils import list_to_numpy
from util.video_to_numpy_array import get_video_frames


def vnlb_denoise(video_list, noise_std=50):
    """
    Denoises a video represented as a list of NumPy arrays.

    Parameters:
    - video_list: list of NumPy arrays, where each array represents a frame of the video.
    - noise_std: standard deviation of the noise used during the denoising process.

    Returns:
    - denoised_list: list of NumPy arrays, denoised frames.
    """
    if noise_std > 50:
        noise_std =50
    if noise_std < 0:
        noise_std = 0

    # Convert list of frames to a single NumPy array with shape (nframes, height, width, channels)
    video_np = list_to_numpy(video_list)
    video_np = np.transpose(video_np, (0, 3, 1, 2))  # Convert to (nframes, channels, height, width)

    # Ensure the video is in float32 format
    video_np = video_np.astype(np.float32) / 255.0  # Normalize to [0, 1]

    # Apply VNLB denoising
    denoised_tensor, _, _ = vnlb.denoise(video_np, noise_std)

    # Move the tensor to the CPU and convert to a NumPy array
    denoised_np = denoised_tensor.cpu().numpy()

    # Convert back to (nframes, height, width, channels) and then to a list of NumPy arrays
    denoised_np = np.transpose(denoised_np, (0, 2, 3, 1))
    denoised_np = (denoised_np * 255.0).clip(0, 255).astype(np.uint8)  # Convert back to uint8
    denoised_list = [frame for frame in denoised_np]

    return denoised_list


def add_noise(video_list, std):
    """
    Adds Gaussian noise to a video represented as a list of NumPy arrays.

    Parameters:
    - video_list: list of NumPy arrays, where each array represents a frame of the video.
    - std: standard deviation of the Gaussian noise.

    Returns:
    - noisy_list: list of NumPy arrays, noisy frames.
    """
    noisy_list = [np.random.normal(frame, scale=std).astype(np.float32) for frame in video_list]
    return noisy_list


def main():
    # Load the video
    video_path = r"C:\Users\orior\PycharmProjects\VideoImageEnhancement\data\images\videos\Car_Going_Down\scene_0_resized_short_compressed.mp4"
    video_frames = get_video_frames(video_path)
    video_frames = video_frames[:4]  # Load and take the first 4 frames

    # Denoise the video without adding noise
    noise_std = 50.0
    denoised_video = vnlb_denoise(video_frames, noise_std)

    # Print the results
    print("Original video frames (as numpy arrays):")
    print(video_frames)

    print("\nDenoised video frames (as numpy arrays):")
    print(denoised_video)


if __name__ == "__main__":
    main()
