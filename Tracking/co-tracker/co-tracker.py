import torch
import imageio.v3 as iio
import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector


def generate_grid_points(xmin, ymin, xmax, ymax, grid_size, fill_all=False):
    if fill_all:
        x_points = np.arange(xmin, xmax + 1)
        y_points = np.arange(ymin, ymax + 1)
    else:
        x_points = np.linspace(xmin, xmax, grid_size)
        y_points = np.linspace(ymin, ymax, grid_size)

    points = np.array([[x, y] for x in x_points for y in y_points])
    return points


def onselect(eclick, erelease):
    global box_coords
    xmin, ymin, xmax, ymax = eclick.xdata, eclick.ydata, erelease.xdata, erelease.ydata
    box_coords.append((xmin, ymin, xmax, ymax))


def select_boxes_on_frame(frame):
    global box_coords
    box_coords = []

    fig, ax = plt.subplots()
    ax.imshow(frame)
    ax.set_title("Draw boxes around the areas to track and close the window when done")

    rect_selector = RectangleSelector(ax, onselect, interactive=True)
    plt.show()

    return box_coords


def load_video(video_source):
    if isinstance(video_source, str):
        # Assuming the source is a file path or URL
        frames = iio.imread(video_source, plugin="FFMPEG")  # plugin="pyav"
    elif isinstance(video_source, torch.Tensor):
        frames = video_source.cpu().numpy()  # Convert tensor to numpy
    elif isinstance(video_source, np.ndarray):
        frames = video_source
    else:
        raise ValueError("Unsupported video source type. Provide a file path, URL, tensor, or numpy array.")

    return frames


def track_points_in_video(video_source, points=None, grid_size=10, interactive=False, fill_all=False):
    frames = load_video(video_source)

    if interactive:
        # Allow the user to select multiple boxes on the first frame
        first_frame = frames[0]
        box_coords = select_boxes_on_frame(first_frame)
        points = []
        for (xmin, ymin, xmax, ymax) in box_coords:
            points.extend(generate_grid_points(xmin, ymin, xmax, ymax, grid_size, fill_all))

    device = 'cuda'
    video = torch.tensor(frames).permute(0, 3, 1, 2)[None].float().to(device)  # B T C H W

    # Convert points to tensor and move to device
    frame_number = 0  # Use the first frame for queries
    queries = torch.tensor([[frame_number, x, y] for x, y in points], device=device).float()

    # Run Offline CoTracker:
    cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker2").to(device)

    # Perform tracking
    pred_tracks, pred_visibility = cotracker(video, queries=queries[None])

    # Convert tensors to numpy for returning
    pred_tracks = pred_tracks[0].cpu().detach().numpy()  # T N 2
    pred_visibility = pred_visibility[0].cpu().detach().numpy()  # T N

    return pred_tracks, pred_visibility, frames


def plot_tracking_results(pred_tracks, pred_visibility, frames):
    num_frames = pred_tracks.shape[0]
    num_points = pred_tracks.shape[1]

    fig, ax = plt.subplots()

    def update(frame_num):
        ax.clear()
        frame = frames[frame_num]
        ax.imshow(frame)

        for point in range(num_points):
            track = pred_tracks[:frame_num + 1, point, :]  # Up to the current frame
            visibility = pred_visibility[:frame_num + 1, point]  # Up to the current frame

            for i in range(len(track) - 1):
                if visibility[i] > 0.5 and visibility[i + 1] > 0.5:  # Visible points
                    ax.plot(track[i:i + 2, 0], track[i:i + 2, 1], 'g-', linewidth=1)
                else:  # Non-visible points
                    ax.plot(track[i:i + 2, 0], track[i:i + 2, 1], 'r-', linewidth=1)

        ax.set_title("CoTracker Predictions")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

    ani = animation.FuncAnimation(fig, update, frames=num_frames, repeat=False)
    plt.show()


if __name__ == "__main__":
    video_source = r"C:\Users\dudyk\PycharmProjects\NehorayWorkSpace\Shaback\scene_10.mp4"
    pred_tracks, pred_visibility, frames = track_points_in_video(video_source, interactive=True, grid_size=5,
                                                                 fill_all=False)
    plot_tracking_results(pred_tracks, pred_visibility, frames)
