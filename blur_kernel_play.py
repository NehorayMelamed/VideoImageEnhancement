import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# Load the blur kernel from the provided image
kernel_path = '/home/nehoray/PycharmProjects/VideoImageEnhancement/Yoav_denoise_new/yoav_blur_kernel/output4/avg_kernel.png'
kernel_image = cv2.imread(kernel_path, cv2.IMREAD_GRAYSCALE)

# Check if the image was loaded properly
if kernel_image is None:
    raise FileNotFoundError(f"Unable to load image at path: {kernel_path}")

# Ensure the kernel is normalized and background is black
kernel_image = kernel_image / 255.0
kernel_image[kernel_image < 0.1] = 0  # Set background to black


# Function to apply rotation and scaling to the kernel
def transform_kernel(kernel, angle, scale):
    size = kernel.shape[0]
    center = (size / 2, size / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    transformed_kernel = cv2.warpAffine(kernel, rotation_matrix, (size, size), borderMode=cv2.BORDER_CONSTANT,
                                        borderValue=0)

    # Create a black background and place the transformed kernel on it
    black_background = np.zeros_like(kernel)
    kernel_h, kernel_w = transformed_kernel.shape
    bg_h, bg_w = black_background.shape

    start_x = (bg_w - kernel_w) // 2
    start_y = (bg_h - kernel_h) // 2

    black_background[start_y:start_y + kernel_h, start_x:start_x + kernel_w] = transformed_kernel

    return black_background


# Initial parameters
initial_angle = 0
initial_scale = 1.0

# Create initial transformed kernel
transformed_kernel = transform_kernel(kernel_image, initial_angle, initial_scale)

# Create figure and axis
fig, ax = plt.subplots(facecolor='black')
plt.subplots_adjust(left=0.1, bottom=0.35)
ax.set_facecolor('black')

# Hide all axes, grid lines, and ticks
ax.axis('off')

# Display the initial kernel
kernel_display = ax.imshow(transformed_kernel, cmap='gray', vmin=0, vmax=1)

# Create angle slider
ax_angle = plt.axes([0.1, 0.2, 0.65, 0.03], facecolor='black')
angle_slider = Slider(ax_angle, 'Angle', 0, 180, valinit=initial_angle, color='white')

# Create scale slider
ax_scale = plt.axes([0.1, 0.1, 0.65, 0.03], facecolor='black')
scale_slider = Slider(ax_scale, 'Scale', 0.1, 2.0, valinit=initial_scale, color='white')

# Create save button
ax_save = plt.axes([0.8, 0.025, 0.1, 0.04])
save_button = Button(ax_save, 'Save')


# Update function to redraw the kernel based on slider values
def update(val):
    angle = angle_slider.val
    scale = scale_slider.val
    new_kernel = transform_kernel(kernel_image, angle, scale)
    kernel_display.set_data(new_kernel)
    fig.canvas.draw_idle()


angle_slider.on_changed(update)
scale_slider.on_changed(update)


# Save function to save the current kernel
def save(event):
    angle = angle_slider.val
    scale = scale_slider.val
    new_kernel = transform_kernel(kernel_image, angle, scale)
    cv2.imwrite('saved_kernel.jpg', new_kernel * 255)


save_button.on_clicked(save)

plt.show()
