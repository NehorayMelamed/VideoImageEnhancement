import math
import os.path
import sys

import scipy.io
import torch.linalg

sys.path.append("/home/yoav/rdnd")
sys.path.append("/home/yoav/rdnd/Speckels")
sys.path.append("/home/yoav")
sys.path.append("/home/yoav/Anvil")

import scipy.io.wavfile as wavfile
# import tensorly as tl
# tl.set_backend('pytorch')
# import librosa
from RapidBase.Anvil.import_all import *
from RapidBase.Anvil.optical_flow_objects import *
from RapidBase.Anvil._transforms.shift_matrix_subpixel import _shift_matrix_subpixel_fft_batch, _shift_matrix_subpixel_fft_batch_with_channels
from math import pi, sqrt, exp
# import pywt
# from cross_correlation_stuff import *
# from torchaudio.transforms import Spectrogram
from scipy.optimize import curve_fit
from skimage.restoration import unwrap_phase
from typing import List

# from Speckles.cross_correlation_stuff import *

def weight_function_uniform(i, j, k):
    return 1


def weight_function_parabolic_proximity(i, j, k):
    return 1 / ((i-k)**2 + (j-k)**2)


def weight_function_delta(i, j, k):
    # basically ignores the majority vote and only using the normal cross correlation calculation
    return 1 if (k == i or k == j) else 0



def create_1d_gauss_weights(n=11,sigma=1):
    r = range(-n//2, n//2)
    gauss_weights_list = [1 / (sigma * sqrt(2*pi)) * exp(-float(x)**2/(2*sigma**2)) for x in r]
    gauss_weights_torch = torch.tensor(gauss_weights_list)
    return gauss_weights_torch / gauss_weights_torch.sum()  # sum is not needed theoretically but here for numerical reasons(sum is not exactly 1)


def create_speckles_sequence(final_shape =(100,1,500,500),
                             speckle_size_in_pixels=10,
                             polarization=0,
                             speckle_contrast=0.8,
                             max_decorrelation=0.02,
                             max_beam_wonder=0.05,
                             max_tilt_wonder=0.05):
    # #FOR NOW ONLY ACCEPTS H=W!!!!!
    # # ### TODO: delete, for testing: ###
    # input_tensor = read_image_default_torch()
    # input_tensor = crop_torch_batch(input_tensor, (500, 500))
    # output_tensor = read_video_default_torch()
    # output_tensor = RGB2BW(output_tensor)
    # output_tensor = crop_tensor(output_tensor, (720, 720))
    # final_shape = output_tensor.shape
    # # final_shape = (100,1,500,500)
    # T, C, H, W = final_shape
    # speckle_size_in_pixels = 50
    # polarization = 0.
    # max_decorrelation = 0.0
    # max_beam_wonder = 0.0
    # max_tilt_wonder = 0.0001  #TODO: maybe present in pixels?
    # speckle_contrast = 1

    surface_phase_factor = 1

    ### Get Parameters: ###
    T, C, H, W = final_shape
    N = H

    # Calculations:
    wf = (N / speckle_size_in_pixels)

    # Create 2D frequency space of size NxN
    x = torch.arange(-N / 2, N / 2, 1)
    [X, Y] = torch.meshgrid(x, x)
    # Assign random values to the frequencies
    beam = torch.exp(- ((X / 2) ** 2 + (Y / 2) ** 2) / wf ** 2)
    # beam = beam / torch.sqrt(sum(sum(abs(beam) ** 2)))
    beam_initial = beam / torch.sqrt((beam.abs() ** 2).sum())

    # Polarization:
    # # Get Surfaces:
    # surface_1 = torch.exp(2 * torch.pi * 1j * 1 * torch.randn(1, C, H, W))
    # surface_2 = torch.exp(2 * torch.pi * 1j * 1 * torch.rand(1, C, H, W))
    # surface_3 = torch.exp(2 * torch.pi * 1j * 1 * torch.rand(1, C, H, W))
    # surface_4 = torch.exp(2 * torch.pi * 1j * 1 * torch.rand(1, C, H, W))
    # Get Surface Phase:
    surface_1_phase = (2 * torch.pi * 1j * 1 * torch.rand(1, C, H, W))
    surface_2_phase = (2 * torch.pi * 1j * 1 * torch.rand(1, C, H, W))
    surface_3_phase = (2 * torch.pi * 1j * 1 * torch.rand(1, C, H, W))
    surface_4_phase = (2 * torch.pi * 1j * 1 * torch.rand(1, C, H, W))
    surface_1 = torch.exp(surface_1_phase)
    surface_2 = torch.exp(surface_2_phase)
    surface_3 = torch.exp(surface_3_phase)
    surface_4 = torch.exp(surface_4_phase)
    surface_1 = surface_1 - surface_1.mean()
    surface_2 = surface_2 - surface_2.mean()
    surface_3 = surface_3 - surface_3.mean()
    surface_4 = surface_4 - surface_4.mean()
    decorrelation_percent = 0

    ### Initialize Things For Tilt Phase: ###
    # Get tilt phases k-space:
    x = np.arange(-np.fix(W / 2), np.ceil(W / 2), 1)
    y = np.arange(-np.fix(H / 2), np.ceil(H / 2), 1)
    delta_f1 = 1 / W
    delta_f2 = 1 / H
    f_x = x * delta_f1
    f_y = y * delta_f2
    # Use fftshift on the 1D vectors for effeciency sake to not do fftshift on the final 2D array:
    f_x = np.fft.fftshift(f_x)
    f_y = np.fft.fftshift(f_y)
    # Build k-space meshgrid:
    [kx, ky] = np.meshgrid(f_x, f_y)
    # get displacement matrix:
    kx = torch.tensor(kx).unsqueeze(0).unsqueeze(0)
    ky = torch.tensor(ky).unsqueeze(0).unsqueeze(0)

    ### Initialize Output Speckles List: ###
    output_frames_list = []
    output_frames_torch = torch.zeros(T, C, H, W)

    ### Get Initial Speckles: ###
    # Get randomly two beams representing wave's polarization
    beam_one = beam_initial * surface_1
    beam_two = beam_initial * surface_2
    # Calculate speckle pattern (practically fft)
    speckle_pattern1 = (torch.fft.fft2(torch.fft.fftshift(beam_one)))
    speckle_pattern2 = (torch.fft.fft2(torch.fft.fftshift(beam_two)))
    # Calculate weighted average of the two beams
    speckle_pattern_total_intensity = (1 - polarization) * speckle_pattern1.abs() ** 2 + polarization * speckle_pattern2.abs() ** 2
    output_frames_torch[0:1, :, :, :] = speckle_pattern_total_intensity
    # output_tensor[0:1, :, :, :] = output_tensor[0:1] * ((1 - speckle_contrast) + speckle_pattern_total_intensity / surface_phase_factor * speckle_contrast)

    ### Get Beam-Wonder & Beam-Tilt Numbers: ###
    # TODO: maybe use get random noise with frequency content
    # TODO: maybe use cumsum twice to get more low frequencies?
    beam_wonder_in_pixels_H = get_random_number_in_range(-max_beam_wonder, max_beam_wonder, array_size=T)
    beam_wonder_in_pixels_W = get_random_number_in_range(-max_beam_wonder, max_beam_wonder, array_size=T)
    beam_wonder_in_pixels_H = np.cumsum(beam_wonder_in_pixels_H)
    beam_wonder_in_pixels_W = np.cumsum(beam_wonder_in_pixels_W)
    # plot(beam_wonder_in_pixels_H); plt.show()
    # plot(beam_wonder_in_pixels_W); plt.show()
    beam_tilt_in_pixels_H = get_random_number_in_range(-max_tilt_wonder, max_tilt_wonder, array_size=T)
    beam_tilt_in_pixels_W = get_random_number_in_range(-max_tilt_wonder, max_tilt_wonder, array_size=T)
    beam_tilt_in_pixels_H = np.cumsum(beam_tilt_in_pixels_H)
    beam_tilt_in_pixels_W = np.cumsum(beam_tilt_in_pixels_W)
    # beam_tilt_in_pixels_H = np.cumsum(beam_tilt_in_pixels_H)
    # beam_tilt_in_pixels_W = np.cumsum(beam_tilt_in_pixels_W)
    # plot(beam_tilt_in_pixels_H); plt.show()
    # plot(beam_tilt_in_pixels_W); plt.show()
    decorrelation_percent = get_random_number_in_range(0, max_decorrelation, T)
    decorrelation_percent = np.cumsum(decorrelation_percent)
    # plot(decorrelation_percent); plt.show()

    ### Get Surface Phase: ###
    surface_1_phase_list = []
    surface_2_phase_list = []
    surface_1_phase_list.append(surface_1_phase)
    surface_2_phase_list.append(surface_2_phase)
    surface_1_phase_list.append(surface_3_phase)
    surface_2_phase_list.append(surface_4_phase)
    max_number_of_total_decorrelations = np.ceil(decorrelation_percent.max())
    for i in np.arange(1, max_number_of_total_decorrelations):
        surface_1_phase_new = (2 * torch.pi * 1j * 1 * torch.randn(1, C, H, W))
        surface_2_phase_new = (2 * torch.pi * 1j * 1 * torch.randn(1, C, H, W))
        surface_1_phase_list.append(surface_1_phase_new)
        surface_2_phase_list.append(surface_2_phase_new)

    ### Loop Over Frames: ###
    decorrelation_counter = 0
    for frame_index in np.arange(1, T):
        # print(frame_index)
        ### Change The Surface To Create Boiling/Decorrelation: ###
        decorrelation_percent_current = decorrelation_percent[frame_index]
        if decorrelation_percent_current > 1:
            print(str(frame_index) + ', DECORRELATION: ' + str(decorrelation_percent_current))
            # ### Switch Surfaces: ###
            # surface_1 = 0 + copy.deepcopy(surface_3)
            # surface_2 = 0 + copy.deepcopy(surface_4)
            # surface_3 = torch.exp(2 * torch.pi * 1j * 1 * torch.rand(1, C, H, W))
            # surface_4 = torch.exp(2 * torch.pi * 1j * 1 * torch.rand(1, C, H, W))
            # ### Switch Surface Phases: ###
            # surface_1_phase = copy.deepcopy(surface_3_phase)
            # surface_2_phase = copy.deepcopy(surface_4_phase)
            # surface_3_phase = (2 * torch.pi * 1j * 10 * torch.rand(1, C, H, W))
            # surface_4_phase = (2 * torch.pi * 1j * 10 * torch.rand(1, C, H, W))

            ### Switch Surface Phases Using Lists: ###
            decorrelation_counter += 1
            decorrelation_percent = decorrelation_percent - 1
            decorrelation_percent_current = decorrelation_percent_current - 1

            # surface_1_phase = surface_1_phase_list[decorrelation_counter]
            # surface_2_phase = surface_2_phase_list[decorrelation_counter]
            # surface_3_phase = surface_1_phase_list[decorrelation_counter + 1]
            # surface_4_phase = surface_2_phase_list[decorrelation_counter + 1]
            # surface_1_total_phase = surface_1_phase * (1 - decorrelation_percent_current) + surface_3_phase * (decorrelation_percent_current)
            # surface_2_total_phase = surface_2_phase * (1 - decorrelation_percent_current) + surface_4_phase * (decorrelation_percent_current)

            # imshow_torch((surface_1_total_phase_previous - surface_1_total_phase).abs())
            # imshow_torch((surface_1_total_phase_previous).abs())
            # (surface_1_phase - surface_1_phase_previous).abs().max()
            # (surface_3_phase - surface_1_phase_previous).abs().max()
            # (surface_3_phase - surface_3_phase_previous).abs().max()
            # (surface_3_phase_previous - surface_1_phase).abs().max()
            # (surface_4_phase_previous - surface_2_phase).abs().max()


        # print(decorrelation_percent_current)
        # ### Add Surfaces: ###
        # surface_1_total = surface_1 * (1 - decorrelation_percent_current) + surface_3 * (decorrelation_percent_current)
        # surface_2_total = surface_2 * (1 - decorrelation_percent_current) + surface_4 * (decorrelation_percent_current)
        ### Add Phases: ###
        surface_1_phase = surface_1_phase_list[decorrelation_counter]
        surface_2_phase = surface_2_phase_list[decorrelation_counter]
        surface_3_phase = surface_1_phase_list[decorrelation_counter + 1]
        surface_4_phase = surface_2_phase_list[decorrelation_counter + 1]
        surface_1_total_phase = surface_1_phase * (1 - decorrelation_percent_current) + surface_3_phase * (decorrelation_percent_current)
        surface_2_total_phase = surface_2_phase * (1 - decorrelation_percent_current) + surface_4_phase * (decorrelation_percent_current)
        # surface_1_total_phase = surface_1_total_phase - surface_1_total_phase.mean()
        # surface_2_total_phase = surface_2_total_phase - surface_1_total_phase.mean()
        surface_1_total = torch.exp(surface_1_total_phase)
        surface_2_total = torch.exp(surface_2_total_phase)
        surface_1_total = surface_1_total - surface_1_total.mean()
        surface_2_total = surface_2_total - surface_2_total.mean()

        ### Change Beam Position (Beam Wonder): ###
        beam_wonder_in_pixels_H_current = wf * beam_wonder_in_pixels_H[frame_index]
        beam_wonder_in_pixels_W_current = wf * beam_wonder_in_pixels_W[frame_index]
        beam = shift_matrix_subpixel_torch(beam_initial, torch.tensor(beam_wonder_in_pixels_W_current), torch.tensor(beam_wonder_in_pixels_H_current))

        ### Change Beam Tilt: ###
        shift_W_tilt = W * beam_tilt_in_pixels_H[frame_index]
        shift_H_tilt = H * beam_tilt_in_pixels_W[frame_index]
        tilt_phase = torch.exp(-(1j * 2 * torch.pi * ky * shift_H_tilt + 1j * 2 * torch.pi * kx * shift_W_tilt))

        ### Get New Speckles: ###
        # Get randomly two beams representing wave's polarization
        beam_one = beam * surface_1_total * tilt_phase
        beam_two = beam * surface_2_total * tilt_phase
        # Calculate speckle pattern (practically fft)
        # speckle_pattern1 = torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(beam_one)))
        # speckle_pattern2 = torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(beam_two)))
        speckle_pattern1 = (torch.fft.fft2(torch.fft.fftshift(beam_one)))
        speckle_pattern2 = (torch.fft.fft2(torch.fft.fftshift(beam_two)))
        # Calculate weighted average of the two beams
        speckle_pattern_total_intensity = (1 - polarization) * speckle_pattern1.abs() ** 2 + polarization * speckle_pattern2.abs() ** 2
        output_frames_torch[frame_index:frame_index + 1, :, :, :] = speckle_pattern_total_intensity

    ### Correct for speckle contrast: ###
    output_frames_torch = (1 - speckle_contrast) + output_frames_torch * speckle_contrast

    imshow_torch_video(output_frames_torch, FPS=25, frame_stride=1)

    return output_frames_torch

def create_training_sample_for_fourier_estimation(speckle_size_in_pixels=10, N=512, polarization=0, flag_gauss_circ=1, flag_normalize=1, flag_imshow=0):
    intensity, amplitude, amplitude2, total_beam = create_speckles_of_certain_size_in_pixels(speckle_size_in_pixels, N, polarization, flag_gauss_circ, flag_normalize, flag_imshow)
    amplitude_torch = torch.tensor(amplitude)
    max_shift = 0.1
    shift_x = torch.tensor(np.random.rand(1) / max_shift)
    shift_y = torch.tensor(np.random.rand(1) / max_shift)
    # shift_y = torch.tensor(np.array(0))
    total_beam = torch.tensor(total_beam)
    shift_layer = Shift_Layer_Torch()

    speckle_pattern_amplitude_shifted, _, _, fft_image_displaced, fft_image = shift_layer.forward(amplitude_torch, -shift_x, -shift_y)
    speckle_pattern_intensity_shifted = speckle_pattern_amplitude_shifted.abs() ** 2
    fft_image = torch_fftshift(fft_image, -2)
    fft_image_displaced = torch_fftshift(fft_image_displaced, -2)

    # displacement_matrix = fft_image_displaced / (fft_image + 1e-6)
    # displacement_matrix_angle = displacement_matrix.angle()
    # displacement_matrix_angle_cropped = crop_tensor(displacement_matrix_angle, (N // 2, N // 2))
    # dispalcement_matrix_angle_cropped_numpy = displacement_matrix_angle_cropped.cpu().numpy()
    # displacement_matrix_angle_cropped_numpy_unwrapped = unwrap_phase(dispalcement_matrix_angle_cropped_numpy)
    # #
    # imshow_torch(speckle_pattern_intensity_shifted, title_str="speckle pattern")
    # imshow_torch(fft_image.abs(), title_str="fft image")
    # imshow_torch(fft_image_displaced.abs(), title_str="displaced fft image")
    # imshow_torch(total_beam.abs(), title_str="total beam")
    # phase_image = fft_image_displaced / (fft_image + 1e-6)
    # imshow_torch(phase_image.abs(), title_str="phase of displaced fft")
    # imshow_torch(displacement_matrix_angle, title_str="phase of displacement matrix")
    # plt.imshow(dispalcement_matrix_angle_cropped_numpy); plt.title("cropped phase"); plt.show()
    # plt.imshow(displacement_matrix_angle_cropped_numpy_unwrapped); plt.title("unwraped cropped phase"); plt.show()
    # #
    # ## plot regular fft estimation of the INTENSITY(not amplitude)
    # imshow_torch(torch_fftshift(torch_fft2(speckle_pattern_intensity_shifted)).abs())

    # todo: extreact phase from unwrapped phase image and compare it to gt shift somehow
    return torch.tensor(speckle_pattern_intensity_shifted, dtype=torch.float32).unsqueeze(0), \
           torch.tensor(fft_image_displaced.abs(), dtype=torch.float32).unsqueeze(0) / 255


# todo: parallelize this function
def create_speckles_of_certain_size_in_pixels(speckle_size_in_pixels=10, N=512, polarization=0, flag_gauss_circ=1, flag_normalize=1, flag_imshow=0):
    #    import numpy
    #    from numpy import arange
    #    from numpy.random import *

    # #Parameters:
    # speckle_size_in_pixels = 10
    # N = 512
    # polarization = 0
    # flag_gauss_circ = 0

    # Calculations:
    assert False, "Use batched implementation"
    wf = (N / speckle_size_in_pixels)

    if flag_gauss_circ == 1:
        x = np.arange(-N / 2, N / 2, 1)
        distance_between_the_two_beams_x = 0
        distance_between_the_two_beams_y = 0
        [X, Y] = np.meshgrid(x, x)
        beam_one = np.exp(- ((X - distance_between_the_two_beams_x / 2) ** 2 + (Y - distance_between_the_two_beams_y / 2) ** 2) / wf ** 2)
        beam_two = np.exp(- ((X + distance_between_the_two_beams_x / 2) ** 2 + (Y + distance_between_the_two_beams_y / 2) ** 2) / wf ** 2)
        total_beam = beam_one + beam_two
        total_beam = total_beam / np.sqrt(sum(sum(abs(total_beam) ** 2)))
    else:
        x = np.arange(-N / 2, N / 2, 1) * 1
        y = x
        [X, Y] = np.meshgrid(x, y)
        c = 0
        distance_between_the_two_beams_y = 0
        beam_one = ((X - distance_between_the_two_beams_y / 2) ** 2 + (Y - distance_between_the_two_beams_y / 2) ** 2 <= wf ** 2)
        beam_two = ((X + distance_between_the_two_beams_y / 2) ** 2 + (Y + distance_between_the_two_beams_y / 2) ** 2 <= wf ** 2)
        total_beam = beam_one + beam_two

    # Polarization:
    if (polarization > 0 & polarization < 1):
        beam_one = total_beam * np.exp(2 * np.pi * 1j * 10 * np.random.randn(N, N))
        beam_two = total_beam * np.exp(2 * np.pi * 1j * 10 * np.random.randn(N, N))
        speckle_pattern1 = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(beam_one)))
        speckle_pattern2 = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(beam_two)))
        speckle_pattern_total_intensity = (1 - polarization) * abs(speckle_pattern1) ** 2 + polarization * abs(speckle_pattern2) ** 2
    else:
        total_beam = total_beam * np.exp(2 * np.pi * 1j * (10 * np.random.randn(N, N)))
        speckle_pattern1 = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(total_beam)))
        speckle_pattern2 = np.empty_like(speckle_pattern1)
        speckle_pattern_total_intensity = np.abs(speckle_pattern1) ** 2

    # if flag_normalize == 1: bla = bla-bla.min() bla=bla/bla.max()
    # if flag_imshow == 1: imshow(speckle_pattern_total_intensity) colorbar()

    return speckle_pattern_total_intensity, speckle_pattern1, speckle_pattern2, total_beam


# from RapidBase_dudy.Utils.IO.tic_toc import *
# todo: parallelize this function
def create_speckles_of_certain_size_in_pixels_torch(speckle_size_in_pixels=10, B=32, N_pixels=512, polarization=0, flag_gauss_circ=1, flag_normalize=1, flag_imshow=0, decorr_percentage=0, device=0):
    #    import numpy
    #    from numpy import arange
    #    from numpy.random import *

    # #Parameters:
    # speckle_size_in_pixels = 10
    # N = 512
    # polarization = 0
    # flag_gauss_circ = 0
    # Calculations:
    wf = (N_pixels / speckle_size_in_pixels)
    if flag_gauss_circ == 1:
        x = np.arange(-N_pixels / 2, N_pixels / 2, 1)
        distance_between_the_two_beams_x = 0
        distance_between_the_two_beams_y = 0
        [X, Y] = np.meshgrid(x, x)
        beam_one = np.exp(- ((X - distance_between_the_two_beams_x / 2) ** 2 + (Y - distance_between_the_two_beams_y / 2) ** 2) / wf ** 2)
        beam_two = np.exp(- ((X + distance_between_the_two_beams_x / 2) ** 2 + (Y + distance_between_the_two_beams_y / 2) ** 2) / wf ** 2)
        total_beam = beam_one + beam_two
        total_beam = total_beam / np.sqrt(sum(sum(abs(total_beam) ** 2)))
    else:
        x = np.arange(-N_pixels / 2, N_pixels / 2, 1) * 1
        y = x
        [X, Y] = np.meshgrid(x, y)
        c = 0
        distance_between_the_two_beams_y = 0
        beam_one = ((X - distance_between_the_two_beams_y / 2) ** 2 + (Y - distance_between_the_two_beams_y / 2) ** 2 <= wf ** 2)
        beam_two = ((X + distance_between_the_two_beams_y / 2) ** 2 + (Y + distance_between_the_two_beams_y / 2) ** 2 <= wf ** 2)
        total_beam = beam_one + beam_two

    ### Transfer to torch tensor and repeat by batch size: ###
    total_beam = torch.from_numpy(total_beam).to(device)
    total_beam = total_beam.unsqueeze(0).unsqueeze(0)
    total_beam = total_beam.repeat(B, 1, 1, 1)

    ### Randomize random phase screens: ###
    random_phase_screen1 = torch.exp(2*torch.pi*1j*10*torch.randn_like(total_beam))
    random_phase_screen2 = torch.exp(2*torch.pi*1j*10*torch.randn_like(total_beam))

    ### Perform Multiplicaiton by random phase screens: ###
    beam_one = total_beam * random_phase_screen1
    beam_two = total_beam * random_phase_screen2

    ### Perform FFT: ###
    decorr_factor = wf * (decorr_percentage/100)
    shifted_beam_abs_one, shifted_beam_raw_one = \
        _shift_matrix_subpixel_fft_batch(total_beam.squeeze(), torch.tensor(decorr_factor), torch.tensor(decorr_factor))
    shifted_beam_abs_two, shifted_beam_raw_two = \
        _shift_matrix_subpixel_fft_batch(total_beam.squeeze(), torch.tensor(decorr_factor), torch.tensor(decorr_factor))


    speckle_pattern1 = torch_fftshift(torch.fft.fftn(torch_fftshift(beam_one), dim=(-2,-1)))
    speckle_pattern1_decorr = torch_fftshift(torch.fft.fftn(torch_fftshift(shifted_beam_raw_one.unsqueeze(1)), dim=(-2,-1)))
    speckle_pattern2 = torch_fftshift(torch.fft.fftn(torch_fftshift(beam_two), dim=(-2, -1)))
    speckle_pattern2_decorr = torch_fftshift(torch.fft.fftn(torch_fftshift(shifted_beam_raw_two.unsqueeze(1)), dim=(-2,-1)))

    # imshow_torch(beam_one.real, title_str="total_beam")
    # imshow_torch(shifted_beam_raw.real.unsqueeze(1), title_str="total_beam_shifted")
    # imshow_torch(speckle_pattern1[0].real, title_str="speckle_pattern1")
    # imshow_torch(speckle_pattern1_decorr[0].real, title_str="speckle_pattern1_decorr")

    ### Get final internsity: ###
    if (polarization > 0 & polarization < 1):
        speckle_pattern_total_intensity = (1 - polarization) * speckle_pattern1_decorr.abs() ** 2 + polarization * speckle_pattern2_decorr.abs() ** 2
    else:
        speckle_pattern_total_intensity = speckle_pattern1_decorr.abs() ** 2

    # imshow_torch(speckle_pattern1.real, title_str="speckle_pattern_1")
    # imshow_torch(speckle_pattern2.real, title_str="speckle_pattern_2")
    # imshow_torch(speckle_pattern1.abs()**2, title_str="speckle_pattern_1_total_intensity")
    # imshow_torch(speckle_pattern2.abs()**2, title_str="speckle_pattern_2_total_intensity")
    # for alpha in [0.3, 0.7, 0.8, 0.9]:
    #     imshow_torch((alpha*speckle_pattern1 + (1-alpha)*speckle_pattern2).abs()**2, title_str=f"decorrelated_total_intensity, {alpha}")
    #
    # if flag_normalize == 1: bla = bla-bla.min() bla=bla/bla.max()
    # if flag_imshow == 1: imshow(speckle_pattern_total_intensity) colorbar()

    return speckle_pattern_total_intensity, speckle_pattern1_decorr, speckle_pattern2_decorr, total_beam


def field_decorr_create_speckles_of_certain_size_in_pixels_torch(speckle_size_in_pixels=10, B=32, N_pixels=512, polarization=0,
                                                    flag_gauss_circ=1, flag_normalize=1, flag_imshow=0,
                                                    decorr_percentage=0, device=0):

    wf = (N_pixels / speckle_size_in_pixels)
    if flag_gauss_circ == 1:
        x = np.arange(-N_pixels / 2, N_pixels / 2, 1)
        distance_between_the_two_beams_x = 0
        distance_between_the_two_beams_y = 0
        [X, Y] = np.meshgrid(x, x)
        beam_one = np.exp(- ((X - distance_between_the_two_beams_x / 2) ** 2 + (
                    Y - distance_between_the_two_beams_y / 2) ** 2) / wf ** 2)
        beam_two = np.exp(- ((X + distance_between_the_two_beams_x / 2) ** 2 + (
                    Y + distance_between_the_two_beams_y / 2) ** 2) / wf ** 2)
        total_beam = beam_one + beam_two
        total_beam = total_beam / np.sqrt(sum(sum(abs(total_beam) ** 2)))
    else:
        x = np.arange(-N_pixels / 2, N_pixels / 2, 1) * 1
        y = x
        [X, Y] = np.meshgrid(x, y)
        c = 0
        distance_between_the_two_beams_y = 0
        beam_one = ((X - distance_between_the_two_beams_y / 2) ** 2 + (
                    Y - distance_between_the_two_beams_y / 2) ** 2 <= wf ** 2)
        beam_two = ((X + distance_between_the_two_beams_y / 2) ** 2 + (
                    Y + distance_between_the_two_beams_y / 2) ** 2 <= wf ** 2)
        total_beam = beam_one + beam_two

    ### Transfer to torch tensor and repeat by batch size: ###
    total_beam = torch.from_numpy(total_beam).to(device)
    total_beam = total_beam.unsqueeze(0).unsqueeze(0)
    total_beam = total_beam.repeat(B, 1, 1, 1)

    ### Randomize random phase screens: ###
    random_phase_screen = torch.exp(2 * torch.pi * 1j * 10 * torch.randn_like(total_beam))

    ### Shift laser beams: ###
    decorr_factor = wf * (decorr_percentage / 100)
    beam_one_shifted_abs, beam_one_shifted = \
        _shift_matrix_subpixel_fft_batch(total_beam.squeeze(), torch.tensor(decorr_factor), torch.tensor(decorr_factor))

    ### Perform Multiplicaiton by random phase screens: ###
    beam_one = total_beam * random_phase_screen
    beam_one_shifted = beam_one_shifted_abs.unsqueeze(1) * random_phase_screen

    ### Create speckle pattern: ###
    speckle_pattern1 = torch_fftshift(torch.fft.fftn(torch_fftshift(beam_one), dim=(-2, -1)))
    speckle_pattern1_decorr = torch_fftshift(torch.fft.fftn(torch_fftshift(beam_one_shifted), dim=(-2, -1)))

    ### Get final internsity: ###
    speckle_pattern_total_intensity = speckle_pattern1.abs() ** 2
    speckle_pattern_total_intensity_decorr = speckle_pattern1_decorr.abs() ** 2

    imshow_torch(speckle_pattern_total_intensity, title_str="speckle_pattern")
    imshow_torch(speckle_pattern_total_intensity_decorr, title_str="speckle_pattern_decorr")

    return speckle_pattern_total_intensity, speckle_pattern_total_intensity_decorr


# field_decorr_create_speckles_of_certain_size_in_pixels_torch(N_pixels=64, decorr_percentage=10, device=1)

def create_shir_decorrelated_speckle_video_torch(speckle_size_in_pixels=10, B=32, N_pixels=512, polarization=0,
                                                 flag_gauss_circ=1, flag_normalize=1, flag_imshow=0,
                                                 decorr_percentage=0, device=0):
    wf = (N_pixels / speckle_size_in_pixels)
    if flag_gauss_circ == 1:
        x = np.arange(-N_pixels / 2, N_pixels / 2, 1)
        distance_between_the_two_beams_x = 0
        distance_between_the_two_beams_y = 0
        [X, Y] = np.meshgrid(x, x)
        beam_one = np.exp(- ((X - distance_between_the_two_beams_x / 2) ** 2 + (
                    Y - distance_between_the_two_beams_y / 2) ** 2) / wf ** 2)
        beam_two = np.exp(- ((X + distance_between_the_two_beams_x / 2) ** 2 + (
                    Y + distance_between_the_two_beams_y / 2) ** 2) / wf ** 2)
        total_beam = beam_one + beam_two
        total_beam = total_beam / np.sqrt(sum(sum(abs(total_beam) ** 2)))
    else:
        x = np.arange(-N_pixels / 2, N_pixels / 2, 1) * 1
        y = x
        [X, Y] = np.meshgrid(x, y)
        c = 0
        distance_between_the_two_beams_y = 0
        beam_one = ((X - distance_between_the_two_beams_y / 2) ** 2 + (
                    Y - distance_between_the_two_beams_y / 2) ** 2 <= wf ** 2)
        beam_two = ((X + distance_between_the_two_beams_y / 2) ** 2 + (
                    Y + distance_between_the_two_beams_y / 2) ** 2 <= wf ** 2)
        total_beam = beam_one + beam_two

    ### Transfer to torch tensor and repeat by batch size: ###
    total_beam = torch.from_numpy(total_beam).to(device)
    total_beam = total_beam.unsqueeze(0).unsqueeze(0)
    total_beam = total_beam.repeat(B, 1, 1, 1)

    ### Randomize random phase screens: ###
    # extract a single random phase screen
    random_phase_screen = torch.exp(2 * torch.pi * 1j * 10 * torch.randn_like(total_beam))[0:1]
    ### Shift laser beams: ###
    # TODO: NOTE, i removed the radius DOF since the speckles came out too similiar.
    shift_max_radius = 2 * wf * (decorr_percentage / 100)
    alpha = torch.rand(B).to(device) * 2 * math.pi
    shift_x = shift_max_radius * torch.cos(alpha)
    shift_y = shift_max_radius * torch.sin(alpha)

    beam_one_shifted_abs, beam_one_shifted = \
        _shift_matrix_subpixel_fft_batch(total_beam.squeeze(), torch.tensor(shift_y), torch.tensor(shift_x))

    ### Perform Multiplicaiton by random phase screens: ###
    shifted_beams = beam_one_shifted_abs.unsqueeze(1) * random_phase_screen

    ### Create speckle pattern: ###
    decorr_patterns = torch_fftshift(torch.fft.fftn(torch_fftshift(shifted_beams), dim=(-2, -1)))

    ### Get final internsity: ###
    decorr_speckle_pattern_total_intensity = decorr_patterns.abs() ** 2

    # imshow_torch(decorr_speckle_pattern_total_intensity[0])
    # imshow_torch(decorr_speckle_pattern_total_intensity[1])
    # imshow_torch(decorr_speckle_pattern_total_intensity[2])
    # imshow_torch(decorr_speckle_pattern_total_intensity[3])
    # imshow_torch(decorr_speckle_pattern_total_intensity[4])
    return decorr_speckle_pattern_total_intensity, shifted_beams


def create_shir_decorrelated_speckle_batch_torch(speckle_size_in_pixels=10, num_frames=2, video_len=32, N_pixels=512, polarization=0,
                                                 flag_gauss_circ=1, flag_normalize=1, flag_imshow=0,
                                                 decorr_percentage=0, device=0):
    """
    the simple function(create_shir_decorrelated_speckle_video_torch) returned the same pattern for the entire video.
    this time we want a batch in which we have 2 degrees of freedom, One is the batch size or video length, and the other is number of frames.
    we want [N, F] speckles such that for [i, :] speckles they are the same decorrelated pattern and each i is a different speckle pattern.

    Returns:

    """
    wf = (N_pixels / speckle_size_in_pixels)
    if flag_gauss_circ == 1:
        x = np.arange(-N_pixels / 2, N_pixels / 2, 1)
        distance_between_the_two_beams_x = 0
        distance_between_the_two_beams_y = 0
        [X, Y] = np.meshgrid(x, x)
        beam_one = np.exp(- ((X - distance_between_the_two_beams_x / 2) ** 2 + (
                    Y - distance_between_the_two_beams_y / 2) ** 2) / wf ** 2)
        beam_two = np.exp(- ((X + distance_between_the_two_beams_x / 2) ** 2 + (
                    Y + distance_between_the_two_beams_y / 2) ** 2) / wf ** 2)
        total_beam = beam_one + beam_two
        total_beam = total_beam / np.sqrt(sum(sum(abs(total_beam) ** 2)))
    else:
        x = np.arange(-N_pixels / 2, N_pixels / 2, 1) * 1
        y = x
        [X, Y] = np.meshgrid(x, y)
        c = 0
        distance_between_the_two_beams_y = 0
        beam_one = ((X - distance_between_the_two_beams_y / 2) ** 2 + (
                    Y - distance_between_the_two_beams_y / 2) ** 2 <= wf ** 2)
        beam_two = ((X + distance_between_the_two_beams_y / 2) ** 2 + (
                    Y + distance_between_the_two_beams_y / 2) ** 2 <= wf ** 2)
        total_beam = beam_one + beam_two

    ### Transfer to torch tensor and repeat by batch size: ###
    total_beam = torch.from_numpy(total_beam).to(device)
    total_beam = total_beam.unsqueeze(0).unsqueeze(0)
    total_beam = total_beam.repeat(video_len, 1, 1, 1)

    ### Randomize random phase screens: ###
    # extract a video_len phase screens for a different speckle pattern
    random_phase_screen = torch.exp(2 * torch.pi * 1j * 10 * torch.randn_like(total_beam))
    ### Shift laser beams: ###
    # TODO: NOTE, i removed the radius DOF since the speckles came out too similiar.
    shift_max_radius = 2 * wf * (decorr_percentage / 100)
    alpha = torch.rand((video_len, num_frames)).to(device) * 2 * math.pi
    shift_x = shift_max_radius.unsqueeze(1) * torch.cos(alpha)
    shift_y = shift_max_radius.unsqueeze(1) * torch.sin(alpha)

    beams = []
    for i in range(num_frames):
        beam_one_shifted_abs, beam_one_shifted = \
            _shift_matrix_subpixel_fft_batch(total_beam.squeeze(), torch.tensor(shift_y[:, i]), torch.tensor(shift_x[:, i]))
        beams.append(beam_one_shifted_abs)
    ### Perform Multiplicaiton by random phase screens: ###
    beams = torch.cat([b.unsqueeze(1) for b in beams], 1)
    shifted_beams = beams * random_phase_screen
    ### Create speckle pattern: ###
    decorr_patterns = torch_fftshift(torch.fft.fftn(torch_fftshift(shifted_beams), dim=(-2, -1)))

    ### Get final internsity: ###
    decorr_speckle_pattern_total_intensity = decorr_patterns.abs() ** 2

    # imshow_torch(decorr_speckle_pattern_total_intensity[1, 0])
    # imshow_torch(decorr_speckle_pattern_total_intensity[0, 0])
    # imshow_torch(decorr_speckle_pattern_total_intensity[2])
    # imshow_torch(decorr_speckle_pattern_total_intensity[3])
    # imshow_torch(decorr_speckle_pattern_total_intensity[4])
    return decorr_speckle_pattern_total_intensity, shifted_beams

# create_shir_decorrelated_speckle_video_torch(speckle_size_in_pixels=5, B=100, N_pixels=64, decorr_percentage=20)


def create_poretz_decorrelated_speckle_video_torch(speckle_size_in_pixels=10, B=32, N_pixels=512, polarization=0,
                                                    flag_gauss_circ=1, flag_normalize=1, flag_imshow=0,
                                                    decorr_percentage=0, device=0):
    wf = (N_pixels / speckle_size_in_pixels)
    if flag_gauss_circ == 1:
        x = np.arange(-N_pixels / 2, N_pixels / 2, 1)
        distance_between_the_two_beams_x = 0
        distance_between_the_two_beams_y = 0
        [X, Y] = np.meshgrid(x, x)
        beam_one = np.exp(- ((X - distance_between_the_two_beams_x / 2) ** 2 + (
                    Y - distance_between_the_two_beams_y / 2) ** 2) / wf ** 2)
        beam_two = np.exp(- ((X + distance_between_the_two_beams_x / 2) ** 2 + (
                    Y + distance_between_the_two_beams_y / 2) ** 2) / wf ** 2)
        total_beam = beam_one + beam_two
        total_beam = total_beam / np.sqrt(sum(sum(abs(total_beam) ** 2)))
    else:
        x = np.arange(-N_pixels / 2, N_pixels / 2, 1) * 1
        y = x
        [X, Y] = np.meshgrid(x, y)
        c = 0
        distance_between_the_two_beams_y = 0
        beam_one = ((X - distance_between_the_two_beams_y / 2) ** 2 + (
                    Y - distance_between_the_two_beams_y / 2) ** 2 <= wf ** 2)
        beam_two = ((X + distance_between_the_two_beams_y / 2) ** 2 + (
                    Y + distance_between_the_two_beams_y / 2) ** 2 <= wf ** 2)
        total_beam = beam_one + beam_two

    ### Transfer to torch tensor and repeat by batch size: ###
    total_beam = torch.from_numpy(total_beam).to(device)
    total_beam = total_beam.unsqueeze(0).unsqueeze(0)

    ### Randomize random phase screens: ###
    ### Shift laser beams: ###
    shift = wf * (decorr_percentage / 100)
    phase_size_factor = 100
    random_phase_screen = torch.exp(2 * torch.pi * 1j * 10 * torch.randn_like(total_beam.repeat(1, 1, phase_size_factor, 1)))

    speckles = torch.zeros_like(total_beam).repeat(B, 1, 1, 1)
    current_shift = 0
    for i in range(B):
        current_shift += shift
        beam_shift = current_shift % 1
        phase_shift = int(current_shift // 1)

        beam_one_shifted_abs, beam_one_shifted = \
            _shift_matrix_subpixel_fft_batch(total_beam.squeeze(), torch.tensor(0), torch.tensor(beam_shift))

        ### Perform Multiplicaiton by random phase screens: ###
        shifted_beam = beam_one_shifted_abs.unsqueeze(1) * random_phase_screen[:, :, phase_shift:phase_shift+total_beam.shape[-1], :]

        ### Create speckle pattern: ###
        decorr_pattern = torch_fftshift(torch.fft.fftn(torch_fftshift(shifted_beam), dim=(-2, -1)))

        ### Get final internsity: ###
        decorr_speckle_pattern_total_intensity = decorr_pattern.abs() ** 2
        speckles[i] = decorr_speckle_pattern_total_intensity.squeeze(0)

    return speckles


def create_poretz_decorrelated_speckle_batch_torch(speckle_size_in_pixels=10, num_frames=2, video_len=32, N_pixels=512,
                                                   polarization=0,
                                                   flag_gauss_circ=1, flag_normalize=1, flag_imshow=0,
                                                   decorr_percentage=0, device=0):
    wf = (N_pixels / speckle_size_in_pixels)
    if flag_gauss_circ == 1:
        x = np.arange(-N_pixels / 2, N_pixels / 2, 1)
        distance_between_the_two_beams_x = 0
        distance_between_the_two_beams_y = 0
        [X, Y] = np.meshgrid(x, x)
        beam_one = np.exp(- ((X - distance_between_the_two_beams_x / 2) ** 2 + (
                    Y - distance_between_the_two_beams_y / 2) ** 2) / wf ** 2)
        beam_two = np.exp(- ((X + distance_between_the_two_beams_x / 2) ** 2 + (
                    Y + distance_between_the_two_beams_y / 2) ** 2) / wf ** 2)
        total_beam = beam_one + beam_two
        total_beam = total_beam / np.sqrt(sum(sum(abs(total_beam) ** 2)))
    else:
        x = np.arange(-N_pixels / 2, N_pixels / 2, 1) * 1
        y = x
        [X, Y] = np.meshgrid(x, y)
        c = 0
        distance_between_the_two_beams_y = 0
        beam_one = ((X - distance_between_the_two_beams_y / 2) ** 2 + (
                    Y - distance_between_the_two_beams_y / 2) ** 2 <= wf ** 2)
        beam_two = ((X + distance_between_the_two_beams_y / 2) ** 2 + (
                    Y + distance_between_the_two_beams_y / 2) ** 2 <= wf ** 2)
        total_beam = beam_one + beam_two

    ### Transfer to torch tensor and repeat by batch size: ###
    total_beam = torch.from_numpy(total_beam).to(device)
    total_beam = total_beam.unsqueeze(0).unsqueeze(0)
    total_beam = total_beam.repeat(video_len, 1, 1, 1)

    ### Randomize random phase screens: ###
    ### Shift laser beams: ###
    shifts = wf * (decorr_percentage.unsqueeze(1).repeat(1, num_frames) / 100)
    phase_size_factor = 100
    random_phase_screen = torch.exp(2 * torch.pi * 1j * 10 * torch.randn_like(total_beam.repeat(1, 1, phase_size_factor, 1)))

    shifts_cumsum = shifts.cumsum(1)
    beam_shifts = shifts_cumsum % 1
    phase_shifts = (shifts_cumsum // 1).to(torch.int)

    shifted_beams = []
    # todo: fix this using phase_shifts instead of i which is a constant shift of 1
    for i in range(num_frames):
        beam_one_shifted_abs, beam_one_shifted = \
            _shift_matrix_subpixel_fft_batch(total_beam.squeeze(), torch.tensor(0), torch.tensor(beam_shifts[:, i]))
        shifted_beams.append(beam_one_shifted_abs.unsqueeze(1))

    shifted_beams = torch.cat(shifted_beams, 1)
    ### Perform Multiplicaiton by random phase screens: ####
    for i in range(num_frames):
        shifted_beams[:, i] = shifted_beams[:, i] * random_phase_screen[:, 0, i:i + total_beam.shape[-1], :]

    ### Create speckle pattern: ###
    decorr_pattern = torch_fftshift(torch.fft.fftn(torch_fftshift(shifted_beams), dim=(-2, -1)))

    ### Get final internsity: ###
    decorr_speckle_pattern_total_intensity = decorr_pattern.abs() ** 2

    # imshow_torch(decorr_speckle_pattern_total_intensity[1, 0])
    # imshow_torch(decorr_speckle_pattern_total_intensity[1, 1])

    return decorr_speckle_pattern_total_intensity


# create_poretz_decorrelated_speckle_video_torch(speckle_size_in_pixels=5, B=100, N_pixels=64, decorr_percentage=20)



def create_decorrelated_speckle_pairs_of_certain_size_in_pixels_torch(speckle_size_in_pixels=10, B=32, N_pixels=512, polarization=0, flag_gauss_circ=1, flag_normalize=1, flag_imshow=0, decorr_percentage=0, device=0, speckle_pattern1=None):

    wf = (N_pixels / speckle_size_in_pixels)
    if flag_gauss_circ == 1:
        x = np.arange(-N_pixels / 2, N_pixels / 2, 1)
        distance_between_the_two_beams_x = 0
        distance_between_the_two_beams_y = 0
        [X, Y] = np.meshgrid(x, x)
        beam_one = np.exp(- ((X - distance_between_the_two_beams_x / 2) ** 2 + (Y - distance_between_the_two_beams_y / 2) ** 2) / wf ** 2)
        beam_two = np.exp(- ((X + distance_between_the_two_beams_x / 2) ** 2 + (Y + distance_between_the_two_beams_y / 2) ** 2) / wf ** 2)
        total_beam = beam_one + beam_two
        total_beam = total_beam / np.sqrt(sum(sum(abs(total_beam) ** 2)))
    else:
        x = np.arange(-N_pixels / 2, N_pixels / 2, 1) * 1
        y = x
        [X, Y] = np.meshgrid(x, y)
        c = 0
        distance_between_the_two_beams_y = 0
        beam_one = ((X - distance_between_the_two_beams_y / 2) ** 2 + (Y - distance_between_the_two_beams_y / 2) ** 2 <= wf ** 2)
        beam_two = ((X + distance_between_the_two_beams_y / 2) ** 2 + (Y + distance_between_the_two_beams_y / 2) ** 2 <= wf ** 2)
        total_beam = beam_one + beam_two

    ### Transfer to torch tensor and repeat by batch size: ###
    total_beam = torch.from_numpy(total_beam).to(device)
    total_beam = total_beam.unsqueeze(0).unsqueeze(0)
    total_beam = total_beam.repeat(B, 1, 1, 1)

    ### Randomize random phase screens: ###
    random_phase_screen1 = torch.exp(2*torch.pi*1j*10*torch.randn_like(total_beam))
    random_phase_screen2 = torch.exp(2*torch.pi*1j*10*torch.randn_like(total_beam))

    ### Perform Multiplicaiton by random phase screens: ###
    beam_one = total_beam * random_phase_screen1
    beam_two = total_beam * random_phase_screen2

    if speckle_pattern1 is None:
        speckle_pattern1 = torch_fftshift(torch.fft.fftn(torch_fftshift(beam_one), dim=(-2,-1)))
    speckle_pattern2 = torch_fftshift(torch.fft.fftn(torch_fftshift(beam_two), dim=(-2, -1)))

    # imshow_torch(beam_one.real, title_str="total_beam")
    # imshow_torch(shifted_beam_raw.real.unsqueeze(1), title_str="total_beam_shifted")
    # imshow_torch(speckle_pattern1[0].real, title_str="speckle_pattern1")
    # imshow_torch(speckle_pattern1_decorr[0].real, title_str="speckle_pattern1_decorr")

    alpha = (1 - decorr_percentage / 100).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    # alpha should be of same shape as batch
    decorr_speckle_pattern_total_intensity = (alpha*speckle_pattern1 + (1-alpha)*speckle_pattern2).abs()**2
    speckle_pattern_total_intensity = (speckle_pattern1).abs()**2

    # imshow_torch(speckle_pattern1.real, title_str="speckle_pattern_1")
    # imshow_torch(speckle_pattern2.real, title_str="speckle_pattern_2")
    # imshow_torch(speckle_pattern1.abs()**2, title_str="speckle_pattern_1_total_intensity")
    # imshow_torch(speckle_pattern2.abs()**2, title_str="speckle_pattern_2_total_intensity")
    # for alpha in [0.3, 0.7, 0.8, 0.9]:
    #     imshow_torch((alpha*speckle_pattern1 + (1-alpha)*speckle_pattern2).abs()**2, title_str=f"decorrelated_total_intensity, {alpha}")
    #
    # if flag_normalize == 1: bla = bla-bla.min() bla=bla/bla.max()
    # if flag_imshow == 1: imshow(speckle_pattern_total_intensity) colorbar()

    return speckle_pattern_total_intensity, decorr_speckle_pattern_total_intensity, speckle_pattern1, speckle_pattern2


def create_decorrelated_speckle_video_torch(speckle_size_in_pixels=10, B=32, N_pixels=512, polarization=0, flag_gauss_circ=1, flag_normalize=1, flag_imshow=0, decorr_percentage=0, device=0):
    wf = (N_pixels / speckle_size_in_pixels)
    if flag_gauss_circ == 1:
        x = np.arange(-N_pixels / 2, N_pixels / 2, 1)
        distance_between_the_two_beams_x = 0
        distance_between_the_two_beams_y = 0
        [X, Y] = np.meshgrid(x, x)
        beam_one = np.exp(- ((X - distance_between_the_two_beams_x / 2) ** 2 + (
                    Y - distance_between_the_two_beams_y / 2) ** 2) / wf ** 2)
        beam_two = np.exp(- ((X + distance_between_the_two_beams_x / 2) ** 2 + (
                    Y + distance_between_the_two_beams_y / 2) ** 2) / wf ** 2)
        total_beam = beam_one + beam_two
        total_beam = total_beam / np.sqrt(sum(sum(abs(total_beam) ** 2)))
    else:
        x = np.arange(-N_pixels / 2, N_pixels / 2, 1) * 1
        y = x
        [X, Y] = np.meshgrid(x, y)
        c = 0
        distance_between_the_two_beams_y = 0
        beam_one = ((X - distance_between_the_two_beams_y / 2) ** 2 + (
                    Y - distance_between_the_two_beams_y / 2) ** 2 <= wf ** 2)
        beam_two = ((X + distance_between_the_two_beams_y / 2) ** 2 + (
                    Y + distance_between_the_two_beams_y / 2) ** 2 <= wf ** 2)
        total_beam = beam_one + beam_two

    ### Transfer to torch tensor: ###
    total_beam = torch.from_numpy(total_beam).to(device)
    total_beam = total_beam.unsqueeze(0).unsqueeze(0)

    def create_speckle_pattern():
        ### Randomize random phase screens: ###
        random_phase_screen = torch.exp(2*torch.pi*1j*10*torch.randn_like(total_beam))
        ### Perform Multiplicaiton by random phase screens: ###
        beam = total_beam * random_phase_screen
        speckle_pattern = torch_fftshift(torch.fft.fftn(torch_fftshift(beam), dim=(-2,-1)))
        return speckle_pattern
    """
    alpha = (decorr / 100)
    iterations = (1 / alpha) or (100 / decorr), basically how many iterations of decorrelation we do untill we need a new pattern
    change_counter = batch_size / iterations
    
    for i in range(change_counter):
        for j in range(iterations):
            new_patterns_for_place_(iterations*j) = get_decorrelation(p1, p2, decorr=[0, alpha, 2*alpha, ..., 1])    
    """
    # todo: make the dimensions seamless, i dont want batch size to must be divisible by decorr
    # how many times does the decor percentage fits in 100
    iterations = (100 / decorr_percentage).to(torch.int)
    # how many speckle patterns will the entire (video len)/(batch size) need...
    change_counter = (B / iterations).to(int)
    # this is floored so add 1 to always cover the entire batch size we need and not less
    if B % iterations != 0:
        change_counter += 1
    # convert to actual percentage
    decorr_percentage = (decorr_percentage / 100)
    # create an array of alpha blending, as much as fits in the 100 percentage(iterations counter)
    alphas = decorr_percentage.repeat(iterations).cumsum(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).flip(0)

    p1, p2 = create_speckle_pattern(), create_speckle_pattern()
    patterns = []
    for i in range(change_counter):
        decorr_speckle_pattern_total_intensity = (alphas*p1.repeat(iterations, 1, 1, 1) + (1-alphas)*p2.repeat(iterations, 1, 1, 1)).abs()**2

        p1 = p2
        p2 = create_speckle_pattern()
        if (i+1)*iterations > B:
            patterns.append(decorr_speckle_pattern_total_intensity[:(B % iterations)])
        else:
            patterns.append(decorr_speckle_pattern_total_intensity)

    patterns = torch.cat(patterns, 0)
    # imshow_torch(patterns[0])
    # imshow_torch(patterns[1])
    # imshow_torch(patterns[5])
    # imshow_torch(patterns[6])
    # imshow_torch(patterns[7])
    # imshow_torch(patterns[8])
    # imshow_torch(patterns[9])
    # imshow_torch(patterns[11])
    return patterns



# def get_speckled_illumination(input_tensor=None, speckle_size_in_pixels=10, polarization=0, max_decorrelation=0.05, max_beam_wonder=0.05, max_tilt_wonder=0.05):
#     # ### TODO: delete, for testing: ###
#     input_tensor = read_image_default_torch()
#     input_tensor = crop_torch_batch(input_tensor, (500, 500))
#     input_tensor = read_video_default_torch()
#     input_tensor = RGB2BW(input_tensor)
#     final_shape = input_tensor.shape
#     # final_shape = (100,1,500,500)
#     T,C,H,W = final_shape
#     speckle_size_in_pixels = 10
#     polarization = 0.5
#     max_decorrelation = 0.02
#     max_beam_wonder = 0.02
#     max_tilt_wonder = 0.01
#     speckle_contrast = 0.35
#
#     output_speckle_sequence = create_speckles_sequence(final_shape,
#                                                      speckle_size_in_pixels,
#                                                      polarization,
#                                                      speckle_contrast,
#                                                      max_decorrelation,
#                                                      max_beam_wonder,
#                                                      max_tilt_wonder)
#     output_tensor = input_tensor * output_speckle_sequence/output_speckle_sequence.max()
#
#     # imshow_torch_video(output_frames_torch, FPS=5, frame_stride=1)
#     # imshow_torch_video(BW2RGB(output_tensor*255).clamp(0,255).type(torch.uint8), FPS=5, frame_stride=1)
#
#     return output_tensor


def apply_svd_to_cc_on_speckles(ncc_speckle_shifts, n_comp=300, prompt="with_svd", device=0):
    ncc_speckle_shifts = torch.tensor(ncc_speckle_shifts).to(device)
    # H, W, C = ncc_speckle_shifts.shape
    # ncc_speckle_shifts = ncc_speckle_shifts[:, :, 1]

    U, S, V = tl.truncated_svd(ncc_speckle_shifts, n_eigenvecs=n_comp)
    var_explained = torch.round(S ** 2 / torch.sum(S ** 2), decimals=6)
    print(var_explained.cumsum(0))
    output_tensor = U @ torch.diag(S) @ V
    # imshow_torch(ncc_speckle_shifts);
    # imshow_torch(output_tensor);
    # plt.show();
    # wavfile.write(f"/home/yoav/rdnd/speckle_sounds_{prompt}.wav", 22050, output_tensor)

    return output_tensor


def average_shift_matrix(shift_matrix):
    """
    average shift[i, j] and shift[j, i] to create a more balanced shift
    Args:
        shift_matrix:

    Returns:

    """
    h, w, c = shift_matrix.shape
    corrected_shifts = torch.zeros_like(shift_matrix)
    assert c == 2
    assert h == w
    for i in range(h):
        for j in range(i+1, h):
            average_shift_ij = (shift_matrix[i, j] - shift_matrix[j, i]) / 2
            average_shift_ji = (shift_matrix[j, i] - shift_matrix[i, j]) / 2
            corrected_shifts[i, j] = average_shift_ij
            corrected_shifts[j, i] = average_shift_ji
    return corrected_shifts


def compute_one_fold_majority_vote(shift_matrix, weight_function=weight_function_parabolic_proximity):
    """
    given the pairwise shift matrix computes the one fold majority vote
    Args:
        shift_matrix: pair wise shift matrix

    Returns: corrected shift matrix
    """
    h, w, c = shift_matrix.shape
    corrected_shifts = torch.zeros_like(shift_matrix)
    assert c == 2
    assert h == w
    for i in range(h):
        for j in range(h):
            if i == j:
                continue
            # now compute 2 node distances over all k(shift(i, k) + shift(k, j)) over all k
            dist = 0
            weight_sum = 0
            for k in range(h):
                weight = weight_function(i, j, k)
                weight_sum += weight
                dist += weight * ((shift_matrix[i, k] - shift_matrix[k, j]) / 2)
            corrected_shifts[i, j] = dist / weight_sum
    return corrected_shifts


def corrupt_signal_with_turbulence(speckle_movie_path="/home/yoav/rdnd/speckle_movie.npy", std=1, device=0):
    # todo: create a version that supports speckle turbulence
    save_path = os.path.join(os.path.dirname(os.path.dirname(speckle_movie_path)), "noisy", os.path.basename(speckle_movie_path))
    os.makedirs(save_path, exist_ok=True)

    for id, frame_path in enumerate(sorted(glob(os.path.join(speckle_movie_path, "*.npy")))):
        raw_input_tensor = np.load(frame_path)

        # raw_input_tensor = raw_input_tensor[:20]

        raw_input_tensor = torch.tensor(raw_input_tensor).unsqueeze(1).to(device)

        # corrupt signal
        # raw_input_tensor = raw_input_tensor.permute(1, 0, 2, 3)
        B, C, H, W = raw_input_tensor.shape
        # turb_deform = TurbulenceDeformationLayerAnvil(H, W, B, device=0)
        # # to [b, c, h, w]
        # corrupted_input_tensor, delta_X, delta_y = turb_deform(raw_input_tensor, std=std)
        # corrupted_input_tensor = corrupted_input_tensor.squeeze().unsqueeze(1)
        turb_deform = TurbulenceDeformationLayerAnvil(H, W, 1, device=0)
        # corrupted_input_tensor = torch.zeros_like(raw_input_tensor)
        corrupted_tensor, delta_X, delta_y = turb_deform(raw_input_tensor, std=std)

        np.save(os.path.join(save_path, os.path.basename(frame_path)), corrupted_tensor.squeeze().unsqueeze(0).cpu().numpy())
        # save_video_torch_as_npy(corrupted_input_tensor, save_path)

        # np.save(save_path, corrupted_input_tensor.cpu().numpy())
        # imshow_torch(corrupted_input_tensor[5, :, :, :])
        # imshow_torch(raw_input_tensor[5, :, :, :])
        # plt.show()
    return save_path


def analyze_speckles():
    ## Analyze speckle sequence: ####
    ## Get Cross Correlation: ###
    input_tensor = np.load("/home/yoav/rdnd/speckle_movie.npy")
    input_tensor = torch.tensor(input_tensor).unsqueeze(1).cuda()
    # input_tensor = input_tensor[0:5000]
    warped_matrix, shifts_h, shifts_w, cc = align_to_reference_frame_circular_cc(input_tensor[1:], input_tensor[0:-1])

    ### Present Shift Inference Results: ###
    shifts_w = shifts_w.cpu().numpy()
    shifts_h = shifts_h.cpu().numpy()
    plt.figure()
    plt.plot(shifts_w[:20])
    plt.plot(shifts_h[:20])
    legend(['shift_x', 'shift_y'])
    plt.show()
    ### Sound outputs to earphones: ###
    # np.save("/home/yoav/rdnd/speckle_sounds.npy", shifts_w)
    wavfile.write("/home/yoav/rdnd/speckle_sounds.wav", 22050, shifts_w)
    # plt.plot(shifts_w[0:100])
    # plt.show()


# def save_torch_video_as_png_images(torch_video, path_to_save):
#     save_video_torch()
#     for id, frame in enumerate(torch_video):
#         save_image_torch(frame, f"{path_to_save}/{str(id).zfill(4)}")


def get_correlation_matrix_ncc(frames, shape, device="cpu"):
    frames = frames.to(device)
    num_frames = frames.shape[0]
    cross_correlation_matrix = torch.zeros(shape).to(device)
    for i in range(num_frames):
        print(i)
        for j in range(i+1, num_frames):
            cc = normalized_cc_wrapper(frames[i].unsqueeze(0), frames[j].unsqueeze(0), corr_size=3)
            delta_shiftx, delta_shifty, CC_peak_value = get_shifts_from_NCC(cc)
            cross_correlation_matrix[i, j, :] = torch.tensor([delta_shiftx, delta_shifty])

            cc = normalized_cc_wrapper(frames[j].unsqueeze(0), frames[i].unsqueeze(0), corr_size=3)
            delta_shiftx, delta_shifty, CC_peak_value = get_shifts_from_NCC(cc)
            cross_correlation_matrix[j, i, :] = torch.tensor([delta_shiftx, delta_shifty])

    # this might not be empirically correct so compute twice in the loop
    # cross_correlation_matrix = cross_correlation_matrix - cross_correlation_matrix.transpose(0, 1)
    return cross_correlation_matrix


def create_speckles_movie_with_song(wav_file_path="/home/yoav/rdnd/harvard.wav", save=False, SNR=np.inf, amplitude=0.2, device=0):
    ### Create speckles movie with song embedded into it: ###
    ### Load song to embedd in speckles: ###
    wave_file_path_res = os.path.join(os.path.dirname(os.path.dirname(wav_file_path)), "gt", os.path.basename(wav_file_path)[:-4])
    original_song, fs = librosa.load(wav_file_path)
    # load 1/4 to 3/4 of wav file
    number_of_samples = len(original_song) // 2
    n = len(original_song) // 4
    original_song = original_song[n:n+number_of_samples]

    # todo: changed here to not subtract the min because shift layer is fucked
    # original_song -= original_song.min()
    original_song /= np.abs(original_song).max()
    max_diff = amplitude

    original_song *= max_diff
    original_song_no_cumsum = torch.Tensor(original_song).clone().to(device)
    original_song = original_song.cumsum()
    original_song = torch.Tensor(original_song).to(device)
    ### Initialize shift layer to shift speckles: ###
    ###############################################################################################################
    shift_layer = Shift_Layer_Torch()
    ### Creet basic speckles pattern: ##
    final_size = 64
    max_shift = int(np.ceil(original_song.abs().max().item()))
    speckle_pattern_total_intensity, speckle_pattern_field1, speckle_pattern_field2, total_beam = create_speckles_of_certain_size_in_pixels(3, final_size + max_shift, 0, 1, 1, 0)
    speckle_pattern_total_intensity_torch = torch.Tensor(speckle_pattern_total_intensity).unsqueeze(0).to(device)
    shifted_speckles = torch.zeros((len(original_song), final_size, final_size))

    # speckle_pattern_intensity_shifted, shiftx, shifty = shift_layer.forward(speckle_pattern_total_intensity_torch, -original_song.detach().cpu().numpy(), np.array([0]).astype(float32))
    # # imshow_torch(speckle_pattern_intensity_shifted[0])
    # # imshow_torch(speckle_pattern_intensity_shifted[5])
    # # imshow_torch(speckle_pattern_intensity_shifted[10])
    # # imshow_torch(speckle_pattern_intensity_shifted[-4000])
    # # add noise
    # speckle_pattern_intensity_shifted_std = speckle_pattern_intensity_shifted.std([1, 2])
    # sigma_noise_speckle_pattern_intensity_shifted = speckle_pattern_intensity_shifted_std / SNR  # snr = std_signal / std_noise
    # speckle_pattern_intensity_shifted += sigma_noise_speckle_pattern_intensity_shifted.unsqueeze(1).unsqueeze(1) * torch.randn_like(speckle_pattern_intensity_shifted)
    # # crop
    # speckle_pattern_total_intensity_torch = crop_tensor(speckle_pattern_total_intensity_torch, (final_size, final_size))
    # speckle_pattern_intensity_shifted = crop_tensor(speckle_pattern_intensity_shifted, (final_size, final_size))
    # # speckle_pattern_intensity_shifted = speckle_pattern_intensity_shifted.abs()**2
    # # imshow_torch(speckle_pattern_intensity_shifted[0].abs()**2)
    # # imshow_torch(speckle_pattern_intensity_shifted[5])
    # # imshow_torch(speckle_pattern_intensity_shifted[10])
    # # imshow_torch(speckle_pattern_intensity_shifted[-4000])
    for i in range(len(original_song)):
        if i % 1000 == 0:
            print(f"{i}/{len(original_song)}")
        shift_x = original_song[i]
        shift_y = 0.0
        speckle_pattern_intensity_shifted, shiftx, shifty, _, _ = shift_layer.forward(speckle_pattern_total_intensity_torch, -shift_x.detach().cpu().numpy(), np.array([-shift_y]).astype(float32))
        # speckle_pattern_intensity_shifted, shiftx, shifty = shift_layer.forward(speckle_pattern_total_intensity_torch, -np.array(-1000), -np.array(-1000))
        # imshow_torch(speckle_pattern_total_intensity_torch)
        # imshow_torch(speckle_pattern_intensity_shifted)
        # add noise
        speckle_pattern_intensity_shifted_std = speckle_pattern_intensity_shifted.std()
        sigma_noise_speckle_pattern_intensity_shifted = speckle_pattern_intensity_shifted_std / SNR  # snr = std_signal / std_noise
        speckle_pattern_intensity_shifted += sigma_noise_speckle_pattern_intensity_shifted * torch.randn_like(speckle_pattern_intensity_shifted)
        # crop
        # speckle_pattern_total_intensity_torch = crop_tensor(speckle_pattern_total_intensity_torch, (final_size, final_size))
        speckle_pattern_intensity_shifted = crop_tensor(speckle_pattern_intensity_shifted, (final_size, final_size))

        shifted_speckles[i] = speckle_pattern_intensity_shifted.squeeze()
    speckle_pattern_intensity_shifted = shifted_speckles

    # imshow_torch_video_seamless(speckle_pattern_intensity_shifted.unsqueeze(1), FPS=150, number_of_frames=500)

    save_path = wave_file_path_res
    if save:
        os.makedirs(save_path, exist_ok=True)
        save_video_torch_as_npy(shifted_speckles.unsqueeze(1), save_path)

    return speckle_pattern_intensity_shifted, save_path, original_song_no_cumsum
    #
    # shiftx_array = np.array(shiftx_list)
    # shifty_array = np.array(shifty_list)
    # ##############
    # plt.plot(original_song_no_cumsum.cpu().detach().numpy()); plt.show();
    # plt.plot(original_song.cpu().detach().numpy()); plt.show();
    #
    # shiftx_mean = np.mean(shiftx_array)
    # shiftx_std = np.std(shiftx_array - shiftx_mean)
    # plt.title('shiftx STD: ' + str(shiftx_std) + ',  Mean: ' + str(shiftx_mean))
    # plt.show()
    # ### Plot x: ###
    # plt.plot(shiftx_array);
    # shiftx_mean = np.mean(shiftx_array)
    # shiftx_std = np.std(shiftx_array-shiftx_mean)
    # plt.title('shiftx STD: ' + str(shiftx_std) + ',  Mean: ' + str(shiftx_mean))
    # plt.show()
    # ### Plot y: ###
    # plt.plot(shifty_array);
    # shifty_mean = np.mean(shifty_array)
    # shifty_std = np.std(shifty_array - shifty_mean)
    # plt.title('shifty STD: ' + str(shifty_std) + ',  Mean: ' + str(shifty_mean))
    # plt.show()
    #
    #
    #
    # shifts_h, shifts_w, cc = normalized_cc_shifts(speckle_pattern_total_intensity_torch, speckle_pattern_intensity_shifted, 3)
    # print(shifts_w)
    # print(shifts_h)
    # shifts_w = shifts_w.squeeze().cpu().numpy()
    # shifts_h = shifts_h.squeeze().cpu().numpy()
    # plt.figure()
    # plt.plot(shifts_w[:20])
    # plt.plot(shifts_h[:20])
    # legend(['shift_x', 'shift_y'])
    # plt.show()
    #
    # delta_shiftx = -b_x / (2 * a_x)
    # delta_shifty = -b_y / (2 * a_y)

    # ############################################################################################################




    # ############################################################################################################
    # ### Get Input Tensor: ###
    # # input_tensor = torch.cat([speckle_pattern_total_intensity_torch]*number_of_samples, 0)
    # input_tensor = torch.cat([speckle_pattern_field1_torch]*number_of_samples)
    # current_shift = 0
    # ### Shift basic speckles pattern according to song: ###
    # shifted_tensor, shiftx, shifty = shift_layer.forward(input_tensor, np.ones_like(original_song.cpu().numpy()), 0*original_song.cpu().numpy())
    # # imshow_torch(input_tensor[0].abs()); imshow_torch(shifted_tensor[10].abs()); imshow_torch(shifted_tensor[-1].abs()); plt.show()
    # imshow_torch(input_tensor.abs()[10])
    # shifted_tensor = shifted_tensor.abs()**2
    # shifted_tensor = crop_tensor(shifted_tensor, 64)
    # # ### Present Results: ###
    # # imshow_torch(input_tensor[0])
    # # imshow_torch(shifted_tensor[46204])
    # # imshow_torch(shifted_tensor[46205])
    # # plt.show()
    # # ### Save original movie with ofra haza inside: ###
    # # np_file_save_res = wave_file_path_res[:-4] + ".npy"
    # # np.save(np_file_save_res, input_tensor.numpy())
    # save_path = wave_file_path_res
    # if save:
    #     os.makedirs(save_path, exist_ok=True)
    #     save_video_torch_as_npy(shifted_tensor.unsqueeze(1), save_path)
    # return shifted_tensor, save_path, original_song_no_cumsum
    # #########################################################################################################


def add_noise_to_speckle_with_given_snr(speckle, SNR=np.inf):
    speckle_std = speckle.std()
    sigma_noise_speckle = speckle_std / SNR  # snr = std_signal / std_noise
    noise_term = torch_get_4D(sigma_noise_speckle, 'T').squeeze(0) * torch.randn_like(speckle)
    return speckle + noise_term


def create_speckles_movie_with_song_n_lasers(wav_file_path="/home/yoav/rdnd/harvard.wav", save=False, SNR=np.inf, amplitude=0.2, laser_count=1, speckle_size=3, number_of_samples=None, demo=False, device=0):
    ### Create speckles movie with song embedded into it: ###
    ### Load song to embedd in speckles: ###
    original_song, fs = librosa.load(wav_file_path)
    # plt.plot(original_song[56000:160000]); plt.show();
    if demo: # demo means i want to take a specific song with specific time stamps to test audio result
        original_song = original_song[60000:160000]
    else:
        if number_of_samples:
            number_of_samples = number_of_samples
        else:
            number_of_samples = len(original_song) // 2
        n = len(original_song) // 4
        # load 1/4 to 3/4 of wav file(given that no number_of_samples was given)
        original_song = original_song[n:n+number_of_samples]

    append_zeros = False
    if append_zeros:
        original_song = np.concatenate([np.zeros(25000), original_song])

    original_song /= np.abs(original_song).max()
    max_diff = amplitude

    original_song = original_song.cumsum()
    window_size = 10
    # todo: formalize this normalization, this is a bit tricky,
    #       i want a max diff between speckles of given amplitude but my algorithm later on also uses speckles that are not near each other,
    #       which given the cum sum gives us a larger difference than expected, this must be countered using a smaller amplitude,
    #       if i have a window of 10 than i will compare speckle 1 with speckle 11, thus i need a max difference that is
    #       order of magnitude(exactly window size) smaller

    original_song *= ((1/window_size) * max_diff)
    original_song = torch.tensor(original_song, dtype=torch.double).to(device)
    gt_shifts = original_song[1:] - original_song[0:-1]
    ### Initialize shift layer to shift speckles: ###
    ###############################################################################################################
    shift_layer = Shift_Layer_Torch()
    ### Creet basic speckles pattern: ##
    final_size = 64
    max_shift = int(np.ceil(original_song.abs().max().item()))
    speckle_patterns = torch.zeros((laser_count, final_size + max_shift, final_size + max_shift), dtype=torch.double)
    # create a speckle pattern for each laser to add up
    for i in range(laser_count):
        speckle_pattern_total_intensity, speckle_pattern_field1, speckle_pattern_field2, total_beam = create_speckles_of_certain_size_in_pixels(speckle_size, final_size + max_shift, 0, 1, 1, 0)
        speckle_pattern_total_intensity_torch = torch.Tensor(speckle_pattern_total_intensity).to(device)
        speckle_patterns[i] = speckle_pattern_total_intensity_torch

    # shifted_speckles = torch.zeros((len(original_song), final_size, final_size), dtype=torch.double)
    # shifted_speckles_no_noise_for_gt = torch.zeros((len(original_song), final_size, final_size), dtype=torch.double)

    # speckle_pattern_intensity_shifted, shiftx, shifty = shift_layer.forward(speckle_pattern_total_intensity_torch, -original_song.detach().cpu().numpy(), np.array([0]).astype(float32))
    # # imshow_torch(speckle_pattern_intensity_shifted[0])
    # # imshow_torch(speckle_pattern_intensity_shifted[5])
    # # imshow_torch(speckle_pattern_intensity_shifted[10])
    # # imshow_torch(speckle_pattern_intensity_shifted[-4000])
    # # add noise
    # speckle_pattern_intensity_shifted_std = speckle_pattern_intensity_shifted.std([1, 2])
    # sigma_noise_speckle_pattern_intensity_shifted = speckle_pattern_intensity_shifted_std / SNR  # snr = std_signal / std_noise
    # speckle_pattern_intensity_shifted += sigma_noise_speckle_pattern_intensity_shifted.unsqueeze(1).unsqueeze(1) * torch.randn_like(speckle_pattern_intensity_shifted)
    # # crop
    # speckle_pattern_total_intensity_torch = crop_tensor(speckle_pattern_total_intensity_torch, (final_size, final_size))
    # speckle_pattern_intensity_shifted = crop_tensor(speckle_pattern_intensity_shifted, (final_size, final_size))
    # # speckle_pattern_intensity_shifted = speckle_pattern_intensity_shifted.abs()**2
    # # imshow_torch(speckle_pattern_intensity_shifted[0].abs()**2)
    # # imshow_torch(speckle_pattern_intensity_shifted[5])
    # # imshow_torch(speckle_pattern_intensity_shifted[10])
    # # imshow_torch(speckle_pattern_intensity_shifted[-4000])

    # shift and crop each speckle according to song, note that the first speckle is also shifted,
    # thus if we calculate shift between first and second speckle we would get the second value from the song
    # we could not do that or simply know it exists and fix it after...
    # for i in range(len(original_song)):
    #     # if i % 1000 == 0:
    #     #     print(f"{i}/{len(original_song)}")
    #     # set the shift_x to be the song value(post cumsum because we later on compute difference to regenerate song)
    #     shift_x = original_song[i]
    #     shift_y = 0.0
    #     shifted_one_speckle_patterns = torch.zeros_like(speckle_patterns, dtype=torch.double)
    #     shifted_one_speckle_patterns_no_noise = torch.zeros_like(speckle_patterns, dtype=torch.double)
    #
    #     # shift each speckle pattern the same and add random noise to each
    #     for j in range(laser_count):
    #         speckle_pattern_intensity_shifted, shiftx, shifty, _, _ = shift_layer.forward(speckle_patterns[j:j+1], -shift_x.detach().cpu().numpy(), np.array([-shift_y]).astype(float32))
    #         # add noise
    #         # speckle_pattern_intensity_shifted_std = speckle_pattern_intensity_shifted.std()
    #         # sigma_noise_speckle_pattern_intensity_shifted = speckle_pattern_intensity_shifted_std / SNR  # snr = std_signal / std_noise
    #         # speckle_pattern_intensity_shifted += sigma_noise_speckle_pattern_intensity_shifted * torch.randn_like(speckle_pattern_intensity_shifted)
    #         temp_to_save_gt = speckle_pattern_intensity_shifted.clone()
    #         speckle_pattern_intensity_shifted = add_noise_to_speckle_with_given_snr(speckle_pattern_intensity_shifted, SNR)
    #
    #         shifted_one_speckle_patterns[j] = speckle_pattern_intensity_shifted
    #         shifted_one_speckle_patterns_no_noise[j] = temp_to_save_gt
    #
    #     # average all laser speckles across different lasers
    #     shifted_one_speckle_patterns = shifted_one_speckle_patterns.mean(0)
    #     shifted_one_speckle_patterns_no_noise = shifted_one_speckle_patterns_no_noise.mean(0)
    #     # speckle_pattern_intensity_shifted, shiftx, shifty = shift_layer.forward(speckle_pattern_total_intensity_torch, -np.array(-1000), -np.array(-1000))
    #     # imshow_torch(speckle_pattern_total_intensity_torch)
    #     # imshow_torch(speckle_pattern_intensity_shifted)
    #     # crop
    #     # speckle_pattern_total_intensity_torch = crop_tensor(speckle_pattern_total_intensity_torch, (final_size, final_size))
    #     # crop final shifted speckle
    #     speckle_pattern_intensity_shifted = crop_tensor(shifted_one_speckle_patterns, (final_size, final_size))
    #     shifted_one_speckle_patterns_no_noise = crop_tensor(shifted_one_speckle_patterns_no_noise, (final_size, final_size))
    #     # add to speckle movie
    #     shifted_speckles[i] = speckle_pattern_intensity_shifted.squeeze()
    #     shifted_speckles_no_noise_for_gt[i] = shifted_one_speckle_patterns_no_noise.squeeze()


    speckle_pattern_intensity_shifted = torch.zeros((len(original_song), final_size + max_shift, final_size + max_shift), dtype=torch.double)
    for j in range(laser_count):
        speckle_pattern_shifted, _ = _shift_matrix_subpixel_fft_batch(speckle_patterns[j:j + 1].repeat(len(original_song), 1, 1), 0.0, -original_song)
        speckle_pattern_shifted += add_noise_to_speckle_with_given_snr(speckle_pattern_shifted, SNR)
        speckle_pattern_intensity_shifted += speckle_pattern_shifted

    speckle_pattern_intensity_shifted /= laser_count
    speckle_pattern_intensity_shifted = crop_tensor(speckle_pattern_intensity_shifted, (final_size, final_size))

    # imshow_torch_video_seamless(speckle_pattern_intensity_shifted.unsqueeze(1), FPS=150, number_of_frames=500)
    # imshow_torch(speckle_pattern_intensity_shifted[0])
    # imshow_torch(shifted_speckles_no_noise_for_gt[0])

    wave_file_path_res_gt = os.path.join(os.path.dirname(os.path.dirname(wav_file_path)), "gt", os.path.basename(wav_file_path)[:-4])
    wave_file_path_res_noisy = os.path.join(os.path.dirname(os.path.dirname(wav_file_path)), "noisy", os.path.basename(wav_file_path)[:-4])

    # if save:
    #     os.makedirs(wave_file_path_res_gt, exist_ok=True)
    #     save_video_torch_as_npy(shifted_speckles_no_noise_for_gt.unsqueeze(1), wave_file_path_res_gt)
    #
    #     os.makedirs(wave_file_path_res_noisy, exist_ok=True)
    #     save_video_torch_as_npy(speckle_pattern_intensity_shifted.unsqueeze(1), wave_file_path_res_noisy)

    # remove the first shift(from original_song_no_cumsum) since all the speckles were shifted according to song, the first diff is corresponding to second shift
    return speckle_pattern_intensity_shifted, gt_shifts

    # speckle_pattern_intensity_shifted -= speckle_pattern_intensity_shifted.min()
    # speckle_pattern_intensity_shifted /= speckle_pattern_intensity_shifted.max()
    #
    # save_video_torch(speckle_pattern_intensity_shifted.unsqueeze(1), "/raid/datasets/speckles/speckle_movies/SNR=np.inf", flag_convert_bgr2rgb=False, flag_scale_by_255=True)


def get_chirp_signal(fs, alpha):
    signal = [2*math.pi*alpha*t*t for t in range(fs//2)]
    signal = torch.tensor(signal)
    signal = torch.sin(signal)
    return signal

# signal = get_chirp_signal(500, 0.0003)
# plt.plot(gt_shifts_x.cpu()); plt.show();
# n_fft = 128
# win_length = None
# hop_length = 64
# spectrogram = T.Spectrogram(
#     n_fft=n_fft,
#     win_length=win_length,
#     hop_length=hop_length,
#     center=True,
#     pad_mode="reflect",
#     power=2.0,
# )
# spec = spectrogram(signal)
# imshow_torch(torch.tensor(spec))

def create_speckles_movie_with_song_n_lasers_plus_y_movement(wav_file_path="/raid/datasets/speckles/wav_files_for_rvrt/bush_never_once.wav", save=False,
                                                             SNR=np.inf, decorrelation=0, amplitude=0.2, laser_count=1,
                                                             speckle_size=3, number_of_samples=None, demo=False,
                                                             angle=90, NU_strength=0, device=0, shir_poretz_flag='shir'):
    ### Create speckles movie with song embedded into it: ###
    ### Load song to embedd in speckles: ###

    if demo:  # demo means i want to take a specific song with specific time stamps to test audio result
        original_song, fs = librosa.load(wav_file_path)
        original_song = original_song[:30000]
    else:  # otherwise just use a chirp signal
        original_song = get_chirp_signal(100, 0.0003)
        original_song = torch.cat((original_song, (torch.zeros(150))))
        original_song = torch.cat((original_song, get_chirp_signal(100, 0.0003)))

    original_song /= np.abs(original_song).max()
    max_diff = amplitude

    original_song = original_song.cumsum(0)
    # todo: formalize this normalization, this is a bit tricky,
    #       i want a max diff between speckles of given amplitude but my algorithm later on also uses speckles that are not near each other,
    #       which given the cum sum gives us a larger difference than expected, this must be countered using a smaller amplitude,
    #       if i have a window of 10 than i will compare speckle 1 with speckle 11, thus i need a max difference that is
    #       order of magnitude(exactly window size) smaller

    original_song *= max_diff
    original_song = torch.tensor(original_song, dtype=torch.double).to(device)
    gt_shifts_x = original_song[1:] - original_song[0:-1]
    y_shifts = original_song * (angle / 90)
    gt_shifts_y = y_shifts[1:] - y_shifts[0:-1]
    # y_shifts = torch.rand(original_song.shape[0]) * original_song.mean()
    ### Initialize shift layer to shift speckles: ###
    ###############################################################################################################
    ### Create basic speckles pattern: ##
    final_size = 64
    max_shift = int(np.ceil(original_song.abs().max().item()))
    speckle_patterns = torch.zeros((laser_count, len(original_song), final_size + max_shift, final_size + max_shift), dtype=torch.double)
    # create a speckle pattern for each laser to add up
    #  todo: batch this , see dense_cc_n generation function
    for i in range(laser_count):
        # speckle_pattern_total_intensity, speckle_pattern_field1, speckle_pattern_field2, total_beam = create_speckles_of_certain_size_in_pixels(speckle_size, final_size + max_shift, 0, 1, 1, 0)
        # video_speckle_pattern_total_intensity = create_decorrelated_speckle_video_torch(speckle_size, len(original_song), final_size + max_shift, 0, 1, 1, 0, torch.tensor(decorrelation).to(device), device)
        if shir_poretz_flag == 'shir':
            video_speckle_pattern_total_intensity = create_shir_decorrelated_speckle_video_torch(speckle_size, len(original_song), final_size + max_shift, 0, 1, 1, 0, torch.tensor(decorrelation).to(device), device)
        elif shir_poretz_flag == 'poretz':
            video_speckle_pattern_total_intensity = create_poretz_decorrelated_speckle_video_torch(speckle_size, len(original_song), final_size + max_shift, 0, 1, 1, 0, torch.tensor(decorrelation).to(device), device)

        speckle_pattern_total_intensity_torch = torch.Tensor(video_speckle_pattern_total_intensity).to(device)
        speckle_patterns[i] = speckle_pattern_total_intensity_torch.squeeze()

    speckle_pattern_intensity_shifted = torch.zeros((len(original_song), final_size + max_shift, final_size + max_shift), dtype=torch.double)
    for j in range(laser_count): # todo: batch this , see dense_cc_n generation function
        speckle_pattern_shifted, _ = _shift_matrix_subpixel_fft_batch(speckle_patterns[j], -y_shifts, -original_song)
        # todo: test this
        speckle_pattern_shifted -= torch.min(torch.min(speckle_pattern_shifted, 1)[0], 1)[0].unsqueeze(-1).unsqueeze(-1)  # normalize speckle pattern to fit SNR function
        speckle_pattern_shifted /= torch.max(torch.max(speckle_pattern_shifted, 1)[0], 1)[0].unsqueeze(-1).unsqueeze(-1)
        speckle_pattern_shifted += add_noise_to_speckle_with_given_snr(speckle_pattern_shifted, SNR)
        speckle_pattern_intensity_shifted += speckle_pattern_shifted

    speckle_pattern_intensity_shifted /= laser_count
    speckle_pattern_intensity_shifted = crop_tensor(speckle_pattern_intensity_shifted, (final_size, final_size))

    noise_pattern_path = "/raid/datasets/new_NU/Kolmogorov_NU/r0_1p0e+00__L0_1p0e-01__l0_1p0e+00/0000.png"
    NU_part = NU_strength * read_image_torch(noise_pattern_path)[0, 0:1, :64, :64]
    # imshow_torch(speckle_pattern_intensity_shifted[0])
    # imshow_torch(speckle_pattern_intensity_shifted[1])
    return speckle_pattern_intensity_shifted + NU_part, gt_shifts_x, gt_shifts_y


def create_turbulence_speckles_movie_with_song_n_lasers_plus_y_movement(wav_file_path="/raid/datasets/speckles/wav_files_for_rvrt/bush_never_once.wav", save=False,
                                                             SNR=np.inf, amplitude=0.2, laser_count=1,
                                                             speckle_size=3, number_of_samples=None, demo=False,
                                                             angle=90, NU_strength=0, device=0,
                                                            turbulence_factor="1", static=False):
    ### Create speckles movie with song embedded into it: ###
    ### Load song to embedd in speckles: ###

    if demo:  # demo means i want to take a specific song with specific time stamps to test audio result
        original_song, fs = librosa.load(wav_file_path)
        original_song = original_song[:30000]
    else:  # otherwise just use a chirp signal
        original_song = get_chirp_signal(100, 0.0003)
        original_song = torch.cat((original_song, (torch.zeros(150))))
        original_song = torch.cat((original_song, get_chirp_signal(100, 0.0003)))

    original_song /= np.abs(original_song).max()
    max_diff = amplitude

    original_song = original_song.cumsum(0)
    # todo: formalize this normalization, this is a bit tricky,
    #       i want a max diff between speckles of given amplitude but my algorithm later on also uses speckles that are not near each other,
    #       which given the cum sum gives us a larger difference than expected, this must be countered using a smaller amplitude,
    #       if i have a window of 10 than i will compare speckle 1 with speckle 11, thus i need a max difference that is
    #       order of magnitude(exactly window size) smaller

    original_song *= max_diff
    original_song = torch.tensor(original_song, dtype=torch.double).to(device)
    gt_shifts_x = original_song[1:] - original_song[0:-1]
    y_shifts = original_song * (angle / 90)
    gt_shifts_y = y_shifts[1:] - y_shifts[0:-1]
    # y_shifts = torch.rand(original_song.shape[0]) * original_song.mean()
    ### Initialize shift layer to shift speckles: ###
    ###############################################################################################################
    ### Create basic speckles pattern: ##
    final_size = 64
    max_shift = int(np.ceil(original_song.abs().max().item()))
    speckle_patterns = torch.zeros((laser_count, len(original_song), final_size + max_shift, final_size + max_shift), dtype=torch.double)
    # create a speckle pattern for each laser to add up
    #  todo: batch this , see dense_cc_n generation function
    for i in range(laser_count):
        # speckle_pattern_total_intensity, speckle_pattern_field1, speckle_pattern_field2, total_beam = create_speckles_of_certain_size_in_pixels(speckle_size, final_size + max_shift, 0, 1, 1, 0)
        # video_speckle_pattern_total_intensity = create_decorrelated_speckle_video_torch(speckle_size, len(original_song), final_size + max_shift, 0, 1, 1, 0, torch.tensor(decorrelation).to(device), device)

        # load mat file with turbulence and add to speckles
        gui_factor = 250
        try:
            # video_beam = scipy.io.loadmat(f"/raid/datasets/turbulence_speckles/TurbVideoForGUI/{float(gui_factor * eval(turbulence_factor))}/3/l0=0.015,L0=50,Cn2=1e-15,z=1000,wf=0.1.mat")['beam_field_on_surface_full_mat']
            video_beam = scipy.io.loadmat(f"/raid/datasets/turbulence_speckles/NoGlobalShift/long_videos_test/{float(gui_factor * eval(turbulence_factor))}/3/l0=0.015,L0=50,Cn2=1e-15,z=1000,wf=0.1.mat")['beam_field_on_surface_full_mat']
            # video_beam = scipy.io.loadmat(f"/raid/datasets/turbulence_speckles/TurbVideoForGUI/75/1/l0=0.015,L0=50,Cn2=1e-15,z=1000,wf=0.1.mat")['beam_field_on_surface_full_mat']
        except:
            # video_beam = scipy.io.loadmat(f"/raid/datasets/turbulence_speckles/TurbVideoForGUI/{int(gui_factor * eval(turbulence_factor))}/3/l0=0.015,L0=50,Cn2=1e-15,z=1000,wf=0.1.mat")['beam_field_on_surface_full_mat']
            video_beam = scipy.io.loadmat(f"/raid/datasets/turbulence_speckles/NoGlobalShift/long_videos_test/{float(gui_factor * eval(turbulence_factor))}/3/l0=0.015,L0=50,Cn2=1e-15,z=1000,wf=0.1.mat")['beam_field_on_surface_full_mat']
            # video_beam = scipy.io.loadmat(f"/raid/datasets/turbulence_speckles/TurbVideoForGUI/75/1/l0=0.015,L0=50,Cn2=1e-15,z=1000,wf=0.1.mat")['beam_field_on_surface_full_mat']

        beams = torch.tensor(video_beam[:len(original_song)]).to(device)
        if static:
            beams = beams[0:1].repeat(beams.shape[0], 1, 1)
        # normalized_cc_wrapper(beams[0:1], beams[1:2], corr_size=5)
        T, H, W = beams.shape
        # imshow_torch(beams[0].abs())
        # imshow_torch(beams[10].abs())

        # generate random phase screens of shape [batch_size]
        random_phase_screens = torch.exp(2 * torch.pi * 1j * 10 * torch.randn((1, H, W))).to(device)

        # generate speckle pairs of shape [batch_size, 2]
        speckles_fft = beams * random_phase_screens
        speckles = torch_fftshift(torch.fft.fftn(torch_fftshift(speckles_fft), dim=(-2, -1)))
        speckles = speckles.abs() ** 2
        # imshow_torch(speckles[0])
        # imshow_torch(speckles[10])

        factor = 2
        speckles = crop_tensor(speckles, factor*(final_size + max_shift))
        speckles = torch.nn.functional.avg_pool2d(speckles, factor)
        speckle_pattern_total_intensity_torch = torch.Tensor(speckles).to(device)
        speckle_patterns[i] = speckle_pattern_total_intensity_torch.squeeze()

    # imshow_torch(speckles[0])
    # imshow_torch(speckles[1])
    # normalized_cc_wrapper(speckles[0:1], speckles[1:2])

    speckle_pattern_intensity_shifted = torch.zeros((len(original_song), final_size + max_shift, final_size + max_shift), dtype=torch.double)
    for j in range(laser_count): # todo: batch this , see dense_cc_n generation function
        speckle_pattern_shifted, _ = _shift_matrix_subpixel_fft_batch(speckle_patterns[j], -y_shifts, -original_song)
        # todo: test this
        speckle_pattern_shifted -= torch.min(torch.min(speckle_pattern_shifted, 1)[0], 1)[0].unsqueeze(-1).unsqueeze(-1)  # normalize speckle pattern to fit SNR function
        speckle_pattern_shifted /= torch.max(torch.max(speckle_pattern_shifted, 1)[0], 1)[0].unsqueeze(-1).unsqueeze(-1)
        speckle_pattern_shifted += add_noise_to_speckle_with_given_snr(speckle_pattern_shifted, SNR)
        speckle_pattern_intensity_shifted += speckle_pattern_shifted

    speckle_pattern_intensity_shifted /= laser_count
    speckle_pattern_intensity_shifted = crop_tensor(speckle_pattern_intensity_shifted, (final_size, final_size))

    # imshow_torch(speckle_pattern_intensity_shifted[0])
    # imshow_torch(speckle_pattern_intensity_shifted[1])
    # imshow_torch(speckle_pattern_intensity_shifted[5])
    # imshow_torch(speckle_pattern_intensity_shifted[10])

    noise_pattern_path = "/raid/datasets/new_NU/Kolmogorov_NU/r0_1p0e+00__L0_1p0e-01__l0_1p0e+00/0000.png"
    NU_part = NU_strength * read_image_torch(noise_pattern_path)[0, 0:1, :64, :64]

    return speckle_pattern_intensity_shifted + NU_part, gt_shifts_x, gt_shifts_y




def demo_create_basic_speckles(SNR=np.inf, laser_count=1, speckle_size=3, device=0):
    ### Create speckles movie with song embedded into it: ###
    ### Load song to embedd in speckles: ###
    original_song = [0.016, 0.0165, 0.0162, 0.0156, 0.0162, 0.0155, 0.0158, 0.0161, 0.0159, 0.0157]

    original_song_no_cumsum = torch.tensor(original_song).clone().to(device)
    original_song = torch.tensor(original_song).to(device)
    original_song = original_song.cumsum(0)
    ### Initialize shift layer to shift speckles: ###
    ###############################################################################################################
    shift_layer = Shift_Layer_Torch()
    ### Creet basic speckles pattern: ##
    final_size = 64
    max_shift = int(np.ceil(original_song.abs().max().item()))
    speckle_patterns = torch.zeros((laser_count, final_size + max_shift, final_size + max_shift))
    for i in range(laser_count):
        speckle_pattern_total_intensity, speckle_pattern_field1, speckle_pattern_field2, total_beam = create_speckles_of_certain_size_in_pixels(speckle_size, final_size + max_shift, 0, 1, 1, 0)
        speckle_pattern_total_intensity_torch = torch.Tensor(speckle_pattern_total_intensity).to(device)
        speckle_patterns[i] = speckle_pattern_total_intensity_torch

    shifted_speckles = torch.zeros((len(original_song), final_size, final_size))

    # speckle_pattern_intensity_shifted, shiftx, shifty = shift_layer.forward(speckle_pattern_total_intensity_torch, -original_song.detach().cpu().numpy(), np.array([0]).astype(float32))
    # # imshow_torch(speckle_pattern_intensity_shifted[0])
    # # imshow_torch(speckle_pattern_intensity_shifted[5])
    # # imshow_torch(speckle_pattern_intensity_shifted[10])
    # # imshow_torch(speckle_pattern_intensity_shifted[-4000])
    # # add noise
    # speckle_pattern_intensity_shifted_std = speckle_pattern_intensity_shifted.std([1, 2])
    # sigma_noise_speckle_pattern_intensity_shifted = speckle_pattern_intensity_shifted_std / SNR  # snr = std_signal / std_noise
    # speckle_pattern_intensity_shifted += sigma_noise_speckle_pattern_intensity_shifted.unsqueeze(1).unsqueeze(1) * torch.randn_like(speckle_pattern_intensity_shifted)
    # # crop
    # speckle_pattern_total_intensity_torch = crop_tensor(speckle_pattern_total_intensity_torch, (final_size, final_size))
    # speckle_pattern_intensity_shifted = crop_tensor(speckle_pattern_intensity_shifted, (final_size, final_size))
    # # speckle_pattern_intensity_shifted = speckle_pattern_intensity_shifted.abs()**2
    # # imshow_torch(speckle_pattern_intensity_shifted[0].abs()**2)
    # # imshow_torch(speckle_pattern_intensity_shifted[5])
    # # imshow_torch(speckle_pattern_intensity_shifted[10])
    # # imshow_torch(speckle_pattern_intensity_shifted[-4000])

    # shift and crop each speckle according to song, note that the first speckle is also shifted,
    # thus if we calculate shift between first and second speckle we would get the second value from the song
    # we could not do that or simply know it exists and fix it after...
    for i in range(len(original_song)):
        # if i % 1000 == 0:
        #     print(f"{i}/{len(original_song)}")
        shift_x = original_song[i]
        shift_y = 0.0
        shifted_one_speckle_patterns = torch.zeros_like(speckle_patterns)

        for j in range(laser_count):
            speckle_pattern_intensity_shifted, shiftx, shifty = shift_layer.forward(speckle_patterns[j:j+1], -shift_x.detach().cpu().numpy(), np.array([-shift_y]).astype(float32))
            # add noise
            # speckle_pattern_intensity_shifted_std = speckle_pattern_intensity_shifted.std()
            # sigma_noise_speckle_pattern_intensity_shifted = speckle_pattern_intensity_shifted_std / SNR  # snr = std_signal / std_noise
            # speckle_pattern_intensity_shifted += sigma_noise_speckle_pattern_intensity_shifted * torch.randn_like(speckle_pattern_intensity_shifted)
            speckle_pattern_intensity_shifted = add_noise_to_speckle_with_given_snr(speckle_pattern_intensity_shifted, SNR)

            shifted_one_speckle_patterns[j] = speckle_pattern_intensity_shifted

        shifted_one_speckle_patterns = shifted_one_speckle_patterns.mean(0)
        # speckle_pattern_intensity_shifted, shiftx, shifty = shift_layer.forward(speckle_pattern_total_intensity_torch, -np.array(-1000), -np.array(-1000))
        # imshow_torch(speckle_pattern_total_intensity_torch)
        # imshow_torch(speckle_pattern_intensity_shifted)
        # crop
        # speckle_pattern_total_intensity_torch = crop_tensor(speckle_pattern_total_intensity_torch, (final_size, final_size))
        speckle_pattern_intensity_shifted = crop_tensor(shifted_one_speckle_patterns, (final_size, final_size))

        shifted_speckles[i] = speckle_pattern_intensity_shifted.squeeze()

    speckle_pattern_intensity_shifted = shifted_speckles


    # remove the first shift(from original_song_no_cumsum) since all the speckles were shifted according to song, the first diff is corresponding to second shift
    return speckle_pattern_intensity_shifted, original_song_no_cumsum[1:]



def fit_polynomial(x: Union[torch.Tensor, list], y: Union[torch.Tensor, list]) -> List[float]:
    # This is correct - checked by desmos :)
    # solve for 2nd degree polynomial deterministically using three points seperated by distance of 1
    a = (y[..., 2] + y[..., 0] - 2 * y[..., 1]) / 2
    b = -(y[..., 0] + 2 * a * x[1] - y[..., 1] - a)
    c = y[..., 1] - b * x[1] - a * x[1] ** 2
    return [c, b, a]


def fit_gaussian(x: Union[torch.Tensor, list], y: Union[torch.Tensor, list]) -> List[float]:
    # assume x is [-1, 0, 1]
    term_0 = torch.log(y[..., 0])
    term_1 = torch.log(y[..., 1])
    term_2 = torch.log(y[..., 2])
    final_shift = (term_0 - term_2) / (2 * (term_0 + term_2 - 2 * term_1))
    return final_shift



def get_shifts_from_NCC(cc):
    x_vec = [-1, 0, 1]
    y_vec = [-1, 0, 1]
    fitting_points_x = cc[..., 1, :]
    fitting_points_y = cc[..., :, 1]
    [c_x, b_x, a_x] = fit_polynomial(x_vec, fitting_points_x.squeeze())
    [c_y, b_y, a_y] = fit_polynomial(y_vec, fitting_points_y.squeeze())
    delta_shiftx = -b_x / (2 * a_x)
    delta_shifty = -b_y / (2 * a_y)
    CC_peak_value = a_x * delta_shiftx ** 2 + b_x * delta_shiftx + c_x
    if delta_shiftx.dim() == 0:
        delta_shiftx = delta_shiftx.unsqueeze(0)
        delta_shifty = delta_shifty.unsqueeze(0)
        CC_peak_value = CC_peak_value.unsqueeze(0)
    return delta_shiftx, delta_shifty, CC_peak_value



def get_shifts_from_NCC_gaussian(cc):
    x_vec = [-1, 0, 1]
    y_vec = [-1, 0, 1]
    fitting_points_x = cc[..., 1, :]
    fitting_points_y = cc[..., :, 1]
    delta_shiftx = fit_gaussian(x_vec, fitting_points_x.squeeze())
    delta_shifty = fit_gaussian(y_vec, fitting_points_y.squeeze())
    if delta_shiftx.dim() == 0:
        delta_shiftx = delta_shiftx.unsqueeze(0)
        delta_shifty = delta_shifty.unsqueeze(0)

    return delta_shiftx, delta_shifty


# def find_gaussian_peak(cc):
#     # Define the Gaussian function
#     def gauss2d(x, y, amp, x0, y0, a, b, c):
#         inner = a * (x - x0) ** 2
#         inner += 2 * b * (x - x0) ** 2 * (y - y0) ** 2
#         inner += c * (y - y0) ** 2
#         return amp * np.exp(-inner)
#
#     x_coords = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
#     y_coords = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
#     x_flat = x_coords.flatten()
#     y_flat = y_coords.flatten()
#     function_values = cc.flatten()
#
#     # torch.linalg.lstsq
#
#     return delta_shift_x, delta_shift_y, CC_peak_value


def normalized_cc_wrapper_with_noise(A, B, corr_size=3, SNR=40):
    M1 = A.clone()
    M2 = B.clone()
    M1_std = M1.std()
    M2_std = M2.std()
    sigma_noise_M1 = M1_std / SNR  # snr = std_signal / std_noise
    sigma_noise_M2 = M2_std / SNR  # snr = std_signal / std_noise
    M1 += sigma_noise_M1 * torch.randn_like(M1)
    M2 += sigma_noise_M2 * torch.randn_like(M2)  # should we use same sigma???
    ncc = normalized_cc(M1.unsqueeze(0).unsqueeze(0), M2.unsqueeze(0).unsqueeze(0), corr_size)


    temp_m1 = M1.repeat([5, 1, 1])
    temp_m2 = M2.repeat([5, 1, 1])
    ncc1 = normalized_cc(temp_m1.unsqueeze(0).unsqueeze(0), temp_m2.unsqueeze(0).unsqueeze(0), corr_size)



    return ncc


def batch_normalized_cc_wrapper_with_noise(A, B, corr_size=3, SNR=40):
    M1 = A.clone()
    M2 = B.clone()

    M1_std = M1.std([1, 2])
    M2_std = M2.std([1, 2])
    sigma_noise_M1 = M1_std / SNR  # snr = std_signal / std_noise
    sigma_noise_M2 = M2_std / SNR  # snr = std_signal / std_noise
    M1 += sigma_noise_M1.unsqueeze(1).unsqueeze(1) * torch.randn_like(M1)
    M2 += sigma_noise_M2.unsqueeze(1).unsqueeze(1) * torch.randn_like(M2)  # should we use same sigma???
    ncc = normalized_cc(M1.unsqueeze(0).unsqueeze(0), M2.unsqueeze(0).unsqueeze(0), corr_size)

    return ncc


def normalized_cc_wrapper(A, B, corr_size=3):
    ncc = normalized_cc(A.unsqueeze(0).unsqueeze(0), B.unsqueeze(0).unsqueeze(0), corr_size)
    return ncc


def fft_normalized_cc_wrapper(A, B, corr_size=3):
    ncc = get_Normalized_Cross_Correlation_FFTImplementation_torch(A, B, corr_size)
    return ncc


def fix_corrupted_signal_ncc_AB_BA_gaussian(input_tensor_A=None, input_tensor_B=None, corr_size=3):
    shifts_h_AB, shifts_w_AB = fix_corrupted_signal_ncc_AB_using_gaussian(input_tensor_A, input_tensor_B, corr_size)
    shifts_h_BA, shifts_w_BA = fix_corrupted_signal_ncc_AB_using_gaussian(input_tensor_B, input_tensor_A, corr_size)

    shifts_w = (shifts_w_AB - shifts_w_BA) / 2
    shifts_h = (shifts_h_AB - shifts_h_BA) / 2

    # shifts_w = shifts_w.squeeze().cpu().numpy()
    # shifts_h = shifts_h.squeeze().cpu().numpy()
    # plt.figure()
    # plt.plot(shifts_w[:20])
    # plt.plot(shifts_h[:20])
    # legend(['shift_x', 'shift_y'])
    # plt.show()
    # ### Sound outputs to earphones: ###
    # wavfile.write(f"/home/yoav/rdnd/speckle_sounds_{prompt}.wav", 22050, shifts_w)
    return shifts_h, shifts_w


def fix_corrupted_signal_ncc_AB_BA(input_tensor_A=None, input_tensor_B=None, corr_size=3, device=0):
    shifts_h_AB, shifts_w_AB = fix_corrupted_signal_ncc_AB(input_tensor_A, input_tensor_B, corr_size, device)
    shifts_h_BA, shifts_w_BA = fix_corrupted_signal_ncc_AB(input_tensor_B, input_tensor_A, corr_size, device)

    shifts_w = (shifts_w_AB - shifts_w_BA) / 2
    shifts_h = (shifts_h_AB - shifts_h_BA) / 2

    # shifts_w = shifts_w.squeeze().cpu().numpy()
    # shifts_h = shifts_h.squeeze().cpu().numpy()
    # plt.figure()
    # plt.plot(shifts_w[:20])
    # plt.plot(shifts_h[:20])
    # legend(['shift_x', 'shift_y'])
    # plt.show()
    # ### Sound outputs to earphones: ###
    # wavfile.write(f"/home/yoav/rdnd/speckle_sounds_{prompt}.wav", 22050, shifts_w)
    return shifts_h, shifts_w


def fix_corrupted_signal_ncc_AB(input_tensor_A=None, input_tensor_B=None, corr_size=3, device=0):
    input_tensor_A = input_tensor_A.to(device)
    input_tensor_B = input_tensor_B.to(device)

    # if input_tensor.dim() == 3:
    #     input_tensor = input_tensor.unsqueeze(1)

    AB_ncc = normalized_cc_wrapper(input_tensor_A, input_tensor_B, corr_size)
    shifts_w_AB, shifts_h_AB, cc_AB = get_shifts_from_NCC(AB_ncc)

    # imshow_torch(input_tensor_A[0])
    # imshow_torch(input_tensor_B[0])

    shifts_w_AB = shifts_w_AB.squeeze().cpu().numpy()
    shifts_h_AB = shifts_h_AB.squeeze().cpu().numpy()

    return shifts_h_AB, shifts_w_AB


def fix_corrupted_signal_ncc_AB_using_gaussian(input_tensor_A=None, input_tensor_B=None, corr_size=3, device=0):
    input_tensor_A = input_tensor_A.to(device)
    input_tensor_B = input_tensor_B.to(device)

    # if input_tensor.dim() == 3:
    #     input_tensor = input_tensor.unsqueeze(1)

    AB_ncc = normalized_cc_wrapper(input_tensor_A, input_tensor_B, corr_size)
    shifts_w_AB, shifts_h_AB = get_shifts_from_NCC_gaussian(AB_ncc)

    shifts_w_AB = shifts_w_AB.squeeze().cpu().numpy()
    shifts_h_AB = shifts_h_AB.squeeze().cpu().numpy()

    return shifts_h_AB, shifts_w_AB


def get_repeated_shift_wrapper_gaussian(A, B, corr_size=3, repetitions=2, device=0):
    A_copy = A.clone()
    B_copy = B.clone()
    shift_sum = 0
    for i in range(repetitions):
        delta_shifty_AB_BA, delta_shiftx_AB_BA = fix_corrupted_signal_ncc_AB_BA_gaussian(A_copy, B_copy, corr_size)
        delta_shifty_AB_BA = torch.tensor(delta_shifty_AB_BA).to(device)
        delta_shiftx_AB_BA = torch.tensor(delta_shiftx_AB_BA).to(device)

        shift_matrix_AB = torch.cat([delta_shiftx_AB_BA.unsqueeze(1), delta_shifty_AB_BA.unsqueeze(1)], axis=1)

        shift_sum += shift_matrix_AB
        # print(shift_matrix_AB)
        A_copy, _ = _shift_matrix_subpixel_fft_batch(A_copy, -delta_shifty_AB_BA, -delta_shiftx_AB_BA)
        # for j in range(A_copy.shape[0]):
        #     # A_copy[j] = torch.tensor(shift_matrix_subpixel(A_copy[j].cpu().numpy(), delta_shiftx_AB_BA[j].cpu().numpy(), delta_shifty_AB_BA[j].cpu().numpy())).to(B.device).squeeze()
        #     A_copy[j] = torch.tensor(shift_matrix_subpixel(A_copy[j].cpu().numpy(), -delta_shiftx_AB_BA[j].cpu().numpy(), -delta_shifty_AB_BA[j].cpu().numpy())).to(B.device).squeeze()


    return shift_sum[:, 1], shift_sum[:, 0]


def get_repeated_shift_wrapper_poly(A, B, corr_size=3, repetitions=2, device=0):
    A_copy = A.clone()
    B_copy = B.clone()
    shift_sum = 0
    for i in range(repetitions):
        delta_shifty_AB_BA, delta_shiftx_AB_BA = fix_corrupted_signal_ncc_AB_BA(A_copy, B_copy, corr_size)
        delta_shifty_AB_BA = torch.tensor(delta_shifty_AB_BA).to(device)
        delta_shiftx_AB_BA = torch.tensor(delta_shiftx_AB_BA).to(device)

        shift_matrix_AB = torch.cat([delta_shiftx_AB_BA.unsqueeze(1), delta_shifty_AB_BA.unsqueeze(1)], axis=1)

        shift_sum += shift_matrix_AB
        # print(shift_matrix_AB)
        A_copy, _ = _shift_matrix_subpixel_fft_batch(A_copy, -delta_shifty_AB_BA, -delta_shiftx_AB_BA)
        # for j in range(A_copy.shape[0]):
        #     # A_copy[j] = torch.tensor(shift_matrix_subpixel(A_copy[j].cpu().numpy(), delta_shiftx_AB_BA[j].cpu().numpy(), delta_shifty_AB_BA[j].cpu().numpy())).to(B.device).squeeze()
        #     A_copy[j] = torch.tensor(shift_matrix_subpixel(A_copy[j].cpu().numpy(), -delta_shiftx_AB_BA[j].cpu().numpy(), -delta_shifty_AB_BA[j].cpu().numpy())).to(B.device).squeeze()


    return shift_sum[:, 1], shift_sum[:, 0]


def yoavs_method_for_sequence(input_tensor, k=5, movement_x=0.1, movement_y=0):
    shifts = []
    # for speckle_id, (A, B) in enumerate(zip(input_tensor[1:], input_tensor[0:-1])):
    for speckle_id, (A, B) in enumerate(zip(input_tensor[0:-1], input_tensor[1:])):
        if speckle_id % 1000 == 0:
            print(f"{speckle_id}/{input_tensor.shape[0]-1}")
        shift_i = batch_generate_intermediate_speckles_and_average_estimates(A.unsqueeze(0), B.unsqueeze(0), k=k, movement_x=movement_x, movement_y=movement_y)
        shifts.append(shift_i)
    return torch.cat([s.unsqueeze(0) for s in shifts], axis=0)


def generate_intermediate_speckles_and_average_estimates_no_noise(A, B, k=5, movement_x=0.1, movement_y=0):
    """
        for both A and B i create k speckles shifted with some new movement.
        then for each one of the k speckles generated from A i compute the ncc between A and C and between C and A.
        thus creating an array of size (k, 2) such that entry (i, j) is the abba correlation
        of speckles A and C_i(one of the k speckles) on axis j(x or y).
        same is done for B(but opposite direction).
        now we have 2 of those arrays such that if i compute for each m(0 to k-1) (arr_a[m] - arr_b[m]) / 2
        i should get shift_AB since arr_a[m] is shift_AC and and arr_b[m] is shift_CB.
        then we have k shifts that are allegedly shift_AB so we average them all and get one final shift
    Args:
        A:
        B:
        k:
        movement_x:
        movement_y:
        SNR:

    Returns: shifts for A and B in both axis

    """
    shift_layer = Shift_Layer_Torch()
    movements_inner = my_linspace(movement_x - 0.1, movement_x + 0.1, k)
    shift_matrix_AC = torch.zeros(k, 2)
    shift_matrix_CB = torch.zeros(k, 2)
    for shift_size_counter_1, movement_x_inner in enumerate(movements_inner):
        C, shiftx_c, shifty_c = shift_layer.forward(A, np.array([-movement_x_inner]).astype(float32), np.array([-movement_y]).astype(float32))

        cc_AC = normalized_cc_wrapper(A, C, corr_size=3)
        cc_CA = normalized_cc_wrapper(C, A, corr_size=3)
        delta_shiftx_AC, delta_shifty_AC, CC_peak_value_AC = get_shifts_from_NCC(cc_AC)
        delta_shiftx_CA, delta_shifty_CA, CC_peak_value_CA = get_shifts_from_NCC(cc_CA)
        delta_shiftx_AC_CA = (delta_shiftx_AC - delta_shiftx_CA) / 2
        delta_shifty_AC_CA = (delta_shifty_AC - delta_shifty_CA) / 2
        shift_matrix_AC[shift_size_counter_1] = torch.tensor([delta_shiftx_AC_CA, delta_shifty_AC_CA])

        cc_BC = normalized_cc_wrapper(B, C, corr_size=3)
        cc_CB = normalized_cc_wrapper(C, B, corr_size=3)
        delta_shiftx_BC, delta_shifty_BC, CC_peak_value_BC = get_shifts_from_NCC(cc_BC)
        delta_shiftx_CB, delta_shifty_CB, CC_peak_value_CB = get_shifts_from_NCC(cc_CB)
        delta_shiftx_CB_BC = (delta_shiftx_CB - delta_shiftx_BC) / 2
        delta_shifty_CB_BC = (delta_shifty_CB - delta_shifty_BC) / 2
        shift_matrix_CB[shift_size_counter_1] = torch.tensor([delta_shiftx_CB_BC, delta_shifty_CB_BC])

    # shift_matrix_AB = (shift_matrix_AC + shift_matrix_CB) / 2
    shift_matrix_AB = (shift_matrix_AC + shift_matrix_CB)
    average_shifts = shift_matrix_AB.mean(0)
    return average_shifts


def batch_generate_intermediate_speckles_and_average_estimates(A, B, k=5, movement_x=0.001, movement_y=0, SNR=40):
    """
        for both A and B i create k speckles shifted with some new movement.
        then for each one of the k speckles generated from A i compute the ncc between A and C and between C and A.
        thus creating an array of size (k, 2) such that entry (i, j) is the abba correlation
        of speckles A and C_i(one of the k speckles) on axis j(x or y).
        same is done for B(but opposite direction).
        now we have 2 of those arrays such that if i compute for each m(0 to k-1) (arr_a[m] - arr_b[m]) / 2
        i should get shift_AB since arr_a[m] is shift_AC and and arr_b[m] is shift_CB.
        then we have k shifts that are allegedly shift_AB so we average them all and get one final shift
    Args:
        A:
        B:
        k:
        movement_x:
        movement_y:
        SNR:

    Returns: shifts for A and B in both axis

    """
    shift_layer = Shift_Layer_Torch()
    movements_inner = my_linspace(movement_x - 0.0001, movement_x + 0.0001, k)
    k = movements_inner.shape[0]  # update in case it is off by 1
    # Cs = torch.zeros_like(A).repeat([k, 1, 1])
    # for shift_size_counter_1, movement_x_inner in enumerate(movements_inner):
    #     C, shiftx_c, shifty_c = shift_layer.forward(A, np.array([-movement_x_inner]).astype(float32), np.array([-movement_y]).astype(float32))
    #     Cs[shift_size_counter_1] = C
    # generate intermediate estimates
    Cs, shiftx_c, shifty_c = shift_layer.forward(A, np.array([-movements_inner]).astype(float32), np.array([-movement_y]).astype(float32))
    Cs = Cs.squeeze()

    average_shifts = average_estimates(A, B, Cs, k)
    return average_shifts


def average_shifts_across_speckle_window(input_tensor, window_size=9):
    """
        computes the pair wise shift matrix using some algorithm(here we use ncc) and instead of taking the direct calculated shift between i and j
        computes the (weighted) average of all the two step paths: meaning average over all k, shift(i, k) + shift(k, j) which should equal shift(i, j)..
        mathematically, shift[i, i+1] = average(shift[i, k] + shift[k, i+1] for some window of k values near i)
        Args:
            input_tensor: all the speckle frames
            window_size: number of frames to average

        Returns: approximated shifts between following speckles
        """
    # shifts = []
    shifts = []
    # for each speckle pair
    for speckle_id, (A, B) in enumerate(zip(input_tensor[0:-1], input_tensor[1:])):
        # if speckle_id % 1000 == 0:
        #     print(f"{speckle_id}/{input_tensor.shape[0]-1}")

        # get minimum and maximum valid index according to array size and window size
        min_index = max(speckle_id-window_size//2, 0)
        max_index = min(speckle_id+window_size//2, input_tensor.shape[0])
        # extract window of speckles
        Cs = input_tensor[min_index:max_index]
        # actual averaging and ncc computation
        shift_i = average_estimates(A.unsqueeze(0), B.unsqueeze(0), Cs)
        # shift_i = average_estimates(A.unsqueeze(0), B.unsqueeze(0), Cs)
        # append to result list
        shifts.append(shift_i)

    shifts = shifts[window_size : -window_size]
    shifts = torch.cat([s.unsqueeze(0) for s in shifts], dim=0)
    # for i in range(shifts.shape[1]):
    #     plt.plot(shifts[:, i, 0].cpu().detach().numpy())
    # plt.show()
    # return result as tensor
    return torch.cat([s.unsqueeze(0) for s in shifts], axis=0)


def average_repeated_shifts_across_speckle_window(input_tensor, window_size=9):
    """
        computes the pair wise shift matrix using some algorithm(here we use ncc) and instead of taking the direct calculated shift between i and j
        computes the (weighted) average of all the two step paths: meaning average over all k, shift(i, k) + shift(k, j) which should equal shift(i, j)..
        mathematically, shift[i, i+1] = average(shift[i, k] + shift[k, i+1] for some window of k values near i)
        Args:
            input_tensor: all the speckle frames
            window_size: number of frames to average

        Returns: approximated shifts between following speckles
        """
    shifts = []
    # for each speckle pair
    for speckle_id, (A, B) in enumerate(zip(input_tensor[0:-1], input_tensor[1:])):
        # if speckle_id % 1000 == 0:
        #     print(f"{speckle_id}/{input_tensor.shape[0]-1}")

        # get minimum and maximum valid index according to array size and window size
        min_index = max(speckle_id-window_size//2, 0)
        max_index = min(speckle_id+window_size//2, input_tensor.shape[0])
        # extract window of speckles
        Cs = input_tensor[min_index:max_index]
        # actual averaging and ncc computation
        shift_i = average_estimates_with_repeated_shifts(A.unsqueeze(0), B.unsqueeze(0), Cs)
        # append to result list
        shifts.append(shift_i)

    # return result as tensor
    return torch.cat([s.unsqueeze(0) for s in shifts], axis=0)


def average_random_shifts_across_speckle_window(input_tensor, window_size=9):
    """
        computes the pair wise shift matrix using some algorithm(here we use ncc) and instead of taking the direct calculated shift between i and j
        computes the (weighted) average of all the two step paths: meaning average over all k, shift(i, k) + shift(k, j) which should equal shift(i, j)..
        mathematically, shift[i, i+1] = average(shift[i, k] + shift[k, i+1] for some window of k values near i)
        Args:
            input_tensor: all the speckle frames
            window_size: number of frames to average

        Returns: approximated shifts between following speckles
        """
    shifts = []
    to_shift = [(-i)**2 * 0.0001 for i in range(10)]
    # for each speckle pair
    for speckle_id, (A, B) in enumerate(zip(input_tensor[0:-1], input_tensor[1:])):
        # if speckle_id % 1000 == 0:
        #     print(f"{speckle_id}/{input_tensor.shape[0]-1}")

        # get minimum and maximum valid index according to array size and window size
        min_index = max(speckle_id-window_size//2, 0)
        max_index = min(speckle_id+window_size//2, input_tensor.shape[0])
        # extract window of speckles
        # Cs = input_tensor[min_index:max_index]
        Cs = [shift_matrix_subpixel(A.cpu().detach(), i, 0) for i in to_shift]
        Cs = torch.cat([A.unsqueeze(0), B.unsqueeze(0), torch.tensor(Cs).squeeze().to(A.device)], dim=0)
        Cs = torch.cat([add_noise_to_speckle_with_given_snr(c, SNR=10).unsqueeze(0) for c in Cs])
        # actual averaging and ncc computation
        shift_i = average_estimates(A.unsqueeze(0), B.unsqueeze(0), Cs)
        # append to result list
        shifts.append(shift_i)

    # return result as tensor
    return torch.cat([s.unsqueeze(0) for s in shifts], axis=0)


def moving_average(a, n=16):
    ret = a.cumsum(0)
    ret = ret[n:, :] - ret[:-n, :]
    return ret[n - 1:] / n


def compute_ncc_on_rows(input_tensor):
    shifts = []
    # for each speckle pair
    for speckle_id, (A, B) in enumerate(zip(input_tensor[0:-1], input_tensor[1:])):
        # if speckle_id % 1000 == 0:
        #     print(f"{speckle_id}/{input_tensor.shape[0]-1}")

        # average A and B across rows
        A_averaged = A.mean(0).repeat(A.shape[0], 1).unsqueeze(0)
        B_averaged = B.mean(0).repeat(B.shape[0], 1).unsqueeze(0)

        # use rolling average instead to get N different rows
        A_averaged = moving_average(A, n=5).unsqueeze(0)
        B_averaged = moving_average(B, n=5).unsqueeze(0)

        # imshow_torch(A)
        # imshow_torch(A_averaged)
        # imshow_torch(B)
        # imshow_torch(B_averaged)
        # compute shifts between A and B
        cc_AB = normalized_cc_wrapper(A_averaged, B_averaged, corr_size=3)
        cc_BA = normalized_cc_wrapper(B_averaged, A_averaged, corr_size=3)
        delta_shiftx_AB, delta_shifty_AB, CC_peak_value_AB = get_shifts_from_NCC(cc_AB)
        delta_shiftx_BA, delta_shifty_BA, CC_peak_value_BA = get_shifts_from_NCC(cc_BA)
        # extract acca shifts
        delta_shiftx_AB_BA = (delta_shiftx_AB - delta_shiftx_BA) / 2
        # delta_shifty_AB_BA = (delta_shifty_AB - delta_shifty_BA) / 2
        shifts.append(delta_shiftx_AB_BA)

    # return result as tensor
    return torch.cat([s.unsqueeze(0) for s in shifts], axis=0)


def speckle_averaging_accros_window(input_tensor, window_size=9, estimation_function=None):
    offset = (window_size - 1) // 2
    averaged_speckles = torch.zeros_like(input_tensor)
    t, h, w = input_tensor.shape
    for index, reference_speckle in enumerate(input_tensor):
        ## for simplicity at the begininng ignore first and last elements
        # if index < offset or index >= (t - offset):
        #     averaged_speckles[index] = reference_speckle
        #     continue

        min_range = max(0, index - offset)
        max_range = min(t, index + offset + 1)

        warped_window_speckles = 0
        for sub_index in range(min_range, max_range):
            _, shift = estimation_function(torch.cat([reference_speckle.unsqueeze(0), input_tensor[sub_index:sub_index+1]]))
            warped = shift_matrix_subpixel(input_tensor[sub_index:sub_index+1].cpu().numpy(), shift, 0)
            warped_window_speckles += warped.squeeze()
            # warp sub index according to shift
        # average all warped sppeckles
        # insert to averaged tensor
        averaged_speckles[index] = torch.tensor(warped_window_speckles).to(input_tensor.device) / (sub_index+1)

    return averaged_speckles


def fast_speckle_averaging_accros_window(input_tensor, window_size=9, estimation_function=None):
    """

    Args:
        input_tensor:
        window_size:
        estimation_function:

    Returns:

    """
    averaged_speckles = torch.zeros_like(input_tensor)
    t, h, w = input_tensor.shape
    first_shift = 0
    second_shift = 0
    for index, reference_speckle in enumerate(input_tensor):

        warped_window_speckles = 0
        if index == 0 or index == input_tensor.shape[0] - 1:  # could use a differnet window of 3 but why bother, these are only 2 values
            averaged_speckles[index] = reference_speckle

        elif index == 1:
            _, first_shift = estimation_function(torch.cat([reference_speckle.unsqueeze(0), input_tensor[0:1]]))
            warped = shift_matrix_subpixel(input_tensor[0:1].cpu().numpy(), first_shift, 0)
            warped_window_speckles += warped.squeeze()

            _, second_shift = estimation_function(torch.cat([reference_speckle.unsqueeze(0), input_tensor[2:3]]))
            warped = shift_matrix_subpixel(input_tensor[2:3].cpu().numpy(), second_shift, 0)
            warped_window_speckles += warped.squeeze()

            # warped_window_speckles += reference_speckle.squeeze().cpu().numpy()
            averaged_speckles[index] = torch.tensor(warped_window_speckles).to(input_tensor.device) / 2

        else:
            first_shift = -second_shift
            warped = shift_matrix_subpixel(input_tensor[index-1:index].cpu().numpy(), first_shift, 0)
            warped_window_speckles += warped.squeeze()

            _, second_shift = estimation_function(torch.cat([reference_speckle.unsqueeze(0), input_tensor[index+1:index+2]]))
            warped = shift_matrix_subpixel(input_tensor[index+1:index+2].cpu().numpy(), second_shift, 0)
            warped_window_speckles += warped.squeeze()

            # warped_window_speckles += reference_speckle.squeeze().cpu().numpy()
            averaged_speckles[index] = torch.tensor(warped_window_speckles).to(input_tensor.device) / 2

    return averaged_speckles


def reallyyyyy_fast_speckle_averaging_accros_window_general(input_tensor, window_size=9, estimation_function=None, rep=4):
    """

    Args:
        input_tensor:
        window_size:
        estimation_function:

    Returns:

    """
    averaged_speckles = torch.zeros_like(input_tensor)
    t, h, w = input_tensor.shape
    offset = (window_size - 1) // 2

    for i in range(0, offset):
        averaged_speckles[i] = input_tensor[i]
        averaged_speckles[t-i-1] = input_tensor[t-i-1]


    # todo: automate this using a base index and window size
    estimations_forwards = []
    estimations_backwards = []
    base_index = offset  # starts and ends at [base_index, -base_index]
    for i in range(1, offset+1):
        # first shifts will be calculated from base index to base_index + offset.
        # we only need to consider indecis that are in range [base_index, -base_index]

        # shifts from any i smaller than reference to the reference
        previous_to_reference_shifts_y, previous_to_reference_shifts_x = estimation_function(
                                                input_tensor[base_index-i:-(base_index+i)],
                                                input_tensor[base_index:-base_index],
                                                corr_size=3, repetitions=rep)
        # shifts from any i larger than reference to the reference
        if base_index != i:
                next_to_reference_shifts_y, next_to_reference_shifts_x = estimation_function(
                                                    input_tensor[base_index+i:-(base_index-i)],
                                                    input_tensor[base_index:-base_index],
                                                    corr_size=3, repetitions=rep)
        else:
            next_to_reference_shifts_y, next_to_reference_shifts_x = estimation_function(
                input_tensor[base_index + i:],
                input_tensor[base_index:-base_index],
                corr_size=3, repetitions=rep)


        estimations_forwards.append([previous_to_reference_shifts_x, previous_to_reference_shifts_y, i])
        estimations_backwards.append([next_to_reference_shifts_x, next_to_reference_shifts_y, i])

    warped_speckles = torch.zeros(input_tensor.shape[0] - 2*offset, input_tensor.shape[1], input_tensor.shape[2]).to(input_tensor.device)
    for est1_and_offset, est2_and_offset in zip(estimations_forwards, estimations_backwards):
        est1_x, est1_y, i = est1_and_offset
        est2_x, est2_y, j = est2_and_offset
        assert i == j

        warped_1, _ = _shift_matrix_subpixel_fft_batch(input_tensor[base_index-i:-(base_index+i)], -(est1_y).to(input_tensor.device), -(est1_x).to(input_tensor.device))
        if base_index != j:
            warped_2, _ = _shift_matrix_subpixel_fft_batch(input_tensor[base_index+j:-(base_index-j)], -(est2_y).to(input_tensor.device), -(est2_x).to(input_tensor.device))
        else:
            warped_2, _ = _shift_matrix_subpixel_fft_batch(input_tensor[base_index + j:], -(est2_y).to(input_tensor.device), -(est2_x).to(input_tensor.device))

        warped_speckles += warped_1
        warped_speckles += warped_2

    averaged_speckles[base_index:-base_index] = warped_speckles / (2 * offset)
    return averaged_speckles


def reallyyyyy_fast_speckle_averaging_accros_window_basic(input_tensor, estimation_function=None, rep=5):
    """

    Args:
        input_tensor:
        window_size:
        estimation_function:

    Returns:

    """
    averaged_speckles = torch.zeros_like(input_tensor)
    t, h, w = input_tensor.shape

    averaged_speckles[0] = input_tensor[0]
    averaged_speckles[-1] = input_tensor[-1]

    # add estimate from previous speckle
    try:
        estimates_y, estimates_x = estimation_function(input_tensor[0:-2], input_tensor[1:-1], corr_size=3, repetitions=rep)
    except:
        estimates_y, estimates_x = estimation_function(input_tensor[0:-2], input_tensor[1:-1], corr_size=3)
    # estimates_y, estimates_x = estimation_function(input_tensor[0:-1], corr_size=3)
    warped, _ = _shift_matrix_subpixel_fft_batch(input_tensor[0:-2], -torch.tensor(estimates_y).to(input_tensor.device), -torch.tensor(estimates_x).to(input_tensor.device))
    averaged_speckles[1:-1] += warped / 2

    # add estimate from next speckle
    # estimates_y, estimates_x = estimation_function(input_tensor[1:].flip(0), corr_size=3)
    # estimates_x = np.flip(estimates_x).copy()
    # estimates_y = np.flip(estimates_y).copy()
    try:
        estimates_y, estimates_x = estimation_function(input_tensor[2:], input_tensor[1:-1], corr_size=3, repetitions=rep)
    except:
        estimates_y, estimates_x = estimation_function(input_tensor[2:], input_tensor[1:-1], corr_size=3)

    warped, _ = _shift_matrix_subpixel_fft_batch(input_tensor[2:], -torch.tensor(estimates_y).to(input_tensor.device), -torch.tensor(estimates_x).to(input_tensor.device))
    averaged_speckles[1:-1] += warped / 2

    return averaged_speckles


def get_avg_reference_given_shifts(video, shifts):
    sx, sy = shifts
    # clean = torch.zeros((video.shape[1:]))
    # todo: batch this
    warped, _ = _shift_matrix_subpixel_fft_batch_with_channels(video, -sy.to(video.device), -sx.to(video.device))
    # for _sx, _sy, frame in zip(sx, sy, video):
    #     warped, _ = _shift_matrix_subpixel_fft_batch(frame, _sy.to(video.device), _sx.to(video.device))
    #     clean += warped
    # return clean / video.shape[0]
    return warped.mean(0)



def reallyyyyy_fast_speckle_averaging_accros_window_with_different_video_to_estimate_shifts_on(input_tensor, video_to_est_shift=None, estimation_function=None):
    """

    Args:
        input_tensor:
        window_size:
        estimation_function:

    Returns:

    """
    averaged_speckles = torch.zeros_like(input_tensor)
    t, h, w = input_tensor.shape

    averaged_speckles[0] = input_tensor[0]
    averaged_speckles[-1] = input_tensor[-1]

    # add estimate from previous speckle
    sy, sx, cc = estimation_function(video_to_est_shift[0:-2].unsqueeze(0).unsqueeze(2), video_to_est_shift[1:-1].unsqueeze(0).unsqueeze(2))
    warped, _ = _shift_matrix_subpixel_fft_batch(input_tensor[0:-2], -sy.to(input_tensor.device), -sx.to(input_tensor.device))
    averaged_speckles[1:-1] += warped / 2

    # add estimate from next speckle
    sy, sx, cc = estimation_function(video_to_est_shift[1:-1].unsqueeze(0).unsqueeze(2), video_to_est_shift[0:-2].unsqueeze(0).unsqueeze(2))
    warped, _ = _shift_matrix_subpixel_fft_batch(input_tensor[2:], -sy.to(input_tensor.device), -sx.to(input_tensor.device))
    averaged_speckles[1:-1] += warped / 2

    return averaged_speckles


def average_estimates(A, B, Cs, k=None):
    """

    Args:
        A: first speckle
        B: second speckle
        Cs: window of speckles
        k: size of window

    Returns : approximate shift between A and B using Cs:

            shift(A, B) = average(shift(A, c), shift(c, B) for every c in Cs)

    """
    if k is None:
        k = Cs.shape[0]

    # compute shifts between A and all Cs
    cc_AC = normalized_cc_wrapper(A.repeat([k, 1, 1]), Cs, corr_size=3)
    cc_CA = normalized_cc_wrapper(Cs, A.repeat([k, 1, 1]), corr_size=3)
    delta_shiftx_AC, delta_shifty_AC, CC_peak_value_AC = get_shifts_from_NCC(cc_AC)
    delta_shiftx_CA, delta_shifty_CA, CC_peak_value_CA = get_shifts_from_NCC(cc_CA)
    # extract acca shifts
    delta_shiftx_AC_CA = (delta_shiftx_AC - delta_shiftx_CA) / 2
    delta_shifty_AC_CA = (delta_shifty_AC - delta_shifty_CA) / 2
    shift_matrix_AC = torch.cat([delta_shiftx_AC_CA.unsqueeze(1), delta_shifty_AC_CA.unsqueeze(1)], axis=1)

    # compute shifts between all Cs and B
    cc_BC = normalized_cc_wrapper(B.repeat([k, 1, 1]), Cs, corr_size=3)
    cc_CB = normalized_cc_wrapper(Cs, B.repeat([k, 1, 1]), corr_size=3)
    delta_shiftx_BC, delta_shifty_BC, CC_peak_value_BC = get_shifts_from_NCC(cc_BC)
    delta_shiftx_CB, delta_shifty_CB, CC_peak_value_CB = get_shifts_from_NCC(cc_CB)
    # extract cbbc shifts
    delta_shiftx_CB_BC = (delta_shiftx_CB - delta_shiftx_BC) / 2
    delta_shifty_CB_BC = (delta_shifty_CB - delta_shifty_BC) / 2
    shift_matrix_CB = torch.cat([delta_shiftx_CB_BC.unsqueeze(1), delta_shifty_CB_BC.unsqueeze(1)], axis=1)

    # plt.plot(shift_matrix_AC[:, 0].cpu().detach().numpy() - movements_inner)
    # # plt.plot(shift_matrix_CB[:, 0].cpu().detach().numpy() - movements_inner)
    # plt.show()
    # add shifts such that for each c shift(A, B) = shift(A, c) + shift(c, B)
    """
    theoretical analysis:
    say my signal is [s0, s0 + s1, s0 + s1 + s2 and so on] such that s_0 is A and s_o + s_1 is B.(remember this is cumsum)
    AC shifted array will be : [0,      s1,   s1 + s2,      s1 + s2 + s3  and so on]
    CB shifted array will be : [s1,    0,      -s2,        -(s2 + s3)]
    such that when i add them up i should get : [s1, s1, s1, s1 ...], and overall the shift between A and B...
    """

    shift_matrix_AB = (shift_matrix_AC + shift_matrix_CB)
    # average all shifts
    # average_shifts = shift_matrix_AB
    average_shifts = shift_matrix_AB.mean(0)

    # todo: can experiment with a different way of averaging for example using a gaussian kernel as following:
    # weights = create_1d_gauss_weights(n=shift_matrix_AB.shape[0], sigma=1.5).to(device)
    # average_shifts = (weights.unsqueeze(1) * shift_matrix_AB)
    # average_shifts = average_shifts.sum(0)

    return average_shifts



def average_estimates_with_repeated_shifts(A, B, Cs, k=None):
    """

    Args:
        A: first speckle
        B: second speckle
        Cs: window of speckles
        k: size of window

    Returns : approximate shift between A and B using Cs:

            shift(A, B) = average(shift(A, c), shift(c, B) for every c in Cs)

    """
    if k is None:
        k = Cs.shape[0]

    # compute shifts between A and all Cs
    shift_matrix_AC = get_repeated_shift_wrapper(A.repeat([k, 1, 1]), Cs, corr_size=3, repetitions=2)

    # compute shifts between all Cs and B
    shift_matrix_CB = get_repeated_shift_wrapper(Cs, B.repeat([k, 1, 1]), corr_size=3, repetitions=2)

    # plt.plot(shift_matrix_AC[:, 0].cpu().detach().numpy() - movements_inner)
    # # plt.plot(shift_matrix_CB[:, 0].cpu().detach().numpy() - movements_inner)
    # plt.show()
    # add shifts such that for each c shift(A, B) = shift(A, c) + shift(c, B)
    """
    theoretical analysis:
    say my signal is [s0, s0 + s1, s0 + s1 + s2 and so on] such that s_0 is A and s_o + s_1 is B.(remember this is cumsum)
    AC shifted array will be : [0,      s1,   s1 + s2,      s1 + s2 + s3  and so on]
    CB shifted array will be : [s1,    0,      -s2,        -(s2 + s3)]
    such that when i add them up i should get : [s1, s1, s1, s1 ...], and overall the shift between A and B...
    """

    shift_matrix_AB = (shift_matrix_AC + shift_matrix_CB)
    # average all shifts
    average_shifts = shift_matrix_AB.mean(0)

    # todo: can experiment with a different way of averaging for example using a gaussian kernel as following:
    # weights = create_1d_gauss_weights(n=shift_matrix_AB.shape[0], sigma=1.5).to(device)
    # average_shifts = (weights.unsqueeze(1) * shift_matrix_AB)
    # average_shifts = average_shifts.sum(0)

    return average_shifts


# TODO: this is incorrect, thus the analysis from before is as well... fix according to function above
def generate_intermediate_speckles_and_average_estimates(A, B, k=5, movement_x=0.1, movement_y=0.0, SNR=40):
    """
        for both A and B i create k speckles shifted with some new movement.
        then for each one of the k speckles generated from A i compute the ncc between A and C and between C and A.
        thus creating an array of size (k, 2) such that entry (i, j) is the abba correlation
        of speckles A and C_i(one of the k speckles) on axis j(x or y).
        same is done for B(but opposite direction).
        now we have 2 of those arrays such that if i compute for each m(0 to k-1) (arr_a[m] - arr_b[m]) / 2
        i should get shift_AB since arr_a[m] is shift_AC and and arr_b[m] is shift_CB.
        then we have k shifts that are allegedly shift_AB so we average them all and get one final shift
    Args:
        A:
        B:
        k:
        movement_x:
        movement_y:
        SNR:

    Returns: shifts for A and B in both axis

    """
    shift_layer = Shift_Layer_Torch()
    movements_inner = my_linspace(movement_x - 0.1, movement_x + 0.1, k)
    shift_matrix_AC = torch.zeros(k, 2)
    shift_matrix_CB = torch.zeros(k, 2)
    for shift_size_counter_1, movement_x_inner in enumerate(movements_inner):
        # C, shiftx_c, shifty_c = shift_layer.forward(A, np.array([-movement_x_inner]).astype(float32), np.array([-movement_y]).astype(float32))
        C, shiftx_c, shifty_c = shift_layer.forward(A, np.array([-movement_x_inner]).astype(float32), np.array([-movement_y]).astype(float32))

        cc_AC = normalized_cc_wrapper_with_noise(A, C, corr_size=3, SNR=SNR)
        cc_CA = normalized_cc_wrapper_with_noise(C, A, corr_size=3, SNR=SNR)
        delta_shiftx_AC, delta_shifty_AC, CC_peak_value_AC = get_shifts_from_NCC(cc_AC)
        delta_shiftx_CA, delta_shifty_CA, CC_peak_value_CA = get_shifts_from_NCC(cc_CA)
        delta_shiftx_AC_CA = (delta_shiftx_AC - delta_shiftx_CA) / 2
        delta_shifty_AC_CA = (delta_shifty_AC - delta_shifty_CA) / 2
        shift_matrix_AC[shift_size_counter_1] = torch.tensor([delta_shiftx_AC_CA, delta_shifty_AC_CA])

        cc_BC = normalized_cc_wrapper_with_noise(B, C, corr_size=3, SNR=SNR)
        cc_CB = normalized_cc_wrapper_with_noise(C, B, corr_size=3, SNR=SNR)
        delta_shiftx_BC, delta_shifty_BC, CC_peak_value_BC = get_shifts_from_NCC(cc_BC)
        delta_shiftx_CB, delta_shifty_CB, CC_peak_value_CB = get_shifts_from_NCC(cc_CB)
        delta_shiftx_CB_BC = (delta_shiftx_CB - delta_shiftx_BC) / 2
        delta_shifty_CB_BC = (delta_shifty_CB - delta_shifty_BC) / 2
        shift_matrix_CB[shift_size_counter_1] = torch.tensor([delta_shiftx_CB_BC, delta_shifty_CB_BC])

    # shift_matrix_AB = (shift_matrix_AC + shift_matrix_CB) / 2
    # shift_matrix_AB = (-shift_matrix_AC - shift_matrix_CB) / 2
    # shift_matrix_AB = (shift_matrix_AC - shift_matrix_CB) / 2
    # shift_matrix_AB = (-shift_matrix_AC + shift_matrix_CB) / 2
    shift_matrix_AB = (shift_matrix_AC + shift_matrix_CB)
    average_shifts = shift_matrix_AB.mean(0)
    return average_shifts


def ncc_bias_experiment():
    ###############################################################################################################
    # START EXPERIMENT
    speckle_sizes = [5]
    img_sizes = [64]
    # movements = np.arange(0, 0.2, step=0.005)
    movements = my_linspace(0, 0.1, 10)
    # movements = np.array([0.01])
    experiment_size = 5
    max_shift = 20
    shift_layer = Shift_Layer_Torch()
    # EXPERIMENT_SIZE
    num_params = 3
    num_movement_axes = 2  # x, y
    SNR = 6.0
    k = 10  # determines how many speckles to create and compare to
    x_vec = [-1, 0, 1]
    y_vec = [-1, 0, 1]
    experiment_table_std = torch.zeros(len(speckle_sizes), len(img_sizes), len(movements), num_movement_axes)
    experiment_table_bias = torch.zeros(len(speckle_sizes), len(img_sizes), len(movements), num_movement_axes)
    experiment_table_mean = torch.zeros(len(speckle_sizes), len(img_sizes), len(movements), num_movement_axes)
    experiment_table_std_AB_BA = torch.zeros(len(speckle_sizes), len(img_sizes), len(movements), num_movement_axes)
    experiment_table_bias_AB_BA = torch.zeros(len(speckle_sizes), len(img_sizes), len(movements), num_movement_axes)
    experiment_table_mean_AB_BA = torch.zeros(len(speckle_sizes), len(img_sizes), len(movements), num_movement_axes)
    experiment_table_std_AB_AA = torch.zeros(len(speckle_sizes), len(img_sizes), len(movements), num_movement_axes)
    experiment_table_bias_AB_AA = torch.zeros(len(speckle_sizes), len(img_sizes), len(movements), num_movement_axes)
    experiment_table_mean_AB_AA = torch.zeros(len(speckle_sizes), len(img_sizes), len(movements), num_movement_axes)
    experiment_table_std_yoav = torch.zeros(len(speckle_sizes), len(img_sizes), len(movements), num_movement_axes)
    experiment_table_bias_yoav = torch.zeros(len(speckle_sizes), len(img_sizes), len(movements), num_movement_axes)
    experiment_table_mean_yoav = torch.zeros(len(speckle_sizes), len(img_sizes), len(movements), num_movement_axes)
    experiment_table_params = torch.zeros(len(speckle_sizes), len(img_sizes), len(movements), num_params)
    for speckle_size_counter, speckle_size in enumerate(speckle_sizes):
        for ROI_size_counter, img_size in enumerate(img_sizes):
            for shift_size_counter, movement_x in enumerate(movements):
                # movement_x = 0.15
                movement_y = 0.0
                ###############################################################################################################
                shiftx_list = []
                shifty_list = []
                shiftx_list_AB_BA = []
                shifty_list_AB_BA = []
                shiftx_list_AB_AA = []
                shifty_list_AB_AA = []
                shiftx_list_yoav = []
                shifty_list_yoav = []
                shifted_speckles = torch.zeros((experiment_size, img_size, img_size))
                for speckle_family_counter in range(experiment_size):  # same movement for different speckles
                    print('current iteration: ' + str(speckle_family_counter))
                    
                    ### Create speckle pattern for this run: ###
                    A, speckle_pattern_field1, speckle_pattern_field2, total_beam = create_speckles_of_certain_size_in_pixels(speckle_size, img_size + max_shift, 0, 1, 1, 0)
                    A = torch.Tensor(A).unsqueeze(0)

                    ############################################################################################################
                    ### Shift the speckle pattern one specific shift: ###
                    B, shiftx, shifty = shift_layer.forward(A, np.array([-movement_x]).astype(float32), np.array([-movement_y]).astype(float32))


                    # todo: yoav added this extra method
                    ##########################################################################################################
                    # Shift the speckle pattern several shifts and use whatever algorithm you want: ###
                    average_shifts = batch_generate_intermediate_speckles_and_average_estimates(A, B, k=k, movement_x=movement_x, movement_y=movement_y, SNR=SNR)
                    shiftx_list_yoav.append(average_shifts[0])
                    shifty_list_yoav.append(average_shifts[1])
                    ############################################################################################################
                    ### Crop tensors: ###
                    A = crop_tensor(A, (img_size, img_size))
                    B = crop_tensor(B, (img_size, img_size))

                    ### Get the noramlized Auto-Correlation of AA and BB: ###
                    #TODO: understand from where to lower delta_shift_AA and delta_shift_BB to get better estimate of the actual predicted shift
                    #TODO: meaning - do i lower delta_shift_AA from get_shifts_from_NCC(cc_AB) or from get_shifts_from_NCC(cc_BA)
                    cc_AA = normalized_cc_wrapper_with_noise(A, A, corr_size=3, SNR=SNR)
                    cc_BB = normalized_cc_wrapper_with_noise(B, B, corr_size=3, SNR=SNR)
                    delta_shiftx_AA, delta_shifty_AA, CC_peak_value_AA = get_shifts_from_NCC(cc_AA)
                    delta_shiftx_BB, delta_shifty_BB, CC_peak_value_BB = get_shifts_from_NCC(cc_BB)

                    ### Get the normalized Cross-Correlation (AB): ###
                    cc = normalized_cc_wrapper_with_noise(A, B, corr_size=3, SNR=SNR)
                    delta_shiftx, delta_shifty, CC_peak_value = get_shifts_from_NCC(cc)
                    # print(delta_shiftx)
                    # print(delta_shifty)
                    ### Add current results to results list: ###
                    shiftx_list.append(delta_shiftx)
                    shifty_list.append(delta_shifty)

                    ### Correct for self (auto-correlation) bias: ###
                    delta_shiftx_AA_corrected = delta_shiftx - delta_shiftx_AA
                    delta_shifty_AA_corrected = delta_shifty - delta_shifty_AA
                    shiftx_list_AB_AA.append(delta_shiftx_AA_corrected)
                    shifty_list_AB_AA.append(delta_shifty_AA_corrected)

                    ### Get the normalized Cross-Correlation (BA): ###
                    cc = normalized_cc_wrapper_with_noise(B, A, corr_size=3, SNR=SNR)
                    delta_shiftx_BA, delta_shifty_BA, CC_peak_value = get_shifts_from_NCC(cc)
                    # print(delta_shiftx)
                    # print(delta_shifty)
                    ### Add current results to results list: ###
                    shiftx_list_AB_BA.append((delta_shiftx-delta_shiftx_BA)/2)
                    shifty_list_AB_BA.append((delta_shifty-delta_shifty_BA)/2)
                    


                ### Convert the lists to numpy array: ###
                shiftx_array = torch.tensor(shiftx_list)
                shifty_array = torch.tensor(shifty_list)
                shiftx_array_AB_BA = torch.tensor(shiftx_list_AB_BA)
                shifty_array_AB_BA = torch.tensor(shifty_list_AB_BA)
                shiftx_array_AB_AA = torch.tensor(shiftx_list_AB_AA)
                shifty_array_AB_AA = torch.tensor(shifty_list_AB_AA)
                shiftx_array_yoav = torch.tensor(shiftx_list_yoav)
                shifty_array_yoav = torch.tensor(shifty_list_yoav)

                # Calculate STD across *one* movement, over *all* speckle families, which will tell us
                # the average noise of our method at each movement over many speckle families, we want a low number here
                # which means a given method is consistent, might have some bias
                # (which we don't care for much - watch out here, we want a constant bias but it is shift dependent)
                # but low variance
                # why that might not be what we want:
                # an argument can be made for that we actually want a low variance over *all* the movements per *one* speckle family,
                # meaning change the for loop ordering to have speckle family above movement,
                # then we will know that given one speckle family, we are good at all movements, which is what we have in a real scenario
                # (although we still show better std over all movements in the other scenario, it is averaged over speckle families and we dont have tha in real life)
                # now that is not enough. we want to take the ratio between the known mean and the real mean, across movements, and have that be roughly constant(or maybe mor law or something)
                # so that we keep the actual recording unchanged


                ### Insert all the results to the proper places in the table: ###
                shift_mean = torch.tensor([shiftx_array.mean(), shifty_array.mean()])
                shift_bias = torch.tensor([movement_x - shift_mean[0], movement_y - shift_mean[1]])
                shift_std = torch.tensor([(shiftx_array - shift_mean[0]).std(), (shifty_array - shift_mean[1]).std()])
                experiment_table_bias[speckle_size_counter, ROI_size_counter, shift_size_counter] = shift_bias
                experiment_table_std[speckle_size_counter, ROI_size_counter, shift_size_counter] = shift_std
                experiment_table_mean[speckle_size_counter, ROI_size_counter, shift_size_counter] = shift_mean
                experiment_table_params[speckle_size_counter, ROI_size_counter, shift_size_counter] = torch.tensor([speckle_size, img_size, movement_x])

                ### Insert all the results to the proper places in the table (AB-BA): ###
                shift_mean = torch.tensor([shiftx_array_AB_BA.mean(), shifty_array_AB_BA.mean()])
                shift_bias = torch.tensor([movement_x - shift_mean[0], movement_y - shift_mean[1]])
                shift_std = torch.tensor([(shiftx_array_AB_BA - shift_mean[0]).std(), (shifty_array_AB_BA - shift_mean[1]).std()])
                experiment_table_bias_AB_BA[speckle_size_counter, ROI_size_counter, shift_size_counter] = shift_bias
                experiment_table_std_AB_BA[speckle_size_counter, ROI_size_counter, shift_size_counter] = shift_std
                experiment_table_mean_AB_BA[speckle_size_counter, ROI_size_counter, shift_size_counter] = shift_mean

                ### Insert all the results to the proper places in the table (AB-AA): ###
                shift_mean = torch.tensor([shiftx_array_AB_AA.mean(), shifty_array_AB_AA.mean()])
                shift_bias = torch.tensor([movement_x - shift_mean[0], movement_y - shift_mean[1]])
                shift_std = torch.tensor([(shiftx_array_AB_AA - shift_mean[0]).std(), (shifty_array_AB_AA - shift_mean[1]).std()])
                experiment_table_bias_AB_AA[speckle_size_counter, ROI_size_counter, shift_size_counter] = shift_bias
                experiment_table_std_AB_AA[speckle_size_counter, ROI_size_counter, shift_size_counter] = shift_std
                experiment_table_mean_AB_AA[speckle_size_counter, ROI_size_counter, shift_size_counter] = shift_mean

                ### Insert all the results to the proper places in the table (yoav): ###
                shift_mean = torch.tensor([shiftx_array_yoav.mean(), shifty_array_yoav.mean()])
                shift_bias = torch.tensor([movement_x - shift_mean[0], movement_y - shift_mean[1]])
                shift_std = torch.tensor([(shiftx_array_yoav - shift_mean[0]).std(), (shifty_array_yoav - shift_mean[1]).std()])
                experiment_table_bias_yoav[speckle_size_counter, ROI_size_counter, shift_size_counter] = shift_bias
                experiment_table_std_yoav[speckle_size_counter, ROI_size_counter, shift_size_counter] = shift_std
                experiment_table_mean_yoav[speckle_size_counter, ROI_size_counter, shift_size_counter] = shift_mean


        ### Present bias and std-error as a function of shift for current working point (speckle size, ROI, ...): ###
        plt.plot(movements, experiment_table_bias[speckle_size_counter, ROI_size_counter, :, 0]); plt.title("NCC, bias"); plt.xlabel('GT shift'); plt.ylabel('bias'); plt.show();
        plt.plot(movements, experiment_table_std[speckle_size_counter, ROI_size_counter, :, 0]); plt.title("NCC, std x"); plt.xlabel('GT shift'); plt.ylabel('std'); plt.show();
        plt.plot(movements, experiment_table_std[speckle_size_counter, ROI_size_counter, :, 1]); plt.title("NCC, std y"); plt.xlabel('GT shift'); plt.ylabel('std'); plt.show();

        ### Present bias and std-error as a function of shift for current working point (speckle size, ROI, ...) After AB-BA Average: ###
        plt.plot(movements, experiment_table_bias_AB_BA[speckle_size_counter, ROI_size_counter, :, 0]); plt.title("NCC AB-BA,  bias");plt.xlabel('GT shift');plt.ylabel('bias');plt.show();
        plt.plot(movements, experiment_table_std_AB_BA[speckle_size_counter, ROI_size_counter, :, 0]); plt.title("NCC AB-BA,  std x"); plt.xlabel('GT shift'); plt.ylabel('std'); plt.show();
        plt.plot(movements, experiment_table_std_AB_BA[speckle_size_counter, ROI_size_counter, :, 1]);plt.title("NCC AB-BA,  std y");plt.xlabel('GT shift');plt.ylabel('std'); plt.show();

        ### Present bias and std-error as a function of shift for current working point (speckle size, ROI, ...) After AB-AA Average: ###
        plt.plot(movements, experiment_table_bias_AB_AA[speckle_size_counter, ROI_size_counter, :, 0]);plt.title("NCC AB-AA,  bias");plt.xlabel('GT shift');plt.ylabel('bias');plt.show();
        plt.plot(movements, experiment_table_std_AB_AA[speckle_size_counter, ROI_size_counter, :, 0]);plt.title("NCC AB-AA,  std x");plt.xlabel('GT shift');plt.ylabel('std');plt.show();
        plt.plot(movements, experiment_table_std_AB_AA[speckle_size_counter, ROI_size_counter, :, 1]);plt.title("NCC AB-AA,  std y");plt.xlabel('GT shift');plt.ylabel('std');plt.show();

        ### Present bias and std-error as a function of shift for current working point (speckle size, ROI, ...) After AB-AA Average: ###
        plt.plot(movements, experiment_table_bias_yoav[speckle_size_counter, ROI_size_counter, :, 0]); plt.title("NCC yoav,  bias"); plt.xlabel('GT shift'); plt.ylabel('bias'); plt.show();
        plt.plot(movements, experiment_table_std_yoav[speckle_size_counter, ROI_size_counter, :, 0]); plt.title("NCC yoav,  std x"); plt.xlabel('GT shift'); plt.ylabel('std'); plt.show();
        plt.plot(movements, experiment_table_std_yoav[speckle_size_counter, ROI_size_counter, :, 1]); plt.title("NCC yoav,  std y"); plt.xlabel('GT shift'); plt.ylabel('std'); plt.show();

        ### Plot ABBA and ABAA: ###
        plt.plot(movements, experiment_table_std[speckle_size_counter, ROI_size_counter, :, 0]);
        plt.plot(movements, experiment_table_std_AB_BA[speckle_size_counter, ROI_size_counter, :, 0]);
        plt.plot(movements, experiment_table_std_AB_AA[speckle_size_counter, ROI_size_counter, :, 0]);
        plt.plot(movements, experiment_table_std_yoav[speckle_size_counter, ROI_size_counter, :, 0]);
        plt.legend(['AB Regular', 'AB-BA', 'AB-AA', 'yoavs_method']);
        # plt.legend(['AB Regular', 'AB-AA', 'yoavs_method']);
        # plt.legend(['AB Regular', 'AB-BA', 'yoavs_method']);
        # plt.legend(['AB Regular', 'AB-BA', 'AB-AA']);
        plt.xlabel("movement")
        plt.ylabel("std")
        plt.title(f"SNR : {SNR}, K iterations : {k}")
        plt.show()

        ### Plot GT-Predicted shifts: ###
        plt.plot(movements, experiment_table_mean_yoav[speckle_size_counter, ROI_size_counter, :, 0]);
        plt.plot(movements, experiment_table_mean_AB_AA[speckle_size_counter, ROI_size_counter, :, 0]);
        plt.plot(movements, experiment_table_mean_AB_BA[speckle_size_counter, ROI_size_counter, :, 0]);
        plt.plot(movements, experiment_table_mean[speckle_size_counter, ROI_size_counter, :, 0]);
        plt.plot(movements, movements);
        plt.title('GT vs. Predicted shift X');
        plt.xlabel('GT shift');
        plt.legend(['yoavs method', 'AB-AA', 'AB-BA', 'regular', 'GT']);
        plt.show()



    save_path = "/raid/yoav/speckles"
    torch.save(experiment_table_bias, os.path.join(save_path, "bias_table.pt"))
    torch.save(experiment_table_std, os.path.join(save_path, "std_table.pt"))
    torch.save(experiment_table_params, os.path.join(save_path, "params_table.pt"))
    # torch.load(os.path.join(save_path, "bias_table.pt"))
    # torch.load(os.path.join(save_path, "std_table.pt"))
    # torch.load(os.path.join(save_path, "params_table.pt"))
    return experiment_table_bias, experiment_table_std, experiment_table_params


def ncc_bias_experiment_updated():
    ###############################################################################################################
    # START EXPERIMENT
    speckle_sizes = [5]
    img_sizes = [64]
    # movements = np.arange(0, 0.2, step=0.005)
    movements = my_linspace(0, 0.1, 5)
    # movements = np.array([0.01])
    experiment_size = 6
    max_shift = 20
    shift_layer = Shift_Layer_Torch()
    # EXPERIMENT_SIZE
    num_params = 3
    num_movement_axes = 2  # x, y
    SNR = 20.0
    k = 10  # determines how many speckles to create and compare to in yoavs algorithm
    x_vec = [-1, 0, 1]
    y_vec = [-1, 0, 1]
    experiment_table_std = torch.zeros(len(speckle_sizes), len(img_sizes), experiment_size, num_movement_axes)
    experiment_table_bias = torch.zeros(len(speckle_sizes), len(img_sizes), experiment_size, num_movement_axes)
    # experiment_table_mean = torch.zeros(len(speckle_sizes), len(img_sizes), experiment_size, num_movement_axes)
    experiment_table_std_AB_BA = torch.zeros(len(speckle_sizes), len(img_sizes), experiment_size, num_movement_axes)
    experiment_table_bias_AB_BA = torch.zeros(len(speckle_sizes), len(img_sizes), experiment_size, num_movement_axes)
    # experiment_table_mean_AB_BA = torch.zeros(len(speckle_sizes), len(img_sizes), experiment_size, num_movement_axes)
    experiment_table_std_AB_AA = torch.zeros(len(speckle_sizes), len(img_sizes), experiment_size, num_movement_axes)
    experiment_table_bias_AB_AA = torch.zeros(len(speckle_sizes), len(img_sizes), experiment_size, num_movement_axes)
    # experiment_table_mean_AB_AA = torch.zeros(len(speckle_sizes), len(img_sizes), experiment_size, num_movement_axes)
    experiment_table_std_yoav = torch.zeros(len(speckle_sizes), len(img_sizes), experiment_size, num_movement_axes)
    experiment_table_bias_yoav = torch.zeros(len(speckle_sizes), len(img_sizes), experiment_size, num_movement_axes)
    # experiment_table_mean_yoav = torch.zeros(len(speckle_sizes), len(img_sizes), experiment_size, num_movement_axes)
    experiment_table_params = torch.zeros(len(speckle_sizes), len(img_sizes), experiment_size, num_params)
    for speckle_size_counter, speckle_size in enumerate(speckle_sizes):
        for ROI_size_counter, img_size in enumerate(img_sizes):
            shiftx_list = []
            shifty_list = []
            shiftx_list_AB_BA = []
            shifty_list_AB_BA = []
            shiftx_list_AB_AA = []
            shifty_list_AB_AA = []
            shiftx_list_yoav = []
            shifty_list_yoav = []

            for family_id, speckle_family_counter in enumerate(range(experiment_size)):  # same movement for different speckles
                ###############################################################################################################
                print('current iteration: ' + str(speckle_family_counter))
                ### Create speckle pattern for this run: ###
                A, speckle_pattern_field1, speckle_pattern_field2, total_beam = create_speckles_of_certain_size_in_pixels(speckle_size, img_size + max_shift, 0, 1, 1, 0)
                A = torch.Tensor(A).unsqueeze(0)
                C, H, W = A.shape

                shifted_speckles = torch.zeros((experiment_size, img_size, img_size))
                movement_shiftx_list = []
                movement_shifty_list = []
                movement_shiftx_list_AB_BA = []
                movement_shifty_list_AB_BA = []
                movement_shiftx_list_AB_AA = []
                movement_shifty_list_AB_AA = []
                movement_shiftx_list_yoav = []
                movement_shifty_list_yoav = []
                for shift_size_counter, movement_x in enumerate(movements):
                    # movement_x = 0.15
                    movement_y = 0.0
                    ############################################################################################################
                    ### Shift the speckle pattern one specific shift: ###
                    B, shiftx, shifty = shift_layer.forward(A, np.array([-movement_x]).astype(float32), np.array([-movement_y]).astype(float32))

                    # todo: yoav added this extra method
                    ##########################################################################################################
                    # Shift the speckle pattern several shifts and use whatever algorithm you want: ###
                    average_shifts = generate_intermediate_speckles_and_average_estimates(A, B, k=k,
                                                                                          movement_x=movement_x,
                                                                                          movement_y=movement_y,
                                                                                          SNR=SNR)
                    movement_shiftx_list_yoav.append(average_shifts[0])
                    movement_shifty_list_yoav.append(average_shifts[1])
                    ############################################################################################################
                    ### Crop tensors: ###
                    A = crop_tensor(A, (img_size, img_size))
                    B = crop_tensor(B, (img_size, img_size))

                    ### Get the noramlized Auto-Correlation of AA and BB: ###
                    # TODO: understand from where to lower delta_shift_AA and delta_shift_BB to get better estimate of the actual predicted shift
                    # TODO: meaning - do i lower delta_shift_AA from get_shifts_from_NCC(cc_AB) or from get_shifts_from_NCC(cc_BA)
                    cc_AA = normalized_cc_wrapper_with_noise(A, A, corr_size=3, SNR=SNR)
                    cc_BB = normalized_cc_wrapper_with_noise(B, B, corr_size=3, SNR=SNR)
                    delta_shiftx_AA, delta_shifty_AA, CC_peak_value_AA = get_shifts_from_NCC(cc_AA)
                    delta_shiftx_BB, delta_shifty_BB, CC_peak_value_BB = get_shifts_from_NCC(cc_BB)

                    ### Get the normalized Cross-Correlation (AB): ###
                    cc = normalized_cc_wrapper_with_noise(A, B, corr_size=3, SNR=SNR)
                    delta_shiftx, delta_shifty, CC_peak_value = get_shifts_from_NCC(cc)
                    # print(delta_shiftx)
                    # print(delta_shifty)
                    ### Add current results to results list: ###
                    movement_shiftx_list.append(delta_shiftx)
                    movement_shifty_list.append(delta_shifty)

                    ### Correct for self (auto-correlation) bias: ###
                    delta_shiftx_AA_corrected = delta_shiftx - delta_shiftx_AA
                    delta_shifty_AA_corrected = delta_shifty - delta_shifty_AA
                    movement_shiftx_list_AB_AA.append(delta_shiftx_AA_corrected)
                    movement_shifty_list_AB_AA.append(delta_shifty_AA_corrected)

                    ### Get the normalized Cross-Correlation (BA): ###
                    cc = normalized_cc_wrapper_with_noise(B, A, corr_size=3, SNR=SNR)
                    delta_shiftx_BA, delta_shifty_BA, CC_peak_value = get_shifts_from_NCC(cc)
                    # print(delta_shiftx)
                    # print(delta_shifty)
                    ### Add current results to results list: ###
                    movement_shiftx_list_AB_BA.append((delta_shiftx - delta_shiftx_BA) / 2)
                    movement_shifty_list_AB_BA.append((delta_shifty - delta_shifty_BA) / 2)

                ### Convert the lists to numpy array: ###
                shiftx_list.append(torch.tensor(movement_shiftx_list))
                shifty_list.append(torch.tensor(movement_shifty_list))
                shiftx_list_AB_BA.append(torch.tensor(movement_shiftx_list_AB_BA))
                shifty_list_AB_BA.append(torch.tensor(movement_shifty_list_AB_BA))
                shiftx_list_AB_AA.append(torch.tensor(movement_shiftx_list_AB_AA))
                shifty_list_AB_AA.append(torch.tensor(movement_shifty_list_AB_AA))
                shiftx_list_yoav.append(torch.tensor(movement_shiftx_list_yoav))
                shifty_list_yoav.append(torch.tensor(movement_shifty_list_yoav))

            shiftx_array = torch.cat([s.unsqueeze(0) for s in shiftx_list])
            shifty_array = torch.cat([s.unsqueeze(0) for s in shifty_list])
            shiftx_array_AB_BA = torch.cat([s.unsqueeze(0) for s in shiftx_list_AB_BA])
            shifty_array_AB_BA = torch.cat([s.unsqueeze(0) for s in shifty_list_AB_BA])
            shiftx_array_AB_AA = torch.cat([s.unsqueeze(0) for s in shiftx_list_AB_AA])
            shifty_array_AB_AA = torch.cat([s.unsqueeze(0) for s in shifty_list_AB_AA])
            shiftx_array_yoav = torch.cat([s.unsqueeze(0) for s in shiftx_list_yoav])
            shifty_array_yoav = torch.cat([s.unsqueeze(0) for s in shifty_list_yoav])

            shift_bias_per_speckle_per_movement_x = shiftx_array - torch.tensor(movements).unsqueeze(0)
            shift_bias_per_speckle_x = shift_bias_per_speckle_per_movement_x.mean(1)
            shift_std_per_speckle_x = shift_bias_per_speckle_per_movement_x.std(1)
            shift_bias_per_speckle_per_movement_y = shifty_array - torch.tensor(movements).unsqueeze(0)
            shift_bias_per_speckle_y = shift_bias_per_speckle_per_movement_y.mean(1)
            shift_std_per_speckle_y = shift_bias_per_speckle_per_movement_y.std(1)
            experiment_table_bias[speckle_size_counter, ROI_size_counter, :, :] = torch.cat([shift_bias_per_speckle_x.unsqueeze(1), shift_bias_per_speckle_y.unsqueeze(1)], axis=1)
            experiment_table_std[speckle_size_counter, ROI_size_counter, :, :] = torch.cat([shift_std_per_speckle_x.unsqueeze(1), shift_std_per_speckle_y.unsqueeze(1)], axis=1)
            experiment_table_params[speckle_size_counter, ROI_size_counter, :, :] = torch.tensor([speckle_size, img_size, movement_x])

            ### Insert all the results to the proper places in the table (AB-BA): ###
            shift_bias_per_speckle_per_movement_x = shiftx_array_AB_BA - torch.tensor(movements).unsqueeze(0)
            shift_bias_per_speckle_x = shift_bias_per_speckle_per_movement_x.mean(1)
            shift_std_per_speckle_x = shift_bias_per_speckle_per_movement_x.std(1)
            shift_bias_per_speckle_per_movement_y = shifty_array_AB_BA - torch.tensor(movements).unsqueeze(0)
            shift_bias_per_speckle_y = shift_bias_per_speckle_per_movement_y.mean(1)
            shift_std_per_speckle_y = shift_bias_per_speckle_per_movement_y.std(1)
            experiment_table_std_AB_BA[speckle_size_counter, ROI_size_counter, :, :] = torch.cat([shift_bias_per_speckle_x.unsqueeze(1), shift_bias_per_speckle_y.unsqueeze(1)], axis=1)
            experiment_table_std_AB_BA[speckle_size_counter, ROI_size_counter, :, :] = torch.cat([shift_std_per_speckle_x.unsqueeze(1), shift_std_per_speckle_y.unsqueeze(1)], axis=1)

            ### Insert all the results to the proper places in the table (AB-AA): ###
            shift_bias_per_speckle_per_movement_x = shiftx_array_yoav - torch.tensor(movements).unsqueeze(0)
            shift_bias_per_speckle_x = shift_bias_per_speckle_per_movement_x.mean(1)
            shift_std_per_speckle_x = shift_bias_per_speckle_per_movement_x.std(1)
            shift_bias_per_speckle_per_movement_y = shifty_array_yoav - torch.tensor(movements).unsqueeze(0)
            shift_bias_per_speckle_y = shift_bias_per_speckle_per_movement_y.mean(1)
            shift_std_per_speckle_y = shift_bias_per_speckle_per_movement_y.std(1)
            experiment_table_std_yoav[speckle_size_counter, ROI_size_counter, :, :] = torch.cat([shift_bias_per_speckle_x.unsqueeze(1), shift_bias_per_speckle_y.unsqueeze(1)], axis=1)
            experiment_table_std_yoav[speckle_size_counter, ROI_size_counter, :, :] = torch.cat([shift_std_per_speckle_x.unsqueeze(1), shift_std_per_speckle_y.unsqueeze(1)], axis=1)

            ### Insert all the results to the proper places in the table (yoav): ###
            shift_bias_per_speckle_per_movement_x = shiftx_array_AB_AA - torch.tensor(movements).unsqueeze(0)
            shift_bias_per_speckle_x = shift_bias_per_speckle_per_movement_x.mean(1)
            shift_std_per_speckle_x = shift_bias_per_speckle_per_movement_x.std(1)
            shift_bias_per_speckle_per_movement_y = shifty_array_AB_AA - torch.tensor(movements).unsqueeze(0)
            shift_bias_per_speckle_y = shift_bias_per_speckle_per_movement_y.mean(1)
            shift_std_per_speckle_y = shift_bias_per_speckle_per_movement_y.std(1)
            experiment_table_std_AB_AA[speckle_size_counter, ROI_size_counter, :, :] = torch.cat([shift_bias_per_speckle_x.unsqueeze(1), shift_bias_per_speckle_y.unsqueeze(1)], axis=1)
            experiment_table_std_AB_AA[speckle_size_counter, ROI_size_counter, :, :] = torch.cat([shift_std_per_speckle_x.unsqueeze(1), shift_std_per_speckle_y.unsqueeze(1)], axis=1)

        ### Present bias and std-error as a function of shift for current working point (speckle size, ROI, ...): ###
        plt.plot(torch.arange(experiment_size), experiment_table_bias[speckle_size_counter, ROI_size_counter, :, 0]); plt.title("NCC, bias"); plt.xlabel('GT shift'); plt.ylabel('bias'); plt.show();
        plt.plot(torch.arange(experiment_size), experiment_table_std[speckle_size_counter, ROI_size_counter, :, 0]); plt.title("NCC, std x"); plt.xlabel('GT shift'); plt.ylabel('std'); plt.show();
        plt.plot(torch.arange(experiment_size), experiment_table_std[speckle_size_counter, ROI_size_counter, :, 1]); plt.title("NCC, std y"); plt.xlabel('GT shift'); plt.ylabel('std'); plt.show();

        ### Present bias and std-error as a function of shift for current working point (speckle size, ROI, ...) After AB-BA Average: ###
        plt.plot(torch.arange(experiment_size), experiment_table_bias_AB_BA[speckle_size_counter, ROI_size_counter, :, 0]); plt.title("NCC AB-BA,  bias"); plt.xlabel('GT shift'); plt.ylabel('bias'); plt.show();
        plt.plot(torch.arange(experiment_size), experiment_table_std_AB_BA[speckle_size_counter, ROI_size_counter, :, 0]); plt.title("NCC AB-BA,  std x"); plt.xlabel('GT shift'); plt.ylabel('std'); plt.show();
        plt.plot(torch.arange(experiment_size), experiment_table_std_AB_BA[speckle_size_counter, ROI_size_counter, :, 1]); plt.title("NCC AB-BA,  std y"); plt.xlabel('GT shift'); plt.ylabel('std'); plt.show();

        ### Present bias and std-error as a function of shift for current working point (speckle size, ROI, ...) After AB-AA Average: ###
        plt.plot(torch.arange(experiment_size), experiment_table_bias_AB_AA[speckle_size_counter, ROI_size_counter, :, 0]); plt.title("NCC AB-AA,  bias"); plt.xlabel('GT shift'); plt.ylabel('bias'); plt.show();
        plt.plot(torch.arange(experiment_size), experiment_table_std_AB_AA[speckle_size_counter, ROI_size_counter, :, 0]); plt.title("NCC AB-AA,  std x"); plt.xlabel('GT shift'); plt.ylabel('std'); plt.show();
        plt.plot(torch.arange(experiment_size), experiment_table_std_AB_AA[speckle_size_counter, ROI_size_counter, :, 1]); plt.title("NCC AB-AA,  std y"); plt.xlabel('GT shift'); plt.ylabel('std'); plt.show();

        ### Present bias and std-error as a function of shift for current working point (speckle size, ROI, ...) After AB-AA Average: ###
        plt.plot(torch.arange(experiment_size), experiment_table_bias_yoav[speckle_size_counter, ROI_size_counter, :, 0]); plt.title("NCC yoav,  bias"); plt.xlabel('GT shift'); plt.ylabel('bias'); plt.show();
        plt.plot(torch.arange(experiment_size), experiment_table_std_yoav[speckle_size_counter, ROI_size_counter, :, 0]); plt.title("NCC yoav,  std x"); plt.xlabel('GT shift'); plt.ylabel('std'); plt.show();
        plt.plot(torch.arange(experiment_size), experiment_table_std_yoav[speckle_size_counter, ROI_size_counter, :, 1]); plt.title("NCC yoav,  std y"); plt.xlabel('GT shift'); plt.ylabel('std'); plt.show();

        ### Plot std of all methods: ###
        plt.plot(torch.arange(experiment_size), experiment_table_std[speckle_size_counter, ROI_size_counter, :, 0]);
        plt.plot(torch.arange(experiment_size), experiment_table_std_AB_BA[speckle_size_counter, ROI_size_counter, :, 0]);
        plt.plot(torch.arange(experiment_size), experiment_table_std_AB_AA[speckle_size_counter, ROI_size_counter, :, 0]);
        plt.plot(torch.arange(experiment_size), experiment_table_std_yoav[speckle_size_counter, ROI_size_counter, :, 0]);
        plt.legend(['AB Regular', 'AB-BA', 'AB-AA', 'yoavs_method']);
        # plt.legend(['AB Regular', 'AB-AA', 'yoavs_method']);
        # plt.legend(['AB Regular', 'AB-BA', 'yoavs_method']);
        # plt.legend(['AB Regular', 'AB-BA', 'AB-AA']);
        plt.xlabel("speckle family index")
        plt.ylabel("std")
        plt.title(f"SNR : {SNR}, K iterations : {k}")
        plt.show()

        # ### Plot GT-Predicted shifts: ###
        # plt.plot(movements, experiment_table_mean_yoav[speckle_size_counter, ROI_size_counter, :, 0]);
        # plt.plot(movements, experiment_table_mean_AB_AA[speckle_size_counter, ROI_size_counter, :, 0]);
        # plt.plot(movements, experiment_table_mean_AB_BA[speckle_size_counter, ROI_size_counter, :, 0]);
        # plt.plot(movements, experiment_table_mean[speckle_size_counter, ROI_size_counter, :, 0]);
        # plt.plot(movements, movements);
        # plt.title('GT vs. Predicted shift X');
        # plt.xlabel('GT shift');
        # plt.legend(['yoavs method', 'AB-AA', 'AB-BA', 'regular', 'GT']);
        # plt.show()

    save_path = "/raid/yoav/speckles"
    torch.save(experiment_table_bias, os.path.join(save_path, "bias_table.pt"))
    torch.save(experiment_table_std, os.path.join(save_path, "std_table.pt"))
    torch.save(experiment_table_params, os.path.join(save_path, "params_table.pt"))
    # torch.load(os.path.join(save_path, "bias_table.pt"))
    # torch.load(os.path.join(save_path, "std_table.pt"))
    # torch.load(os.path.join(save_path, "params_table.pt"))
    return experiment_table_bias, experiment_table_std, experiment_table_params


def plot_speckles_fft(device=0):
    input_tensor = np.load("/home/yoav/rdnd/speckle_movie_with_turbulence.npy")
    input_tensor = np.load("/home/yoav/rdnd/speckle_movie.npy")
    speckle_1 = torch.tensor(input_tensor[0]).to(device).squeeze()
    speckle_2 = torch.tensor(input_tensor[1]).to(device).squeeze()

    fft_speckle_1 = torch.abs(torch.fft.fft2(speckle_1))
    log_fft_speckle_1 = torch.fft.fftshift(torch.log(fft_speckle_1))
    imshow_torch(log_fft_speckle_1, title_str="fft of speckle")
    imshow_torch(speckle_1, title_str="speckle")

    fft_speckle_2 = torch.abs(torch.fft.fft2(speckle_1))
    log_fft_speckle_2 = torch.fft.fftshift(torch.log(fft_speckle_2))
    imshow_torch(log_fft_speckle_2, title_str="fft of speckle")
    imshow_torch(speckle_2, title_str="speckle")

    imshow_torch(fft_speckle_2 - fft_speckle_1, title_str="fft diff")


def save_to_omer_rvrt():
    base_path = "/raid/datasets/speckles/wav_files_for_rvrt"
    wav_files_list = glob(os.path.join(base_path, "*"))
    std = 1
    SNR = 10
    amplitude = 0.1
    laser_count = 1
    speckle_size = 3
    number_of_samples = None
    for wav_file in wav_files_list:
        print(f"starting to save file {wav_file}")
        _, gt_shifts = create_speckles_movie_with_song_n_lasers(wav_file, True, SNR, amplitude, laser_count, speckle_size, number_of_samples)
        shift_save_path = os.path.join("/raid/datasets/speckles/gt_shifts", os.path.basename(wav_file)[:-4])
        if not os.path.isdir(shift_save_path):
            os.makedirs(shift_save_path, exist_ok=True)
        torch.save(gt_shifts, os.path.join(shift_save_path, "shifts.pt"))

# a = np.load("/raid/datasets/speckles/gt/allen_arrogh/00010.npy")
# b = np.load("/raid/datasets/speckles/noisy/allen_arrogh/00010.npy")
# imshow_torch(torch.tensor(a))
# imshow_torch(torch.tensor(b))

def mean_squared_error(estimate_shifts, gt_shifts):
    return np.mean((gt_shifts - estimate_shifts)**2)


def std_error(estimate_shifts, gt_shifts):
    return np.sqrt(mean_squared_error(estimate_shifts, gt_shifts) / len(gt_shifts))


def plot_gt_and_estimate(gt, estimate, min_y=-0.05, max_y=0.2, title_str="No Title Was Passed"):
    # plot gt shifts and estimated shifts with given ranges and title
    plt.figure()
    plt.plot(gt, c="r");
    plt.plot(estimate);
    plt.title(title_str);
    plt.xlim(0, len(estimate));
    plt.ylim(min_y, max_y);


def wavelet_transform(speckle):
    coeffs = pywt.dwt2(speckle.cpu().detach(), 'bior1.3')
    LL, (LH, HL, HH) = coeffs

    # titles = ['Approximation', ' Horizontal detail',
    #           'Vertical detail', 'Diagonal detail']
    #
    # fig = plt.figure(figsize=(12, 3))
    # for i, a in enumerate([LL, LH, HL, HH]):
    #     ax = fig.add_subplot(1, 4, i + 1)
    #     ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
    #     ax.set_title(titles[i], fontsize=10)
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    #
    # fig.tight_layout()
    # plt.show()

    return torch.tensor(LL), torch.tensor(LH), torch.tensor(HL), torch.tensor(HH)


def augment_speckles(speckle):
    """
    receives a speckle movie of shape [n, h, w] and outputs an augmented speckles of shape [# augmentations, n, h, w]
    Args:
        speckle:

    Returns:

    """
    # sobel_H = torch.tensor([[1, 0, -1],
    #                         [2, 0, -2],
    #                         [1, 0, -1]], dtype=torch.float32).to(speckle.device)
    # sobel_W = sobel_H.transpose(0, 1)
    # NoOp_filter = torch.tensor([[0, 0, 0],
    #                             [0, 1, 0],
    #                             [0, 0, 0]], dtype=torch.float32).to(speckle.device)
    # avg_filter = torch.tensor([ [0.1, 0.1, 0.1],
    #                             [0.1, 0.2, 0.1],
    #                             [0.1, 0.1, 0.1]], dtype=torch.float32).to(speckle.device)
    random_filters = torch.randn((100, 5, 5)).to(speckle.device)

    # augmentations = torch.cat([sobel_H.unsqueeze(0), sobel_W.unsqueeze(0), NoOp_filter.unsqueeze(0), avg_filter.unsqueeze(0), random_filters])
    modified = []

    for augmentation in random_filters:
        modified.append(torch.conv2d(input=torch.tensor(speckle.unsqueeze(1), dtype=torch.float32), weight=augmentation.unsqueeze(0).unsqueeze(0)))

    return torch.cat([x.squeeze().unsqueeze(0) for x in modified], dim=0)

# create_training_sample_for_fourier_estimation(N=64)

# speckle = torch.load("/home/yoav/rdnd/Speckles/speckle_pattern.pt")
# wavelet_transform(speckle)

# save_to_omer_rvrt()
# wave_file_path = "/raid/yoav/speckles/quality_testing/ncc_regular/SNR=6.0_Amp=0.05_K=20.wav"
# wave_file_path_1 = "/raid/yoav/speckles/quality_testing/ncc_yoav/SNR=6.0_Amp=0.05_K=20.wav"
# wave_file_path_2 = "/raid/datasets/speckles/wav_files_for_rvrt/bush_never_once.wav"
# original_song, fs = librosa.load(wave_file_path)
# original_song_1, fs_1 = librosa.load(wave_file_path_1)
# original_song_2, fs_2 = librosa.load(wave_file_path_2)
# n = len(original_song) // 4
# plt.plot(original_song_2[n:3*n]); plt.plot(original_song_1); plt.plot(original_song); plt.show();
