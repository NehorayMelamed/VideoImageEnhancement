


# from Rapid_Base.import_all import *

# ### Wiener Filtering: ###
import numpy as np
import cv2
from scipy.signal import convolve2d
from scipy.fft import fft2, ifft2, fftshift
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage import data, restoration
from skimage.restoration import unsupervised_wiener
# import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from skimage.restoration import estimate_sigma  # For estimating noise standard deviation
import torch
from typing import Tuple
from torch import Tensor
from skimage.util import img_as_float


### Import NUBKE: ###
from nubke_utils import *
from Utils import *

from video_editor.imshow_pyqt import *


def add_text_to_images(images, texts, font_size=2, font_thickness=2):
    """
    Adds a text to the center top of each image in the list.

    Parameters:
    -----------
    images : list of np.ndarray
        List of images.
    texts : list of str
        List of strings to print on each image.
    font_size : int
        The size of the font for the text.
    font_thickness : int
        The thickness of the font for the text.

    Returns:
    --------
    list of np.ndarray
        List of images with text printed on them.
    """
    if len(images) != len(texts):
        raise ValueError("The number of images must be equal to the number of texts.")

    font = cv2.FONT_HERSHEY_SIMPLEX

    images_with_text = []

    for img, text in zip(images, texts):
        # Calculate the width and height of the text to be added
        text_size = cv2.getTextSize(text, font, font_size, font_thickness)[0]

        # Calculate X position to center the text
        text_x = (img.shape[1] - text_size[0]) // 2
        text_y = text_size[1] + 10  # A little padding from the top

        # Add text to image
        img_with_text = cv2.putText(img.copy(), text, (text_x, text_y), font, font_size, (255, 255, 255),
                                    font_thickness, cv2.LINE_AA)

        # Append the modified image to the list
        images_with_text.append(img_with_text)

    return images_with_text


class WienerDeconvolution:
    """
    A class for performing various Wiener deconvolution operations on images.
    """

    @staticmethod
    def blur_image_gaussian_blur_with_blur_regularization(img, d=31):
        """
        Blur the edges of an image to reduce boundary effects during deconvolution.

        Parameters:
        img (np.ndarray): Input image with shape (H, W) or (H, W, C).
        d (int): Width of the border to be added around the image for padding. Default is 31.

        Returns:
        np.ndarray: Image with blurred edges with shape (H, W) or (H, W, C).
        """
        ### This Is The Code Block: ###
        h, w = img.shape[:2]  # Get the height and width of the input image
        img_pad = cv2.copyMakeBorder(img, d, d, d, d, cv2.BORDER_WRAP)  # Add padding around the image
        img_blur = cv2.GaussianBlur(img_pad, (2 * d + 1, 2 * d + 1), -1)[d:-d, d:-d]  # Apply Gaussian blur and remove padding
        y, x = np.indices((h, w))  # Create coordinate grids for the image
        dist = np.dstack([x, w - x - 1, y, h - y - 1]).min(-1)  # Compute minimum distance to the edge for each pixel
        w = np.minimum(np.float32(dist) / d, 1.0)  # Normalize distances to create weights
        return img * w + img_blur * (1 - w)  # Blend original image with blurred image using weights

    @staticmethod
    def create_motion_blur_kernel(angle, d, sz=65):
        """
        Create a motion blur kernel.

        Parameters:
        angle (float): Angle of motion blur in radians.
        d (int): Length of the motion blur.
        sz (int): Size of the kernel. Default is 65.

        Returns:
        np.ndarray: Motion blur kernel with shape (sz, sz).
        """
        ### This Is The Code Block: ###
        kern = np.ones((1, d), np.float32)  # Create a row vector with ones representing motion blur
        c, s = np.cos(angle), np.sin(angle)  # Cosine and sine of the angle
        A = np.float32([[c, -s, 0], [s, c, 0]])  # Rotation matrix for the given angle
        sz2 = sz // 2  # Half size of the kernel
        A[:, 2] = (sz2, sz2) - np.dot(A[:2, :2], [(d - 1) * 0.5, 0])  # Translate kernel to center it in the output array
        kern = cv2.warpAffine(kern, A, (sz, sz), flags=cv2.INTER_CUBIC)  # Rotate the kernel
        return kern  # Return the motion blur kernel

    @staticmethod
    def create_defocus_kernel(d, sz=65):
        """
        Create a defocus blur kernel.

        Parameters:
        d (int): Diameter of the defocus blur.
        sz (int): Size of the kernel. Default is 65.

        Returns:
        np.ndarray: Defocus blur kernel with shape (sz, sz).
        """
        ### This Is The Code Block: ###
        kern = np.zeros((sz, sz), np.uint8)  # Create an empty kernel
        cv2.circle(kern, (sz // 2, sz // 2), d, 255, -1, cv2.LINE_AA, shift=1)  # Draw a circle in the kernel
        kern = np.float32(kern) / 255.0  # Normalize the kernel to have values between 0 and 1
        return kern  # Return the defocus blur kernel

    @staticmethod
    def blind_deconvolution_unsupervised_wiener(image, psf, reg=None, user_params=None, is_real=True, clip=False,
                                                rng=None, deblur_strength=1.0, use_bw=False):
        """
        Perform blind deconvolution using Scikit-Image's unsupervised Wiener filter.

        Parameters:
        image (np.ndarray): The input image with shape (H, W) or (H, W, C).
        psf (np.ndarray): Initial estimate of the point spread function (PSF).
        reg (np.ndarray or float, optional): The regularization parameter. Default is None.
        user_params (dict, optional): Parameters passed to the user-defined function. Default is None.
        is_real (bool): If True, assumes the output is a real image. Default is True.
        clip (bool): If True, clips the output image to the range [0, 1]. Default is True.
        rng (numpy.random.Generator or numpy.random.RandomState, optional): Random number generator. Default is None.
        deblur_strength (float, optional): Blending factor between 0 and 1 to control deblurring strength. Default is 1.0.
        use_bw (bool, optional): If True, converts RGB image to BW for deblurring. Default is False.

        Returns:
        tuple: Tuple containing:
            - deblurred_image (np.ndarray): The deblurred image with shape (H, W) or (H, W, C).
            - psf (np.ndarray): The estimated point spread function (PSF) with shape (H, W).
        """

        ### Blend Blur Kernel with Dirac Delta: ###
        psf = WienerDeconvolution.blend_blur_kernel_with_dirac(psf, deblur_strength)  # Blend PSF with Dirac delta

        ### Check Image Dimensionality: ###
        if image.ndim == 3 and image.shape[2] == 3:  # If image is RGB
            if use_bw:  # If BW conversion is requested
                ### Convert RGB to BW and Perform Deblurring: ###
                image_BW = RGB2BW(image)[:, :, 0]  # Convert RGB to BW
                deblurred_BW, unsupervised_dict = unsupervised_wiener(
                    image_BW, psf, reg=reg, user_params=user_params, is_real=is_real, clip=clip, rng=rng
                )  # Perform Unsupervised Wiener Deconvolution on BW image
                deblurred_image = BW2RGB(deblurred_BW)  # Convert deblurred BW image back to RGB

            else:  # If channel-wise deblurring is requested
                deblurred_image = np.zeros_like(image)  # Initialize output image
                for c in range(3):  # Loop through each RGB channel
                    deblurred_image[:, :, c], unsupervised_dict = unsupervised_wiener(
                        image[:, :, c], psf, reg=reg, user_params=user_params, is_real=is_real, clip=clip, rng=rng
                    )  # Perform Unsupervised Wiener Deconvolution on each channel
        elif image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):  # If image is grayscale
            deblurred_image, unsupervised_dict = unsupervised_wiener(
                image, psf, reg=reg, user_params=user_params, is_real=is_real, clip=clip, rng=rng
            )  # Perform Unsupervised Wiener Deconvolution

        else:
            raise ValueError(
                "Input image must be of shape (H, W) or (H, W, C) where C=1 or 3.")  # Raise error for invalid shapes

        # gn_chain = unsupervised_dict['noise']
        # gx_chain = unsupervised_dict['prior']
        return deblurred_image, psf  # Return deblurred image and PSF

    @staticmethod
    def non_blind_deconvolution_classic_wiener(image, psf, snr, deblur_strength=1.0):
        """
        Perform Wiener filtering in the Fourier domain.

        Parameters:
        image (np.ndarray): The input image with shape (H, W) or (H, W, C).
        psf (np.ndarray): The blur kernel with shape (kernel_H, kernel_W).
        snr (float): The estimated Signal-to-Noise Ratio (SNR).
        deblur_strength (float, optional): Blending factor between 0 and 1 to control deblurring strength. Default is 1.0.

        Returns:
        np.ndarray: The deblurred image with shape (H, W) or (H, W, C).
        """
        ### Blend Blur Kernel with Dirac Delta: ###
        psf = WienerDeconvolution.blend_blur_kernel_with_dirac(psf, deblur_strength)

        ### Normalize the PSF: ###
        psf /= psf.sum()  # Normalize the PSF so that its sum is 1

        ### Check Image Dimensionality: ###
        if image.ndim == 3 and image.shape[2] == 3:  # If image is RGB
            deblurred_image = np.zeros_like(image)  # Initialize output image

            ### Loop Over Color Channels: ###
            for c in range(3):  # Process each channel separately
                deblurred_image[:, :, c] = WienerDeconvolution.wiener_deconvolution_fourier(image[:, :, c], psf, snr)  # Recursive call

            return deblurred_image  # Return the deblurred RGB image

        elif image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):  # If image is grayscale
            ### Compute Fourier Transforms: ###
            image_fft = fft2(image)  # Fourier transform of the image
            psf_fft = fft2(psf, s=image.shape[:2])  # Fourier transform of the blur kernel with padding

            ### Wiener Filter Calculation: ###
            psf_fft_conj = np.conj(psf_fft)  # Conjugate of the blur kernel's Fourier transform
            H2 = np.abs(psf_fft) ** 2  # Power spectrum of the blur kernel
            wiener_filter = psf_fft_conj / (H2 + 1 / snr)  # Wiener filter formula

            ### Apply Filter and Inverse Fourier Transform: ###
            deblurred_fft = wiener_filter * image_fft  # Apply Wiener filter in frequency domain
            deblurred_image = np.real(ifft2(deblurred_fft))  # Inverse Fourier transform to get deblurred image

            return deblurred_image  # Return the deblurred image

        else:
            raise ValueError("Input image must be of shape (H, W) or (H, W, C) where C=1 or 3.")  # Raise error for invalid shapes

    @staticmethod
    def non_blind_deconvolution_classic_wiener_opencv(image, psf, noise=1e-6, deblur_strength=1.0):
        """
        Perform non-blind Wiener deconvolution using OpenCV for both RGB and grayscale images.

        Parameters:
        image (np.ndarray): Input image with shape (H, W) or (H, W, C).
        psf (np.ndarray): Point spread function (PSF) with shape (kernel_H, kernel_W).
        noise (float): Noise power for Wiener filter. Default is 1e-6.
        deblur_strength (float, optional): Blending factor between 0 and 1 to control deblurring strength. Default is 1.0.

        Returns:
        np.ndarray: Deblurred image with shape (H, W) or (H, W, C).
        """
        ### Blend Blur Kernel with Dirac Delta: ###
        psf = WienerDeconvolution.blend_blur_kernel_with_dirac(psf, deblur_strength)

        ### Normalize the PSF: ###
        psf /= psf.sum()  # Normalize the PSF so that its sum is 1

        ### Pad the PSF to the Size of the Input Image: ###
        psf_pad = np.zeros_like(image)  # Initialize a padded PSF with zeros
        kh, kw = psf.shape  # Get the dimensions of the PSF
        psf_pad[:kh, :kw] = psf  # Place the PSF in the top-left corner of the padded PSF

        ### Compute the DFT of the Padded PSF: ###
        PSF = cv2.dft(psf_pad, flags=cv2.DFT_COMPLEX_OUTPUT)  # Compute the DFT of the padded PSF
        PSF2 = (PSF ** 2).sum(-1)  # Compute the power spectrum of the PSF
        iPSF = PSF / (PSF2 + noise)[..., np.newaxis]  # Compute the inverse PSF with noise regularization

        if image.ndim == 2:  # If the input image is grayscale
            ### Compute the DFT of the Input Image: ###
            IMG = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)  # Compute the DFT of the input image

            ### Multiply Spectrums (Element-Wise Multiplication in Frequency Domain): ###
            RES = cv2.mulSpectrums(IMG, iPSF, 0)  # Multiply the image DFT with the inverse PSF

            ### Compute the Inverse DFT to Get the Deblurred Image: ###
            res = cv2.idft(RES, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)  # Compute the inverse DFT to get the deblurred image

            ### Correct the Shift Caused by the DFT: ###
            res = np.roll(res, -kh // 2, axis=0)  # Correct the vertical shift
            res = np.roll(res, -kw // 2, axis=1)  # Correct the horizontal shift

            return res  # Return the deblurred grayscale image

        elif image.ndim == 3 and image.shape[2] == 3:  # If the input image is RGB
            res_rgb = np.zeros_like(image)  # Initialize the output RGB image

            ### Loop Over Color Channels: ###
            for c in range(3):  # Process each color channel separately
                ### Compute the DFT of the Color Channel: ###
                IMG_C = cv2.dft(np.float32(image[:, :, c]), flags=cv2.DFT_COMPLEX_OUTPUT)  # Compute the DFT of the color channel

                ### Multiply Spectrums (Element-Wise Multiplication in Frequency Domain): ###
                RES_C = cv2.mulSpectrums(IMG_C, iPSF, 0)  # Multiply the color channel DFT with the inverse PSF

                ### Compute the Inverse DFT to Get the Deblurred Color Channel: ###
                res_c = cv2.idft(RES_C, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)  # Compute the inverse DFT to get the deblurred color channel

                ### Correct the Shift Caused by the DFT: ###
                res_c = np.roll(res_c, -kh // 2, axis=0)  # Correct the vertical shift
                res_c = np.roll(res_c, -kw // 2, axis=1)  # Correct the horizontal shift

                res_rgb[:, :, c] = res_c  # Store the deblurred color channel in the output image

            return res_rgb  # Return the deblurred RGB image

        else:
            raise ValueError("Input image must be of shape (H, W) or (H, W, C) where C=1 or 3.")  # Raise an error for invalid shapes


    @staticmethod
    def blind_non_blind_deconvolution_richardson_lucy(image, psf, num_iter=50, deblur_strength=1.0):
        """
        Perform blind deconvolution using the Richardson-Lucy algorithm.

        Parameters:
        image (np.ndarray): The observed (blurred and noisy) image with shape (H, W) or (H, W, C).
        psf (np.ndarray): Initial estimate of the point spread function (PSF).
        num_iter (int): Number of iterations. Default is 50.
        deblur_strength (float, optional): Blending factor between 0 and 1 to control deblurring strength. Default is 1.0.

        Returns:
        np.ndarray: Deblurred image.
        np.ndarray: Estimated PSF.
        """
        ### Blend Blur Kernel with Dirac Delta: ###
        psf = WienerDeconvolution.blend_blur_kernel_with_dirac(psf, deblur_strength)

        ### Initialize Deconvolved Image: ###
        im_deconv = np.full(image.shape, 0.1, dtype='float')  # Initialize deconvolved image

        ### Looping Over Iterations: ###
        for i in range(num_iter):  # Perform iterations
            psf_mirror = np.flip(psf)  # Flip the PSF
            conv = fftconvolve(im_deconv, psf, mode='same')  # Convolve the deconvolved image with the PSF
            relative_blur = image / conv  # Compute the relative blur
            im_deconv *= fftconvolve(relative_blur, psf_mirror, mode='same')  # Update the deconvolved image
            im_deconv_mirror = np.flip(im_deconv)  # Flip the deconvolved image
            psf *= fftconvolve(relative_blur, im_deconv_mirror, mode='same')  # Update the PSF
            psf /= psf.sum()  # Normalize the PSF

        return im_deconv, psf  # Return the deblurred image and PSF

    @staticmethod
    def non_blind_deconvolution_wiener_skimage(image, psf, balance, reg=None, is_real=True, clip=True, deblur_strength=1.0):
        """
        Perform Wiener deconvolution using the Wiener filter provided by skimage.

        Parameters:
        image (np.ndarray): The observed (blurred and noisy) image with shape (H, W) or (H, W, C).
        psf (np.ndarray): Point spread function (PSF) with shape (kernel_H, kernel_W).
        balance (float): The regularization parameter for Wiener filtering.
        reg (np.ndarray or float, optional): The regularization parameter. Default is None.
        is_real (bool): If True, assumes the output is a real image. Default is True.
        clip (bool): If True, clips the output image to the range [0, 1]. Default is True.
        deblur_strength (float, optional): Blending factor between 0 and 1 to control deblurring strength. Default is 1.0.

        Returns:
        np.ndarray: Deblurred image.
        """
        ### Blend Blur Kernel with Dirac Delta: ###
        psf = WienerDeconvolution.blend_blur_kernel_with_dirac(psf, deblur_strength)

        ### Perform Wiener Deconvolution: ###
        deblurred_image = restoration.wiener(image, psf, balance, reg=reg, is_real=is_real, clip=clip)

        return deblurred_image

    @staticmethod
    def blind_non_blind_deconvolution_richardson_lucy_skimage(image, psf, num_iter=50, num_psf_iter=5, deblur_strength=1.0):
        """
        Perform blind deconvolution using the Richardson-Lucy algorithm with Wiener filtering.

        Parameters:
        image (np.ndarray): The observed (blurred and noisy) image with shape (H, W) or (H, W, C).
        psf (np.ndarray): Initial estimate of the point spread function (PSF).
        num_iter (int): Number of iterations for image and PSF deconvolution. Default is 50.
        num_psf_iter (int): Number of iterations for PSF estimation in each outer loop. Default is 5.
        deblur_strength (float, optional): Blending factor between 0 and 1 to control deblurring strength. Default is 1.0.

        Returns:
        np.ndarray: Deblurred image.
        np.ndarray: Estimated PSF.
        """
        ### Ensure the Image is in Float Format: ###
        image = img_as_float(image)  # Convert image to float

        ### Blend Blur Kernel with Dirac Delta: ###
        psf = WienerDeconvolution.blend_blur_kernel_with_dirac(psf, deblur_strength)

        ### Iterative Richardson-Lucy Deconvolution: ###
        for i in range(num_iter):  # Perform iterations
            ### Update the Estimated Image Using Richardson-Lucy Algorithm: ###
            image_deconv = restoration.richardson_lucy(image, psf, num_iter=num_psf_iter, clip=False)

            ### Update the PSF: ###
            psf_mirror = np.flip(psf)  # Flip the PSF
            relative_blur = image / (fftconvolve(image_deconv, psf, mode='same') + 1e-10)  # Compute the relative blur
            psf_update = fftconvolve(relative_blur, np.flip(image_deconv), mode='same')  # Update the PSF

            ### Pad PSF to Match the Dimensions of PSF Update: ###
            psf_padded = np.zeros_like(psf_update)  # Initialize padded PSF
            kh, kw = psf.shape  # Get dimensions of PSF
            psf_padded[:kh, :kw] = psf  # Place PSF in top-left corner of padded PSF

            ### Normalize the PSF to Ensure its Sum is 1: ###
            psf_padded *= psf_update
            psf_padded /= psf_padded.sum()  # Normalize PSF

            ### Extract the Updated PSF from the Padded PSF: ###
            psf = psf_padded[:kh, :kw]  # Update PSF

        return image_deconv, psf  # Return the deblurred image and PSF

    @staticmethod
    def estimate_snr_and_perform_deconvolution_EM(observed_image, estimated_image, psf, method='wiener_opencv', param=3,
                                                  max_iter=10, reg=None, num_iter=10, num_psf_iter=5):
        """
        Estimate the Signal-to-Noise Ratio (SNR) using the Expectation-Maximization (EM) algorithm.

        Parameters:
        observed_image (np.ndarray): The observed (blurred and noisy) image with shape (H, W) or (H, W, C).
        estimated_image (np.ndarray): The initial estimated (deblurred) image with shape (H, W) or (H, W, C).
        psf (np.ndarray): Point spread function (PSF) with shape (kernel_H, kernel_W).
        method (str or int): Method to use for deconvolution. Options: 'wiener_skimage', 'wiener_opencv',
                             'wiener_classic', 'unsupervised_wiener', 'richardson_lucy', 'richardson_lucy_skimage'.
        param (float or int): Regularization parameter for Wiener filtering or number of iterations for Richardson-Lucy.
                              Default is 3.
        reg (np.ndarray or float, optional): Regularization parameter for Wiener filtering. Default is None.
        num_iter (int): Number of iterations for Richardson-Lucy deconvolution. Default is 10.
        num_psf_iter (int): Number of iterations for PSF estimation in Richardson-Lucy deconvolution. Default is 5.
        max_iter (int): Maximum number of iterations for the EM algorithm. Default is 10.

        Returns:
        float: The estimated SNR.
        np.ndarray: The final estimated deblurred image.
        """
        ### This Is The Code Block: ###
        for i in range(max_iter):
            # E-step: Compute the residual
            residual = observed_image - estimated_image  # Compute the residual image

            # M-step: Update the SNR estimate
            signal_power = np.mean(estimated_image ** 2)  # Compute the power of the estimated image
            noise_power = np.mean(residual ** 2)  # Compute the power of the residual (noise)
            snr = signal_power / noise_power  # Compute the SNR

            # Update the estimated image
            estimated_image = WienerDeconvolution.deconvolution_wrapper(observed_image,
                                                                        psf,
                                                                        method,
                                                                        param=1 / snr,
                                                                        reg=reg,
                                                                        num_iter=num_iter,
                                                                        num_psf_iter=num_psf_iter)  # Perform deconvolution

        return snr, estimated_image

    @staticmethod
    def estimate_snr_residual_between_observed_and_estimated(observed_image, estimated_image):
        """
        Estimate the Signal-to-Noise Ratio (SNR) using residual analysis.

        Parameters:
        observed_image (np.ndarray): The observed (blurred and noisy) image with shape (H, W) or (H, W, C).
        estimated_image (np.ndarray): The estimated (deblurred) image with shape (H, W) or (H, W, C).

        Returns:
        float: The estimated SNR.
        """
        ### This Is The Code Block: ###
        residual = observed_image - estimated_image  # Compute the residual image
        signal_power = np.mean(estimated_image ** 2)  # Compute the power of the estimated image
        noise_power = np.mean(residual ** 2)  # Compute the power of the residual (noise)
        snr = signal_power / noise_power  # Compute the SNR
        return snr

    @staticmethod
    def estimate_snr_using_real_space_local_variance(observed_image, patch_size=7):
        """
        Estimate the Signal-to-Noise Ratio (SNR) using local variance estimation in smooth regions.

        Parameters:
        observed_image (np.ndarray): The observed (blurred and noisy) image with shape (H, W) or (H, W, C).
        patch_size (int): Size of the patch for local variance estimation. Default is 7.

        Returns:
        float: The estimated SNR.
        """
        ### This Is The Code Block: ###
        smooth_regions = cv2.GaussianBlur(observed_image, (patch_size, patch_size), 0)  # Smooth regions of the image
        local_variance = np.var(smooth_regions)  # Compute local variance in smooth regions
        global_variance = np.var(observed_image)  # Compute global variance of the image
        snr = global_variance / local_variance  # Compute the SNR
        return snr

    @staticmethod
    def estimate_snr_using_fourier_psd_analysis(observed_image):
        """
        Estimate the Signal-to-Noise Ratio (SNR) using power spectral density (PSD) analysis.

        Parameters:
        observed_image (np.ndarray): The observed (blurred and noisy) image with shape (H, W) or (H, W, C).

        Returns:
        float: The estimated SNR.
        """
        ### This Is The Code Block: ###
        f = np.fft.fft2(observed_image)  # Compute the 2D FFT of the observed image
        psd = np.abs(f) ** 2  # Compute the power spectral density
        signal_power = np.mean(psd)  # Compute the mean power of the signal
        noise_power = np.var(psd)  # Compute the variance of the power spectral density (noise power)
        snr = signal_power / noise_power  # Compute the SNR
        return snr

    @staticmethod
    def estimate_noise_real_space_skimage(image, average_sigmas=False, channel_axis=None):
        """
        Estimate the noise standard deviation in the image using skimage's estimate_sigma function.

        Parameters:
        image (np.ndarray): The observed image with shape (H, W) or (H, W, C).
        average_sigmas (bool): If True, averages the noise standard deviation across channels. Default is False.
        channel_axis (int or None): If not None, specifies the channel axis. Default is None.

        Returns:
        float or np.ndarray: The estimated noise standard deviation.
        """
        sigma_est = estimate_sigma(image, average_sigmas=average_sigmas, channel_axis=channel_axis)
        return sigma_est

    @staticmethod
    def estimate_snr_real_space(original_image, blurred_image):
        """
        Estimate the Signal-to-Noise Ratio (SNR) based on the power of the original and blurred images.

        Parameters:
        original_image (np.ndarray): The original image with shape (H, W) or (H, W, C).
        blurred_image (np.ndarray): The blurred image with shape (H, W) or (H, W, C).

        Returns:
        float: The estimated SNR.
        """
        ### Compute Power of Original and Blurred Images: ###
        original_power = np.mean(np.square(original_image))  # Power of the original image
        noise_power = np.mean(np.square(original_image - blurred_image))  # Power of the noise
        snr = original_power / noise_power  # SNR calculation
        return snr  # Return the estimated SNR

    @staticmethod
    def estimate_snr_wrapper(observed_image, method, original_image=None, patch_size=7, average_sigmas=False, channel_axis=None):
        """
        Wrapper function to estimate SNR using various methods based on the input string or index.

        Parameters:
        observed_image (np.ndarray): The observed (blurred and noisy) image with shape (H, W) or (H, W, C).
        method (str or int): Method to use for SNR estimation. Options: 'real_space_local_variance',
                             'fourier_psd_analysis', 'noise_real_space', 'real_space'.
        original_image (np.ndarray, optional): The original image for real_space method. Default is None.
        patch_size (int): Size of the patch for local variance estimation. Default is 7.
        average_sigmas (bool): If True, averages the noise standard deviation across channels. Default is False.
        channel_axis (int or None): If not None, specifies the channel axis. Default is None.

        Returns:
        float: The estimated SNR.
        """
        if method == 'real_space_local_variance' or method == 0:
            return WienerDeconvolution.estimate_snr_using_real_space_local_variance(observed_image, patch_size)
        elif method == 'fourier_psd_analysis' or method == 1:
            return WienerDeconvolution.estimate_snr_using_fourier_psd_analysis(observed_image)
        elif method == 'noise_real_space' or method == 2:
            return WienerDeconvolution.estimate_noise_real_space_skimage(observed_image, average_sigmas, channel_axis)
        elif method == 'real_space' or method == 3:
            if original_image is None:
                raise ValueError("original_image must be provided for the 'real_space' method.")
            return WienerDeconvolution.estimate_snr_real_space(original_image, observed_image)
        else:
            raise ValueError("Invalid method. Choose from 'real_space_local_variance', 'fourier_psd_analysis', "
                             "'noise_real_space', 'real_space'.")

    # @staticmethod
    # def blind_non_blind_deconvolution_richardson_lucy(image, psf, num_iter=50):
    #     """
    #     Perform blind deconvolution using the Richardson-Lucy algorithm.
    #
    #     Parameters:
    #     image (np.ndarray): The observed (blurred and noisy) image with shape (H, W) or (H, W, C).
    #     psf (np.ndarray): Initial estimate of the point spread function (PSF).
    #     num_iter (int): Number of iterations. Default is 50.
    #
    #     Returns:
    #     np.ndarray: Deblurred image.
    #     np.ndarray: Estimated PSF.
    #     """
    #     ### This Is The Code Block: ###
    #     im_deconv = np.full(image.shape, 0.1, dtype='float')  # Initialize deconvolved image
    #     ### Looping Over Indices: ###
    #     for i in range(num_iter):  # Perform iterations
    #         psf_mirror = np.flip(psf)  # Flip the PSF
    #         conv = fftconvolve(im_deconv, psf, mode='same')  # Convolve the deconvolved image with the PSF
    #         relative_blur = image / conv  # Compute the relative blur
    #         im_deconv *= fftconvolve(relative_blur, psf_mirror, mode='same')  # Update the deconvolved image
    #         im_deconv_mirror = np.flip(im_deconv)  # Flip the deconvolved image
    #         psf *= fftconvolve(relative_blur, im_deconv_mirror, mode='same')  # Update the PSF
    #         psf /= psf.sum()  # Normalize the PSF
    #     return im_deconv, psf


    # @staticmethod
    # def blind_non_blind_deconvolution_richardson_lucy_skimage(image, psf, num_iter=50, num_psf_iter=5):
    #     """
    #     Perform blind deconvolution using the Richardson-Lucy algorithm with Wiener filtering.
    #
    #     Parameters:
    #     image (np.ndarray): The observed (blurred and noisy) image with shape (H, W) or (H, W, C).
    #     psf (np.ndarray): Initial estimate of the point spread function (PSF).
    #     num_iter (int): Number of iterations for image and PSF deconvolution. Default is 50.
    #     num_psf_iter (int): Number of iterations for PSF estimation in each outer loop. Default is 5.
    #
    #     Returns:
    #     np.ndarray: Deblurred image.
    #     np.ndarray: Estimated PSF.
    #     """
    #     image = img_as_float(image)  # Ensure the image is in float format
    #
    #     for i in range(num_iter):
    #         # Update the estimated image using Richardson-Lucy algorithm
    #         image_deconv = restoration.richardson_lucy(image, psf, num_iter=num_psf_iter, clip=False)
    #
    #         # Update the PSF
    #         psf_mirror = np.flip(psf)
    #         relative_blur = image / (fftconvolve(image_deconv, psf, mode='same') + 1e-10)
    #         psf_update = fftconvolve(relative_blur, np.flip(image_deconv), mode='same')
    #
    #         # Pad psf to match the dimensions of psf_update
    #         psf_padded = np.zeros_like(psf_update)
    #         kh, kw = psf.shape
    #         psf_padded[:kh, :kw] = psf
    #
    #         psf_padded *= psf_update
    #         psf_padded /= psf_padded.sum()  # Normalize the PSF to ensure its sum is 1
    #
    #         # Extract the updated PSF from the padded PSF
    #         psf = psf_padded[:kh, :kw]
    #
    #     return image_deconv, psf

    @staticmethod
    def estimate_snr_fourier(input_image, blur_kernel):
        """
        Estimate SNR for each Fourier component.

        Parameters:
        input_image (np.ndarray): The input image with shape (H, W) or (H, W, C).
        blur_kernel (np.ndarray): The blur kernel with shape (kernel_H, kernel_W).

        Returns:
        np.ndarray: SNR spectrum with shape (H, W) or (H, W, C).
        """
        if input_image.ndim == 3 and input_image.shape[2] == 3:  # If image is RGB
            snr_spectrum = np.zeros_like(input_image, dtype=float)  # Initialize SNR spectrum
            ### Looping Over Channels: ###
            for c in range(3):  # Process each channel separately
                snr_spectrum[:, :, c] = WienerDeconvolution.estimate_snr_fourier(input_image[:, :, c],
                                                                                 blur_kernel)  # Recursive call for each channel
            return snr_spectrum  # Return SNR spectrum for RGB image
        elif input_image.ndim == 2 or (input_image.ndim == 3 and input_image.shape[2] == 1):  # If image is grayscale
            ### Compute Fourier Transforms: ###
            image_fft = fft2(input_image)  # Fourier transform of the image
            kernel_fft = fft2(blur_kernel, s=input_image.shape[:2])  # Fourier transform of the blur kernel with padding

            ### Compute Power Spectra: ###
            image_power_spectrum = np.abs(image_fft) ** 2  # Power spectrum of the image
            kernel_power_spectrum = np.abs(kernel_fft) ** 2  # Power spectrum of the blur kernel

            ### Estimate Noise Power Spectrum: ###
            noise_power_spectrum = np.mean(kernel_power_spectrum)  # Noise power spectrum estimate
            snr_spectrum = image_power_spectrum / (noise_power_spectrum + 1e-10)  # SNR spectrum calculation

            return snr_spectrum  # Return the SNR spectrum
        else:
            raise ValueError(
                "Input image must be of shape (H, W) or (H, W, C) where C=1 or 3.")  # Raise error for invalid shapes

    @staticmethod
    def load_grayscale_image(image_path):
        """
        Load an image from a file in grayscale mode.

        Parameters:
        image_path (str): The file path to the image.

        Returns:
        np.ndarray: The loaded grayscale image with shape (H, W).
        """
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load image in grayscale
        return image  # Return the grayscale image

    @staticmethod
    def display_images_side_by_side(original_image, deblurred_image_fourier, deblurred_image_real=None):
        """
        Display the original and deblurred images side by side.

        Parameters:
        original_image (np.ndarray): The original image with shape (H, W) or (H, W, C).
        deblurred_image_fourier (np.ndarray): The deblurred image using Fourier method with shape (H, W) or (H, W, C).
        deblurred_image_real (np.ndarray): The deblurred image using real space method with shape (H, W) or (H, W, C).
        """
        ### Create a Figure for Display: ###
        plt.figure(figsize=(15, 5))  # Set figure size
        if deblurred_image_real is not None:
            number_of_images = 3
        else:
            number_of_images = 2

        ### Display Original Image: ###
        plt.subplot(1, number_of_images, 1)  # Create subplot for original image
        plt.title('Original Image')  # Set title for original image
        if original_image.ndim == 2:  # If grayscale
            plt.imshow(original_image, cmap='gray')  # Display original image in grayscale
        else:  # If RGB
            plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))  # Display original image in RGB

        ### Display Deblurred Image (Fourier): ###
        plt.subplot(1, number_of_images, 2)  # Create subplot for Fourier deblurred image
        plt.title('Deblurred Image (Fourier)')  # Set title for Fourier deblurred image
        if deblurred_image_fourier.ndim == 2:  # If grayscale
            plt.imshow(deblurred_image_fourier, cmap='gray')  # Display Fourier deblurred image in grayscale
        else:  # If RGB
            plt.imshow(
                cv2.cvtColor(deblurred_image_fourier, cv2.COLOR_BGR2RGB))  # Display Fourier deblurred image in RGB

        ### Display Deblurred Image (Real Space): ###
        if deblurred_image_real is not None:
            plt.subplot(1, number_of_images, 3)  # Create subplot for real space deblurred image
            plt.title('Deblurred Image (Real Space)')  # Set title for real space deblurred image
            if deblurred_image_real.ndim == 2:  # If grayscale
                plt.imshow(deblurred_image_real, cmap='gray')  # Display real space deblurred image in grayscale
            else:  # If RGB
                plt.imshow(
                    cv2.cvtColor(deblurred_image_real, cv2.COLOR_BGR2RGB))  # Display real space deblurred image in RGB

        ### Show the Figure: ###
        plt.show()  # Display the figure with images

    @staticmethod
    def deconvolution_wrapper(image, psf, method, param=3, reg=None, num_iter=10, num_psf_iter=5, output_mode='both'):
        """
        Wrapper function to perform various deconvolution methods based on the input string or index.

        Args:
            image (np.ndarray): The observed (blurred and noisy) image with shape (H, W) or (H, W, C).
            psf (np.ndarray): Point spread function (PSF) with shape (kernel_H, kernel_W).
            method (str or int): Method to use for deconvolution. Options: 'wiener_skimage', 'wiener_opencv',
                                 'wiener_classic', 'unsupervised_wiener', 'richardson_lucy', 'richardson_lucy_skimage'.
            param (float or int): Regularization parameter for Wiener filtering or number of iterations for Richardson-Lucy.
                                  Default is 3.
            reg (np.ndarray or float, optional): Regularization parameter for Wiener filtering. Default is None.
            num_iter (int): Number of iterations for Richardson-Lucy deconvolution. Default is 10.
            num_psf_iter (int): Number of iterations for PSF estimation in Richardson-Lucy deconvolution. Default is 5.
            output_mode (str): Mode of output: 'bw', 'rgb', 'both'. Default is 'both'.

        Returns:
            dict: Dictionary containing 'deblurred_bw' and/or 'deblurred_rgb' based on the output_mode.
        """

        ### Define Single Channel Deblurring Function: ###
        def deblur_single_channel(channel):
            if method == 'wiener_skimage' or method == 0:
                return WienerDeconvolution.non_blind_deconvolution_wiener_skimage(channel, psf, param, reg)
            elif method == 'wiener_opencv' or method == 1:
                return WienerDeconvolution.non_blind_deconvolution_classic_wiener_opencv(channel, psf, noise=param)
            elif method == 'wiener_classic' or method == 2:
                return WienerDeconvolution.non_blind_deconvolution_classic_wiener(channel, psf, snr=param)
            elif method == 'unsupervised_wiener' or method == 3:
                return WienerDeconvolution.blind_deconvolution_unsupervised_wiener(channel, psf)
            elif method == 'richardson_lucy' or method == 4:
                return WienerDeconvolution.blind_non_blind_deconvolution_richardson_lucy(channel, psf, num_iter=param)
            elif method == 'richardson_lucy_skimage' or method == 5:
                return WienerDeconvolution.richardson_lucy_blind(channel, psf, num_iter=num_iter, num_psf_iter=num_psf_iter)
            else:
                raise ValueError("Invalid method. Choose from 'wiener_skimage', 'wiener_opencv', 'wiener_classic', "
                                 "'unsupervised_wiener', 'richardson_lucy', 'richardson_lucy_skimage'.")

        ### Initialize Output Dictionary: ###
        output_dict = {}

        ### Check and Process BW Image: ###
        flag_3channel_bw = False
        flag_3channel_rgb = False
        if len(image.shape) == 3:
            if image.shape[2] == 1:
                flag_3channel_bw = True
            elif image.shape[2] == 3:
                flag_3channel_rgb = True
        if len(image.shape) == 2 or flag_3channel_bw:  # If the image is BW
            deblurred_bw = deblur_single_channel(image)  # Deblur the BW image
            output_dict['deblurred_bw'] = deblurred_bw  # Add deblurred BW image to the output dictionary
            if output_mode == 'bw':  # If the output mode is 'bw'
                output_dict['deblurred_rgb'] = BW2RGB(deblurred_bw)  # Convert BW deblurred image to RGB and add to output dictionary
                return output_dict  # Return the output dictionary

        ### Check and Process RGB Image: ###
        if flag_3channel_rgb:  # If the image is RGB
            deblurred_channels = [deblur_single_channel(image[:, :, i]) for i in range(3)]  # Deblur each RGB channel separately
            deblurred_rgb = np.stack(deblurred_channels, axis=2)  # Stack the deblurred channels to form the RGB image
            if output_mode == 'rgb' or output_mode == 'both':  # If the output mode is 'rgb' or 'both'
                output_dict['deblurred_rgb'] = deblurred_rgb  # Add deblurred RGB image to the output dictionary
            if output_mode == 'both':  # If the output mode is 'both'
                deblurred_bw = deblur_single_channel(RGB2BW(image)[:, :, 0])  # Convert RGB to BW and deblur it
                output_dict['deblurred_bw'] = BW2RGB(deblurred_bw)  # Convert BW deblurred image to RGB and add to output dictionary
            if output_mode == 'bw':  # If the output mode is 'bw'
                deblurred_bw = deblur_single_channel(RGB2BW(image)[:, :, 0])  # Convert RGB to BW and deblur it
                output_dict['deblurred_rgb'] = BW2RGB(deblurred_bw)  # Convert BW deblurred image to RGB and add to output dictionary

        return output_dict  # Return the output dictionary

    @staticmethod
    def try_deconvolution_methods(input_dict,
                                  frames=None,
                                  blur_kernel=None,
                                  balance_list=None,
                                  snr_list=None,
                                  num_iter=10,
                                  num_psf_iter=5,
                                  output_mode='both',
                                  deblur_strength=1.0):
        """
        Try different balance values for non_blind_deconvolution_wiener_skimage and different SNR values for
        non_blind_deconvolution_classic_wiener, and return all results along with explanations.
        Also, include the outputs of blind_non_blind_deconvolution_richardson_lucy_skimage and blind_deconvolution_unsupervised_wiener.
        Can handle both BW and RGB images, and return results based on output_mode.
        The deblur_strength variable controls the amount of deblurring by blending the blur kernel with a Dirac delta function.

        Args:
            input_dict (dict): Dictionary containing input variables.
            frames (np.ndarray or list of np.ndarray, optional): The blurred input image or list of images.
            blur_kernel (np.ndarray, optional): The blur kernel (PSF) with shape (kernel_H, kernel_W).
            balance_list (list of float, optional): List of balance values to try for non_blind_deconvolution_wiener_skimage.
            snr_list (list of float, optional): List of SNR values to try for non_blind_deconvolution_classic_wiener.
            num_iter (int, optional): Number of iterations for Richardson-Lucy deconvolution. Default is 10.
            num_psf_iter (int, optional): Number of iterations for PSF estimation in Richardson-Lucy deconvolution. Default is 5.
            output_mode (str, optional): Output mode ('bw', 'rgb'). Default is 'bw'.
            deblur_strength (float, optional): Blending factor between 0 and 1 to control deblurring strength. Default is 1.0.

        Returns:
            dict: Dictionary containing 'deblurred_images' and 'explanations'.
        """

        ### Extract Inputs from input_dict or Use Default Values: ###
        frames = frames if frames is not None else input_dict.get('frames')
        blur_kernel = blur_kernel if blur_kernel is not None else input_dict.get('blur_kernel')
        balance_list = balance_list if balance_list is not None else input_dict.get('balance_list', [0.1, 0.5, 1, 2, 5, 10, 20, 50])
        snr_list = snr_list if snr_list is not None else input_dict.get('snr_list', [0.1, 0.5, 1, 2, 5, 10, 20, 50])
        num_iter = input_dict.get('num_iter', num_iter)
        num_psf_iter = input_dict.get('num_psf_iter', num_psf_iter)
        output_mode = input_dict.get('output_mode', output_mode)
        deblur_strength = input_dict.get('deblur_strength', deblur_strength)

        results = []  # List to store deblurred images
        explanations = []  # List to store explanations

        ### Ensure Frames is a List: ###
        if isinstance(frames, np.ndarray):
            frames = [frames]

        ### Determine Image Type: ###
        first_frame = frames[0]
        if len(first_frame.shape) == 2 or first_frame.shape[2] == 1:
            is_bw = True
        elif first_frame.shape[2] == 3:
            is_bw = False
        else:
            raise ValueError("Unsupported image format")

        # ### Blend Blur Kernel with Dirac Delta: ###
        # blur_kernel = blend_blur_kernel_with_dirac(blur_kernel, deblur_strength)

        ### Process Each Frame: ###
        for frame in frames:
            if is_bw or output_mode == 'bw':
                bw_frame = RGB2BW(frame) if not is_bw else frame
                bw_frame = bw_frame[:, :, 0] if len(bw_frame.shape) == 3 else bw_frame

                ### Looping Over Balance Values: ###
                for balance in balance_list:
                    deblurred_image = WienerDeconvolution.non_blind_deconvolution_wiener_skimage(bw_frame, blur_kernel, balance, clip=False, deblur_strength=deblur_strength)
                    deblurred_image_rgb = BW2RGB(deblurred_image)
                    results.append(deblurred_image_rgb)
                    explanations.append(f"Classic1 with balance={balance}")

                ### Looping Over SNR Values: ###
                for snr in snr_list:
                    deblurred_image = WienerDeconvolution.non_blind_deconvolution_classic_wiener(bw_frame, blur_kernel, snr, deblur_strength=deblur_strength)
                    deblurred_image_rgb = BW2RGB(deblurred_image)
                    results.append(deblurred_image_rgb)
                    explanations.append(f"Classic2 with SNR={snr}")

                ### Perform Richardson-Lucy Blind Deconvolution: ###
                deblurred_image_rl, _ = WienerDeconvolution.blind_non_blind_deconvolution_richardson_lucy_skimage(
                    bw_frame, blur_kernel, num_iter=num_iter, num_psf_iter=num_psf_iter)
                deblurred_image_rl_rgb = BW2RGB(deblurred_image_rl)
                results.append(deblurred_image_rl_rgb)
                explanations.append("Richardson-Lucy Blind")

                ### Perform Unsupervised Wiener Blind Deconvolution: ###
                deblurred_image_uw, _ = WienerDeconvolution.blind_deconvolution_unsupervised_wiener(bw_frame, blur_kernel, clip=False)
                deblurred_image_uw_rgb = BW2RGB(deblurred_image_uw)
                results.append(deblurred_image_uw_rgb)
                explanations.append("Unsupervised Wiener Blind")

            if not is_bw and output_mode == 'rgb':
                deblurred_rgb = np.zeros_like(frame)
                for c in range(3):
                    channel = frame[:, :, c]

                    ### Looping Over Balance Values: ###
                    for balance in balance_list:
                        deblurred_image = WienerDeconvolution.non_blind_deconvolution_wiener_skimage(channel, blur_kernel, balance, clip=False, deblur_strength=deblur_strength)
                        deblurred_rgb[:, :, c] = deblurred_image
                        # explanations.append(f"Deblurred using Wiener Skimage with balance={balance} (RGB Channel {c})")

                    ### Looping Over SNR Values: ###
                    for snr in snr_list:
                        deblurred_image = WienerDeconvolution.non_blind_deconvolution_classic_wiener(channel, blur_kernel, snr, deblur_strength=deblur_strength)
                        deblurred_rgb[:, :, c] = deblurred_image
                        # explanations.append(f"Deblurred using Classic Wiener with SNR={snr} (RGB Channel {c})")

                    ### Perform Richardson-Lucy Blind Deconvolution: ###
                    deblurred_image_rl, _ = WienerDeconvolution.blind_non_blind_deconvolution_richardson_lucy_skimage(
                        channel, blur_kernel, num_iter=num_iter, num_psf_iter=num_psf_iter)
                    deblurred_rgb[:, :, c] = deblurred_image_rl
                    # explanations.append(f"Deblurred using Richardson-Lucy Blind Deconvolution (RGB Channel {c})")

                    ### Perform Unsupervised Wiener Blind Deconvolution: ###
                    deblurred_image_uw, _ = WienerDeconvolution.blind_deconvolution_unsupervised_wiener(channel, blur_kernel, clip=False)
                    deblurred_rgb[:, :, c] = deblurred_image_uw
                    # explanations.append(f"Deblurred using Unsupervised Wiener Blind Deconvolution (RGB Channel {c})")

                explanations.append(f"Classic1 with balance={balance}")
                explanations.append(f"Classic2 with SNR={snr}")
                explanations.append(f"Richardson-Lucy Blind")
                explanations.append(f"Unsupervised Blind")
                results.append(deblurred_rgb)

        ### Prepare Output Dictionary: ###
        output_dict = {
            'deblurred_images': results,  # List of deblurred images
            'string_explanations': explanations  # List of explanations
        }

        return output_dict  # Return the output dictionary

    @staticmethod
    def get_default_SNRs():
        default_SNRs_list = [0.1, 0.5, 1, 2, 5, 10, 20, 50]
        return default_SNRs_list

    @staticmethod
    def get_default_balance_list():
        default_balance_list = [0.1, 0.5, 1, 2, 5, 10, 20, 50]
        return default_balance_list

    @staticmethod
    def augment_psf(psf, scale_factors=[0.9, 1.0, 1.1], rotation_degrees=[-5, 0, 5]):
        """
        Apply scaling and rotation augmentations to a PSF estimation.

        Parameters:
        psf (np.ndarray): The point spread function (PSF) with shape (H, W).
        scale_factors (list of float): List of scaling factors to apply. Default is [0.9, 1.0, 1.1].
        rotation_degrees (list of int): List of rotation degrees to apply. Default is [-5, 0, 5].

        Returns:
        tuple:
            - list of np.ndarray: List of augmented PSFs.
            - list of tuple: List of tuples with scaling factors and rotation angles.
        """

        ### This Is The Code Block: ###
        h, w = psf.shape  # Height and width of the PSF
        center = (w // 2, h // 2)  # Center of the PSF for rotation
        augmented_psfs = []  # List to store augmented PSFs
        augmentations = []  # List to store the scale and rotation augmentations

        ### Looping Over Indices: ###
        for scale in scale_factors:  # Loop over each scale factor
            for angle in rotation_degrees:  # Loop over each rotation angle
                ### Scaling the PSF: ###
                scaled_psf = cv2.resize(psf, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)  # Scale the PSF
                scaled_psf = crop_tensor(scaled_psf, (h,w))
                scaled_h, scaled_w = scaled_psf.shape  # Height and width of the scaled PSF
                pad_h = (h - scaled_h) // 2  # Padding height
                pad_w = (w - scaled_w) // 2  # Padding width
                scaled_psf = np.pad(scaled_psf, ((pad_h, h - scaled_h - pad_h), (pad_w, w - scaled_w - pad_w)),
                                    mode='constant')  # Pad the scaled PSF

                ### Rotating the PSF: ###
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)  # Rotation matrix for the given angle
                rotated_psf = cv2.warpAffine(scaled_psf, rotation_matrix, (w, h),
                                             flags=cv2.INTER_CUBIC)  # Rotate the scaled PSF

                ### Normalizing the PSF: ###
                rotated_psf /= rotated_psf.sum()  # Normalize the rotated PSF to ensure its sum is 1

                augmented_psfs.append(rotated_psf)  # Append the augmented PSF to the list
                augmentations.append((scale, angle))  # Append the scale and rotation to the list

        return augmented_psfs, augmentations  # Return the list of augmented PSFs and augmentations

    @staticmethod
    def plot_deconvolution_results(blurred_image, deblurred_images, explanations, single_figure=True):
        """
        Plot the original blurred image and all deblurred images with their explanations.

        Parameters:
        blurred_image (np.ndarray): The original blurred image with shape (H, W) or (H, W, C).
        deblurred_images (list of np.ndarray): List of deblurred images with shape (H, W) or (H, W, C).
        explanations (list of str): List of strings explaining the corresponding deblurred images.
        single_figure (bool): If True, plot all images in a single figure. If False, plot each image in a separate figure.
                              Default is True.
        """
        num_images = len(deblurred_images) + 1  # Total number of images to plot (including the original)

        if single_figure:
            ### Plot All Images in a Single Figure: ###
            plt.figure(figsize=(20, 5))  # Set figure size

            ### Plot Original Blurred Image: ###
            num_images_per_col = int(np.ceil(np.sqrt(num_images)))  # Number of images per column
            plt.subplot(num_images_per_col, num_images_per_col, 1)  # Create subplot for original blurred image
            plt.title("Original Blurred Image")  # Set title for original blurred image
            if blurred_image.ndim == 2:  # If grayscale
                plt.imshow(blurred_image, cmap='gray')  # Display original blurred image in grayscale
            else:  # If RGB
                plt.imshow(blurred_image)  # Display original blurred image in RGB
            plt.axis('off')  # Turn off axis

            ### Looping Over Indices: ###
            for i, (deblurred_image, explanation) in enumerate(zip(deblurred_images, explanations),
                                                               start=2):  # Loop over each deblurred image
                num_images_per_col = int(np.ceil(np.sqrt(num_images)))  # Number of images per column
                plt.subplot(num_images_per_col, num_images_per_col, i)  # Create subplot for each deblurred image
                plt.title(explanation)  # Set title for each deblurred image
                if deblurred_image.ndim == 2:  # If grayscale
                    plt.imshow(deblurred_image, cmap='gray')  # Display deblurred image in grayscale
                else:  # If RGB
                    plt.imshow(deblurred_image)  # Display deblurred image in RGB
                plt.axis('off')  # Turn off axis

            plt.tight_layout()  # Adjust subplots to fit into figure area
            plt.show()  # Display the figure

        else:
            ### Plot Each Image in Separate Figures: ###
            plt.figure(figsize=(6, 6))  # Set figure size for original blurred image
            plt.title("Original Blurred Image")  # Set title for original blurred image
            if blurred_image.ndim == 2:  # If grayscale
                plt.imshow(blurred_image, cmap='gray')  # Display original blurred image in grayscale
            else:  # If RGB
                plt.imshow(blurred_image)  # Display original blurred image in RGB
            plt.axis('off')  # Turn off axis
            plt.show()  # Display the figure

            ### Looping Over Indices: ###
            for deblurred_image, explanation in zip(deblurred_images, explanations):  # Loop over each deblurred image
                plt.figure(figsize=(6, 6))  # Set figure size for each deblurred image
                plt.title(explanation)  # Set title for each deblurred image
                if deblurred_image.ndim == 2:  # If grayscale
                    plt.imshow(deblurred_image, cmap='gray')  # Display deblurred image in grayscale
                else:  # If RGB
                    plt.imshow(deblurred_image)  # Display deblurred image in RGB
                plt.axis('off')  # Turn off axis
                plt.show()  # Display the figure

    @staticmethod
    def get_blur_kernels_and_deblurred_images_using_NUBKE(input_dict,
                                                          frames=None,
                                                          input_method=None,
                                                          user_input=None,
                                                          n_iters=30,
                                                          SAVE_INTERMIDIATE=True,
                                                          saturation_threshold=0.99,
                                                          K_number_of_base_elements=25,
                                                          reg_factor=1e-3,
                                                          optim_iters=1e-6,
                                                          gamma_correction_factor=2.2,
                                                          apply_dilation=False,
                                                          apply_smoothing=False,  #TODO: was true
                                                          apply_erosion=False,  #TODO: was true
                                                          flag_use_avg_kernel_on_everything=False):
        """
        Perform NUBKE deconvolution on a list of blurred images to obtain blur kernels and deblurred images.

        Args:
            input_dict (dict): Dictionary containing input variables.
            frames (list of np.ndarray, optional): List of blurred images.
            input_method (str, optional): Method for input handling. Default is 'BB_XYXY'.
            user_input (any, optional): User input data.
            n_iters (int, optional): Number of iterations for deconvolution. Default is 30.
            SAVE_INTERMIDIATE (bool, optional): Flag to save intermediate results. Default is True.
            saturation_threshold (float, optional): Saturation threshold. Default is 0.99.
            K_number_of_base_elements (int, optional): Number of base elements for the kernel. Default is 25.
            reg_factor (float, optional): Regularization factor. Default is 1e-3.
            optim_iters (float, optional): Optimization iterations. Default is 1e-6.
            gamma_correction_factor (float, optional): Gamma correction factor. Default is 2.2.
            apply_dilation (bool, optional): Flag to apply dilation. Default is False.
            apply_smoothing (bool, optional): Flag to apply smoothing. Default is True.
            apply_erosion (bool, optional): Flag to apply erosion. Default is True.
            flag_use_avg_kernel_on_everything (bool, optional): Flag to use average kernel on everything. Default is False.

        Returns:
            dict: Dictionary containing 'average_blur_kernels_list', 'deblurred_crops_list', and 'frames'.
        """

        ### Extract Inputs from input_dict or Use Default Values: ###
        frames = input_dict.get('frames', frames)
        input_method = input_dict.get('input_method', input_method if input_method is not None else 'BB_XYXY')
        user_input = input_dict.get('user_input', user_input)
        n_iters = int(input_dict.get('params', {}).get('n_iters', n_iters))
        SAVE_INTERMIDIATE = input_dict.get('SAVE_INTERMIDIATE', SAVE_INTERMIDIATE)
        saturation_threshold = input_dict.get('saturation_threshold', saturation_threshold)
        K_number_of_base_elements = input_dict.get('K_number_of_base_elements', K_number_of_base_elements)
        reg_factor = input_dict.get('reg_factor', reg_factor)
        optim_iters = input_dict.get('optim_iters', optim_iters)
        gamma_correction_factor = input_dict.get('gamma_correction_factor', gamma_correction_factor)
        apply_dilation = input_dict.get('apply_dilation', apply_dilation)
        apply_smoothing = input_dict.get('apply_smoothing', apply_smoothing)
        apply_erosion = input_dict.get('apply_erosion', apply_erosion)
        flag_use_avg_kernel_on_everything = input_dict.get('flag_use_avg_kernel_on_everything', flag_use_avg_kernel_on_everything)

        ### Ensure Frames is a List: ###
        if isinstance(frames, np.ndarray):  # Check if frames is a single array
            frames = [frames]  # Convert to list

        ### Get Bounding Box Dimensions: ###
        h, w = frames[0].shape[:2]  # Get height and width from the first frame

        ### Convert User Input to All Input Types: ###
        BB_XYXY, polygon_points, segmentation_mask, grid_points, flag_no_input, flag_list = user_input_to_all_input_types(
            user_input, input_method=input_method, input_shape=(h, w))

        ### Initialize Output Lists: ###
        average_blur_kernels_list = []  # Initialize list to store average blur kernels
        average_blur_kernels_threholded_list = []  # Initialize list to store average blur kernels
        average_blur_kernels_straight_line_list = []  # Initialize list to store average blur kernels
        deblurred_crops_list = []  # Initialize list to store deblurred crops
        deblurred_images_list = []  # Initialize list to store deblurred images
        images_with_blur_kernel_on_it_list = []
        images_with_blur_kernel_thresholded_on_it_list = []
        images_with_blur_kernel_straight_line_on_it_list = []

        ### Loop Over Frames and Perform NUBKE Deconvolution: ###
        for current_blurred_image in frames:  # Iterate over each blurred image
            try:
                ### Convert Current Blurred Image to Torch Tensor: ###
                blurred_image_torch = numpy_to_torch(current_blurred_image).cuda()  # Convert current blurred image to torch tensor
                segmentation_mask_torch = numpy_to_torch(segmentation_mask).cuda()  # Convert segmentation mask to torch tensor

                ### Perform NUBKE Deconvolution: ###
                (blur_kernel_torch,
                 blur_kernel_torch_thresholded,
                 blur_kernel_torch_straight_line,
                 K_kernels_basis_tensor,
                 deblurred_image, deblurred_crop,
                 image_with_blur_kernels_on_it,
                 image_with_blur_kernels_thresholded_on_it,
                 image_with_blur_kernels_straight_line_on_it) = get_blur_kernel_and_deblurred_image_using_NUBKE(
                    blurred_image_torch,
                    segmentation_mask_torch=segmentation_mask_torch,
                    NUBKE_model=None,
                    device='cuda',
                    n_iters=n_iters,
                    SAVE_INTERMIDIATE=SAVE_INTERMIDIATE,
                    saturation_threshold=saturation_threshold,
                    K_number_of_base_elements=K_number_of_base_elements,
                    reg_factor=reg_factor,
                    optim_iters=optim_iters,
                    gamma_correction_factor=gamma_correction_factor,
                    apply_dilation=apply_dilation,
                    apply_smoothing=apply_smoothing,
                    apply_erosion=apply_erosion,
                    flag_use_avg_kernel_on_everything=flag_use_avg_kernel_on_everything)
                average_blur_kernel = torch_to_numpy(blur_kernel_torch)  # Convert blur kernel to numpy array
                average_blur_kernel_thresholded = torch_to_numpy(blur_kernel_torch_thresholded)  # Convert blur kernel to numpy array
                average_blur_kernel_straight_line = torch_to_numpy(blur_kernel_torch_straight_line)  # Convert blur kernel to numpy array

                ### Append Results to Lists: ###
                average_blur_kernels_list.append(average_blur_kernel)  # Add blur kernel to list
                average_blur_kernels_threholded_list.append(average_blur_kernel_thresholded)  # Add blur kernel to list
                average_blur_kernels_straight_line_list.append(average_blur_kernel_straight_line)  # Add blur kernel to list
                deblurred_crops_list.append(torch_to_numpy(deblurred_crop))  # Add deblurred crop to list
                deblurred_images_list.append(torch_to_numpy(deblurred_image[0]))  # Add deblurred image to list
                images_with_blur_kernel_on_it_list.append(image_with_blur_kernels_on_it.transpose([1,2,0]))
                images_with_blur_kernel_thresholded_on_it_list.append(image_with_blur_kernels_thresholded_on_it.transpose([1,2,0]))
                images_with_blur_kernel_straight_line_on_it_list.append(image_with_blur_kernels_straight_line_on_it.transpose([1,2,0]))
            except Exception as e:
                bla = 1
        ### Prepare Output Dictionary: ###
        output_dict = {
            'average_blur_kernels_list': average_blur_kernels_list,  # List of average blur kernels
            'average_blur_kernels_threholded_list': average_blur_kernels_threholded_list,  # List of average blur kernels
            'average_blur_kernels_straight_line_list': average_blur_kernels_straight_line_list,  # List of average blur kernels
            'images_with_blur_kernel_on_it_list': images_with_blur_kernel_on_it_list,  # List of average blur kernels
            'images_with_blur_kernel_thresholded_on_it_list': images_with_blur_kernel_thresholded_on_it_list,  # List of average blur kernels
            'images_with_blur_kernel_straight_line_on_it_list': images_with_blur_kernel_straight_line_on_it_list,  # List of average blur kernels
            'deblurred_crops_list': deblurred_crops_list,  # List of deblurred crops
            'deblurred_images_list': deblurred_images_list,  # List of deblurred crops
            'frames': deblurred_crops_list  # List of deblurred images
        }

        return output_dict  # Return the output dictionary


    @staticmethod
    def get_blur_kernel_on_image_using_NUBKE(blurred_image,
                                             input_method=None,
                                             user_input=None):
        ### Get Region From User: ###
        H, W = blurred_image.shape[0:2]
        initial_BB_XYXY, initial_polygon_points, initial_segmentation_mask, grid_points, flag_no_input, flag_list = user_input_to_all_input_types(
            user_input,
            input_method=input_method,
            input_shape=(H, W))

        ### Get Deblurred Image Using NUBKE: ###
        blurred_image_torch = numpy_to_torch(blurred_image)
        initial_segmentation_mask_torch = numpy_to_torch(initial_segmentation_mask)
        blur_kernel_torch, K_kernels_basis_tensor, deblurred_image, deblurred_crop = get_blur_kernel_and_deblurred_image_using_NUBKE(
            blurred_image_torch,
            segmentation_mask_torch=initial_segmentation_mask_torch,
            NUBKE_model=None,
            device='cuda',
            n_iters=30,
            SAVE_INTERMIDIATE=True,
            saturation_threshold=0.99,
            K_number_of_base_elements=25,
            # TODO: probably doesn't need this because it is only necessary for model initialization and visualizations
            reg_factor=1e-3,
            optim_iters=1e-6,
            gamma_correction_factor=2.2,
            apply_dilation=False,
            apply_smoothing=True,
            apply_erosion=True,
            flag_use_avg_kernel_on_everything=False)
        average_blur_kernel = torch_to_numpy(blur_kernel_torch)

        return average_blur_kernel

    @staticmethod
    def get_blur_kernel_and_deblurred_image_using_NUBKE(blurred_image,
                                                        input_method=None,
                                                        user_input=None):
        ### Get Region From User: ###
        H,W = blurred_image.shape[0:2]
        initial_BB_XYXY, initial_polygon_points, initial_segmentation_mask, grid_points, flag_no_input, flag_list = user_input_to_all_input_types(
            user_input,
            input_method=input_method,
            input_shape=(H, W))

        ### Get Deblurred Image Using NUBKE: ###
        blurred_image_torch = numpy_to_torch(blurred_image)
        initial_segmentation_mask_torch = numpy_to_torch(initial_segmentation_mask)
        # blur_kernel_torch, K_kernels_basis_tensor, deblurred_image, deblurred_crop  #TODO: this was the original output
        (avg_kernel, avg_kernel_thresholded, avg_kernel_straight_line,
         K_kernels_basis_tensor, deblurred_image, deblurred_crop,
         image_with_blur_kernels_on_it, image_with_blur_kernels_thresholded_on_it,
         image_with_blur_kernels_straight_line_on_it) = get_blur_kernel_and_deblurred_image_using_NUBKE(
            blurred_image_torch,
            segmentation_mask_torch=initial_segmentation_mask_torch,
            NUBKE_model=None,
            device='cuda',
            n_iters=30,
            SAVE_INTERMIDIATE=True,
            saturation_threshold=0.99,
            K_number_of_base_elements=25, # TODO: probably doesn't need this because it is only necessary for model initialization and visualizations
            reg_factor=1e-3,
            optim_iters=1e-6,
            gamma_correction_factor=2.2,
            apply_dilation=False,
            apply_smoothing=True,
            apply_erosion=True,
            flag_use_avg_kernel_on_everything=False)
        average_blur_kernel = torch_to_numpy(avg_kernel)  #TODO: this originally was blur_kernel_torch

        return average_blur_kernel, deblurred_crop

    @staticmethod
    def get_blur_kernel_using_NUBKE_and_deblur_using_Wiener_all_options(input_dict: dict,
                                                                        blurred_image=None,
                                                                        input_method=None,
                                                                        user_input=None,
                                                                        flag_plot=False):
        """
        Get blur kernel using NUBKE and deblur using all Wiener options.

        Args:
            input_dict (dict): Dictionary containing input variables.
            blurred_image (np.ndarray, optional): The blurred input image with shape (H, W) or (H, W, C).
            input_method (str, optional): Method to get user input ('BB_XYXY', 'polygon', 'segmentation'). Default is 'BB_XYXY'.
            user_input (any, optional): User input for the selected input method. Default is None.
            flag_plot (bool, optional): Flag to plot the results. Default is False.

        Returns:
            dict: Dictionary containing 'deblurred_images' and 'string_explanations'.
        """
        ### Extract Inputs from input_dict or Use Default Values: ###
        blurred_image = input_dict.get('frames', blurred_image)
        input_method = input_dict.get('input_method', input_method if input_method != 'BB_XYXY' else 'BB_XYXY')
        user_input = input_dict.get('user_input', user_input)
        flag_plot = input_dict.get('flag_plot', flag_plot if flag_plot is not False else False)

        blurred_image = blurred_image[0]

        ### Get Bounding Box Dimensions: ###
        H, W = blurred_image.shape[:2]

        ### Convert User Input to All Input Types: ###
        BB_XYXY, polygon_points, segmentation_mask, grid_points, flag_no_input, flag_list = user_input_to_all_input_types(
            user_input, input_method=input_method, input_shape=(H, W))

        ### Get Average Blur Kernel Using NUBKE: ###
        input_dict_nubke = {
            'frames': blurred_image,
            'input_method': 'BB_XYXY',
            'user_input': None
        }
        average_blur_kernel, deblurred_crop = WienerDeconvolution.get_blur_kernel_and_deblurred_image_using_NUBKE(
            blurred_image=input_dict_nubke['frames'],
            input_method=input_dict_nubke['input_method'],
            user_input=input_dict_nubke['user_input'])

        ### Insert Blur Kernel into Wiener Deconvolution: ###
        balance_list = WienerDeconvolution.get_default_balance_list()
        snr_list = WienerDeconvolution.get_default_SNRs()
        blurred_image_BW = RGB2BW(blurred_image)[:, :, 0]

        input_dict_wiener = {
            'frames': blurred_image_BW,
            'blur_kernel': average_blur_kernel,
            'balance_list': balance_list,
            'snr_list': snr_list,
            'num_iter': 10,
            'num_psf_iter': 5
        }
        output_dict_wiener = WienerDeconvolution.try_deconvolution_methods(input_dict_wiener)
        deblurred_images = output_dict_wiener['deblurred_images']
        string_explanations = output_dict_wiener['string_explanations']

        ### Get Deblurred Crop: ###
        X0, Y0, X1, Y1 = BB_XYXY
        deblurred_crops_list = []
        blurred_crops_list = []
        for i in np.arange(len(deblurred_images)):
            blurred_crop = blurred_image[Y0:Y1, X0:X1]
            deblurred_crop = deblurred_images[i][Y0:Y1, X0:X1]
            deblurred_crops_list.append(deblurred_crop)
            blurred_crops_list.append(blurred_crop)

        ### Debug Results: ###
        if flag_plot:
            WienerDeconvolution.plot_deconvolution_results(blurred_crops_list, deblurred_crops_list,
                                                           string_explanations, single_figure=False)

        ### Prepare Output Dictionary: ###
        output_dict = {
            'frames': deblurred_images,
            'deblurred_images': deblurred_images,
            'string_explanations': string_explanations,
            'blur_kernel': average_blur_kernel
        }

        return output_dict

    @staticmethod
    def get_deblurred_image_from_blur_kernel_using_Wiener_all_parameter_options(input_dict,
                                                                      frames=None,
                                                                      blur_kernel=None,
                                                                      input_method=None,
                                                                      user_input=None,
                                                                      flag_plot=False):
        """
        Perform Wiener deconvolution on a blurred image or a list of blurred images using all available options.

        Args:
            input_dict (dict): Dictionary containing input variables.
            frames (np.ndarray or list of np.ndarray, optional): Blurred image or list of blurred images.
            blur_kernel (np.ndarray, optional): Initial blur kernel.
            input_method (str, optional): Method for input handling. Default is 'BB_XYXY'.
            user_input (any, optional): User input data.
            flag_plot (bool, optional): Flag to plot results. Default is False.

        Returns:
            dict: Dictionary containing 'deblurred_images', 'string_explanations', 'deblurred_crops_list', and 'blurred_crops_list'.
        """

        ### Extract Inputs from input_dict or Use Default Values: ###
        frames = input_dict.get('frames', frames)
        blur_kernel = input_dict.get('blur_kernel', blur_kernel)
        input_method = input_dict.get('input_method', input_method if input_method is not None else 'BB_XYXY')
        user_input = input_dict.get('user_input', user_input)
        flag_plot = input_dict.get('flag_plot', flag_plot)

        ### Parameter Lists: ###
        balance_list = input_dict.get('balance_list', None)
        snr_list = input_dict.get('snr_list', None)

        ### Ensure Frames is a List: ###
        if isinstance(frames, np.ndarray):  # Check if frames is a single array
            frames = [frames]  # Convert to list

        ### Get Bounding Box Dimensions: ###
        h, w = frames[0].shape[:2]  # Get height and width from the first frame

        ### Convert User Input to All Input Types: ###
        initial_BB_XYXY, initial_polygon_points, initial_segmentation_mask, initial_grid_points, flag_no_input, flag_list = user_input_to_all_input_types(
            user_input, input_method=input_method, input_shape=(h, w))

        ### Initialize Output Lists: ###
        deblurred_images_list = []  # Initialize list to store deblurred images
        deblurred_crops_list = []  # Initialize list to store deblurred crops
        blurred_crops_list = []  # Initialize list to store blurred crops

        ### Perform Wiener Deconvolution: ###
        if balance_list is None:
            balance_list = WienerDeconvolution.get_default_balance_list()  # Get default balance list
        if snr_list is None:
            snr_list = WienerDeconvolution.get_default_SNRs()  # Get default SNR list
        blurred_image_BW = RGB2BW(frames[0])[:, :, 0]  # Convert blurred image to BW
        output_dict = WienerDeconvolution.try_deconvolution_methods({},
                                                                    blurred_image_BW,
                                                                    blur_kernel,
                                                                    balance_list=balance_list,
                                                                    snr_list=snr_list,
                                                                    num_iter=5,
                                                                    num_psf_iter=5)
        deblurred_images = output_dict['deblurred_images']
        string_explanations = output_dict['string_explanations']

        ### Get Deblurred Crop: ###
        X0, Y0, X1, Y1 = initial_BB_XYXY  # Extract bounding box coordinates
        for i in range(len(deblurred_images)):  # Loop over each deblurred image
            blurred_crop = frames[0][Y0:Y1, X0:X1]  # Get blurred crop
            deblurred_crop = deblurred_images[i][Y0:Y1, X0:X1]  # Get deblurred crop
            deblurred_crops_list.append(deblurred_crop)  # Add deblurred crop to the list
            blurred_crops_list.append(blurred_crop)  # Add blurred crop to the list

        ### Plot Results if Flag is True: ###
        if flag_plot:  # If plot flag is set
            WienerDeconvolution.plot_deconvolution_results(blurred_crops_list, deblurred_crops_list, string_explanations, single_figure=False)

        ### Get frames with string explanations on top of them: ###
        frames_with_text = add_text_to_images(deblurred_crops_list, string_explanations, font_size=1, font_thickness=2)

        ### Prepare Output Dictionary: ###
        output_dict = {
            'frames': deblurred_images,  # List of deblurred images
            'frames_with_text': frames_with_text,  # List of deblurred images
            'deblurred_images': deblurred_images,  # List of deblurred images
            'string_explanations': string_explanations,  # Explanations of deconvolution methods
            'deblurred_crops_list': deblurred_crops_list,  # List of deblurred crops
            'blurred_crops_list': blurred_crops_list  # List of blurred crops
        }

        return output_dict  # Return the output dictionary


    @staticmethod
    def get_deblurred_image_from_blur_kernel_using_Wiener_all_options(input_dict,
                                                                      frames=None,
                                                                      blur_kernel=None,
                                                                      input_method=None,
                                                                      user_input=None,
                                                                      flag_plot=False):
        """
        Perform Wiener deconvolution on a blurred image or a list of blurred images using all available options.

        Args:
            input_dict (dict): Dictionary containing input variables.
            frames (np.ndarray or list of np.ndarray, optional): Blurred image or list of blurred images.
            blur_kernel (np.ndarray, optional): Initial blur kernel.
            input_method (str, optional): Method for input handling. Default is 'BB_XYXY'.
            user_input (any, optional): User input data.
            flag_plot (bool, optional): Flag to plot results. Default is False.

        Returns:
            dict: Dictionary containing 'deblurred_images', 'string_explanations', 'deblurred_crops_list', and 'blurred_crops_list'.
        """

        ### Extract Inputs from input_dict or Use Default Values: ###
        frames = input_dict.get('frames', frames)
        blur_kernel = input_dict.get('blur_kernel', blur_kernel)
        input_method = input_dict.get('input_method', input_method if input_method is not None else 'BB_XYXY')
        user_input = input_dict.get('user_input', user_input)
        flag_plot = input_dict.get('flag_plot', flag_plot)

    @staticmethod
    def get_deblurred_image_from_blur_kernel_using_Wiener_specific_option(input_dict,
                                                                          frames=None,
                                                                          blur_kernel=None,
                                                                          wiener_method=None,
                                                                          wiener_parameter=5,  # SNR or balance parameter
                                                                          input_method=None,
                                                                          user_input=None,
                                                                          flag_plot=False,
                                                                          num_iter=10,
                                                                          num_psf_iter=5):
        """
        Perform Wiener deconvolution on a blurred image or a list of blurred images using a specific method.

        Args:
            input_dict (dict): Dictionary containing input variables.
            frames (np.ndarray or list of np.ndarray, optional): Blurred image or list of blurred images.
            blur_kernel (np.ndarray, optional): Initial blur kernel.
            wiener_method (str, optional): Method for Wiener deconvolution.
            wiener_parameter (float, optional): SNR or balance parameter. Default is 5.
            input_method (str, optional): Method for input handling. Default is 'BB_XYXY'.
            user_input (any, optional): User input data.
            flag_plot (bool, optional): Flag to plot results. Default is False.
            num_iter (int, optional): Number of iterations for deconvolution. Default is 10.
            num_psf_iter (int, optional): Number of PSF iterations. Default is 5.

        Returns:
            dict: Dictionary containing 'deblurred_images', 'blur_kernel', 'frames', and 'deblurred_images_only_crop_deblurred'.
        """

        ### Extract Inputs from input_dict or Use Default Values: ###
        frames = input_dict.get('frames', frames)
        user_input = input_dict.get('user_input', user_input)
        input_method = input_dict.get('input_method', input_method if input_method is not None else 'BB_XYXY')
        blur_kernel = input_dict.get('blur_kernel', blur_kernel)
        wiener_method = input_dict.get('params', {}).get('wiener_method', wiener_method)
        wiener_parameter = input_dict.get('params', {}).get('wiener_parameter', wiener_parameter)
        num_iter = input_dict.get('params', {}).get('num_iter', num_iter)
        num_psf_iter = input_dict.get('params', {}).get('num_psf_iter', num_psf_iter)
        flag_plot = input_dict.get('flag_plot', flag_plot)

        ### Ensure Frames is a List: ###
        if isinstance(frames, np.ndarray):  # Check if frames is a single array
            frames = [frames]  # Convert to list

        ### Get Bounding Box Dimensions: ###
        h, w = frames[0].shape[:2]  # Get height and width from the first frame

        ### Convert User Input to All Input Types: ###
        initial_BB_XYXY, initial_polygon_points, initial_segmentation_mask, initial_grid_points, flag_no_input, flag_list = user_input_to_all_input_types(
            user_input, input_method=input_method, input_shape=(h, w))

        deblurred_images = []  # Initialize list to store deblurred images
        deblurred_images_only_crop_deblurred = []  # Initialize list to store cropped deblurred images

        ### Loop Over Frames: ###
        for blurred_image in frames:  # Iterate over each blurred image
            ### Convert to Grayscale: ###
            blurred_image_BW = RGB2BW(blurred_image)[:, :, 0]  # Convert to grayscale

            ### Perform Wiener Deconvolution: ###
            deblurred_image = WienerDeconvolution.deconvolution_wrapper(blurred_image_BW,
                                                                        blur_kernel,
                                                                        method=wiener_method,
                                                                        param=wiener_parameter,
                                                                        reg=None,
                                                                        num_iter=num_iter,
                                                                        num_psf_iter=num_psf_iter)
            deblurred_images.append(deblurred_image)  # Add the deblurred image to the list

            ### Get Deblurred Crop: ###
            X0, Y0, X1, Y1 = initial_BB_XYXY  # Extract bounding box coordinates
            deblurred_crop = deblurred_image[Y0:Y1, X0:X1]  # Get the deblurred crop

            ### Create Deblurred Image with Only Crop Deblurred: ###
            deblurred_image_cropped = blurred_image.copy()  # Copy the original blurred image
            deblurred_image_cropped[Y0:Y1, X0:X1] = RGB2BW(deblurred_crop)  # Replace the crop area with deblurred crop
            deblurred_images_only_crop_deblurred.append(deblurred_image_cropped)  # Add the cropped deblurred image to the list

        ### Prepare Output Dictionary: ###
        output_dict = {
            'deblurred_images': deblurred_images,  # List of deblurred images
            'blur_kernel': blur_kernel,  # Blur kernel
            'frames': deblurred_images_only_crop_deblurred,  # Main output of the function
            'deblurred_images_only_crop_deblurred': deblurred_images_only_crop_deblurred  # List of images with only crop deblurred
        }

        return output_dict  # Return the output dictionary

    @staticmethod
    def create_dirac_delta_image(shape):
        """
        Create a Dirac delta image with a single one at the center and zeros elsewhere.

        Args:
            shape (tuple): Shape of the Dirac delta image (height, width).

        Returns:
            np.ndarray: Dirac delta image.
        """
        delta_image = np.zeros(shape, dtype=np.float32)
        center = (shape[0] // 2, shape[1] // 2)
        delta_image[center] = 1.0
        return delta_image

    @staticmethod
    def blend_blur_kernel_with_dirac(blur_kernel, alpha):
        """
        Blend the estimated blur kernel with a Dirac delta function based on the alpha value.

        Args:
            blur_kernel (np.ndarray): Estimated blur kernel.
            alpha (float): Blending factor between 0 and 1.

        Returns:
            np.ndarray: Blended and normalized blur kernel.
        """
        dirac_delta = WienerDeconvolution.create_dirac_delta_image(blur_kernel.shape)
        blended_kernel = alpha * blur_kernel + (1 - alpha) * dirac_delta
        normalized_kernel = blended_kernel / np.sum(blended_kernel)
        return normalized_kernel

    @staticmethod
    def generate_gaussian_kernel(size=11, sigma=2):
        """
        Generate a Gaussian blur kernel.

        Args:
            size (int, optional): Size of the kernel (must be an odd number). Default is 11.
            sigma (float, optional): Standard deviation of the Gaussian. Default is 2.

        Returns:
            np.ndarray: Gaussian blur kernel.
        """
        ### Create a Coordinate Grid: ###
        k = (size - 1) // 2  # Calculate the kernel radius
        x, y = np.mgrid[-k:k + 1, -k:k + 1]  # Generate coordinate grid

        ### Calculate Gaussian Function: ###
        kernel = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))  # Apply Gaussian formula
        kernel /= kernel.sum()  # Normalize the kernel to ensure the sum is 1

        return kernel  # Return the Gaussian kernel

    @staticmethod
    def get_blur_kernel_and_deblurred_image_unsupervised_wiener(input_dict,
                                                                frames=None,
                                                                blur_kernel=None,
                                                                input_method=None,
                                                                user_input=None):
        """
        Perform unsupervised Wiener deconvolution on a blurred image or a list of blurred images.

        Args:
            input_dict (dict): Dictionary containing input variables.
            frames (np.ndarray or list of np.ndarray, optional): Blurred image or list of blurred images.
            blur_kernel (np.ndarray, optional): Initial blur kernel. Default is None.
            input_method (str, optional): Method for input handling. Default is 'BB_XYXY'.
            user_input (any, optional): User input data.

        Returns:
            dict: Dictionary containing 'deblurred_images', 'blur_kernel', 'frames', and 'deblurred_images_only_crop_deblurred'.
        """

        ### Extract Inputs from input_dict or Use Default Values: ###
        blur_kernel_default = WienerDeconvolution.generate_gaussian_kernel(11, 2)
        frames = input_dict.get('frames', frames)
        blur_kernel = input_dict.get('blur_kernel', blur_kernel if blur_kernel is not None else blur_kernel_default)
        input_method = input_dict.get('input_method', input_method)
        user_input = input_dict.get('user_input', user_input)

        ### Ensure Frames is a List: ###
        if isinstance(frames, np.ndarray):  # Check if frames is a single array
            frames = [frames]  # Convert to list

        ### Get Bounding Box Dimensions: ###
        h, w = frames[0].shape[:2]  # Get height and width from reference frame shape

        ### Convert User Input to All Input Types: ###
        initial_BB_XYXY, initial_polygon_points, initial_segmentation_mask, initial_grid_points, flag_no_input, flag_list = user_input_to_all_input_types(
            user_input, input_method=input_method, input_shape=(h, w))

        deblurred_images = []  # Initialize list to store deblurred images
        deblurred_images_only_crop_deblurred = []  # Initialize list to store cropped deblurred images
        final_blur_kernels = []  # Initialize list to store final blur kernels

        ### Loop Over Frames: ###
        for blurred_image in frames:  # Iterate over each blurred image
            ### Perform Unsupervised Wiener Deconvolution: ###
            deblurred_image, blur_kernel_output = WienerDeconvolution.blind_deconvolution_unsupervised_wiener(blurred_image,
                                                                                                              psf=blur_kernel,
                                                                                                              use_bw=True,
                                                                                                              deblur_strength=0.9)
            deblurred_images.append(deblurred_image)  # Add the deblurred image to the list
            final_blur_kernels.append(blur_kernel_output)  # Add the blur kernel to the list

            ### Create Deblurred Image with Only Crop Deblurred: ###
            crop_x0, crop_y0, crop_x1, crop_y1 = initial_BB_XYXY  # Extract bounding box coordinates
            deblurred_image_cropped = blurred_image.copy().astype('float')  # Copy the original blurred image
            deblurred_image_cropped[crop_y0:crop_y1, crop_x0:crop_x1] = deblurred_image[crop_y0:crop_y1, crop_x0:crop_x1]  # Replace the crop area with deblurred crop
            deblurred_images_only_crop_deblurred.append(deblurred_image_cropped)  # Add the cropped deblurred image to the list

        # bla = WienerDeconvolution.erode_blob_edges(np.clip(deblurred_image,0,255).astype(np.uint8))
        # bla = WienerDeconvolution.erode_blob_edges_rgb(np.clip(deblurred_image,0,255).astype(np.uint8))
        # bla = WienerDeconvolution.soft_erode_blob_edges(np.clip(deblurred_image,0,255).astype(np.uint8))
        # bla = WienerDeconvolution.soft_erode_blob_edges_bilateral(np.clip(deblurred_image,0,255).astype(np.uint8))
        # bla = WienerDeconvolution.thin_blob_edges(np.clip(deblurred_image,0,255).astype(np.uint8))
        # plt.imshow(bla)
        # plt.imshow(deblurred_image/255)
        ### Prepare Output Dictionary: ###
        output_dict = {
            'frames': deblurred_images_only_crop_deblurred,  # Main output of the function
            'deblurred_images': deblurred_images,  # List of deblurred images
            'blur_kernel': final_blur_kernels,  # Blur kernel or list of blur kernels
            'deblurred_images_only_crop_deblurred': deblurred_images_only_crop_deblurred  # List of images with only crop deblurred
        }

        return output_dict  # Return the output dictionary

    @staticmethod
    def thin_blob_edges(image, low_threshold=100, high_threshold=200):
        """
        Thin the edges of blobs in an RGB image using edge detection and morphological thinning.

        Args:
            image (np.ndarray): Input RGB image with shape (H, W, 3).
            low_threshold (int, optional): Lower threshold for the Canny edge detector. Default is 100.
            high_threshold (int, optional): Upper threshold for the Canny edge detector. Default is 200.

        Returns:
            np.ndarray: Image with thinned blob edges.
        """
        ### Convert RGB Image to Grayscale: ###
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

        ### Detect Edges Using Canny Edge Detector: ###
        edges = cv2.Canny(gray_image, low_threshold, high_threshold)  # Detect edges

        ### Morphological Thinning: ###
        def morphological_thinning(image):
            """
            Apply morphological thinning to the binary image.

            Args:
                image (np.ndarray): Input binary image.

            Returns:
                np.ndarray: Thinned binary image.
            """
            size = np.size(image)
            skel = np.zeros(image.shape, np.uint8)

            element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
            done = False

            while not done:
                eroded = cv2.erode(image, element)
                temp = cv2.dilate(eroded, element)
                temp = cv2.subtract(image, temp)
                skel = cv2.bitwise_or(skel, temp)
                image = eroded.copy()

                zeros = size - cv2.countNonZero(image)
                if zeros == size:
                    done = True

            return skel

        thinned_edges = morphological_thinning(edges)  # Apply morphological thinning

        ### Create a Mask from the Thinned Edges: ###
        mask = cv2.cvtColor(thinned_edges, cv2.COLOR_GRAY2BGR)  # Convert the mask to 3-channel format

        ### Use the Mask to Update the Original Image: ###
        thinned_image = cv2.bitwise_and(image, mask)  # Apply the mask to the original image

        return thinned_image  # Return the image with thinned blob edges

    @staticmethod
    def soft_erode_blob_edges_bilateral(image, kernel_size=3, iterations=1, d=9, sigma_color=75, sigma_space=75):
        """
        Softly erode the edges of blobs in an RGB image using bilateral filtering before erosion.

        Args:
            image (np.ndarray): Input RGB image with shape (H, W, 3).
            kernel_size (int, optional): Size of the structuring element. Default is 3.
            iterations (int, optional): Number of iterations for erosion. Default is 1.
            d (int, optional): Diameter of each pixel neighborhood for bilateral filtering. Default is 9.
            sigma_color (float, optional): Filter sigma in the color space for bilateral filtering. Default is 75.
            sigma_space (float, optional): Filter sigma in the coordinate space for bilateral filtering. Default is 75.

        Returns:
            np.ndarray: Image with softly eroded blob edges.
        """
        ### Apply Bilateral Filter to Preserve Edges While Smoothing: ###
        filtered_image = cv2.bilateralFilter(image, d=d, sigmaColor=sigma_color,
                                             sigmaSpace=sigma_space)  # Apply bilateral filter

        ### Split the Filtered Image into its RGB Channels: ###
        channels = cv2.split(filtered_image)  # Split the filtered image into R, G, B channels

        ### Define Structuring Element for Erosion: ###
        kernel = np.ones((kernel_size, kernel_size), np.uint8)  # Create a square structuring element

        ### Erode Each Channel Softly: ###
        eroded_channels = [cv2.erode(channel, kernel, iterations=iterations) for channel in
                           channels]  # Erode each channel

        ### Merge the Eroded Channels Back Together: ###
        eroded_image = cv2.merge(eroded_channels)  # Merge the eroded channels back into an RGB image

        return eroded_image  # Return the image with softly eroded blob edges

    @staticmethod
    def soft_erode_blob_edges(image, kernel_size=3, iterations=1, blur_ksize=5, sigma=2):
        """
        Softly erode the edges of blobs in an RGB image using Gaussian blur before erosion.

        Args:
            image (np.ndarray): Input RGB image with shape (H, W, 3).
            kernel_size (int, optional): Size of the structuring element. Default is 3.
            iterations (int, optional): Number of iterations for erosion. Default is 1.
            blur_ksize (int, optional): Kernel size for Gaussian blur. Default is 5.
            sigma (float, optional): Standard deviation for Gaussian blur. Default is 2.

        Returns:
            np.ndarray: Image with softly eroded blob edges.
        """
        ### Apply Gaussian Blur to Soften Edges: ###
        blurred_image = cv2.GaussianBlur(image, (blur_ksize, blur_ksize), sigma)  # Apply Gaussian blur

        ### Split the Blurred Image into its RGB Channels: ###
        channels = cv2.split(blurred_image)  # Split the blurred image into R, G, B channels

        ### Define Structuring Element for Erosion: ###
        kernel = np.ones((kernel_size, kernel_size), np.uint8)  # Create a square structuring element

        ### Erode Each Channel Softly: ###
        eroded_channels = [cv2.erode(channel, kernel, iterations=iterations) for channel in
                           channels]  # Erode each channel

        ### Merge the Eroded Channels Back Together: ###
        eroded_image = cv2.merge(eroded_channels)  # Merge the eroded channels back into an RGB image

        return eroded_image  # Return the image with softly eroded blob edges


    @staticmethod
    def erode_blob_edges_rgb(image, kernel_size=3, iterations=1):
        """
        Erode the edges of blobs in an RGB image without thresholding.

        Args:
            image (np.ndarray): Input RGB image with shape (H, W, 3).
            kernel_size (int, optional): Size of the structuring element. Default is 3.
            iterations (int, optional): Number of iterations for erosion. Default is 1.

        Returns:
            np.ndarray: Image with eroded blob edges.
        """
        ### Split the Image into its RGB Channels: ###
        channels = cv2.split(image)  # Split the image into R, G, B channels

        ### Define Structuring Element for Erosion: ###
        kernel = np.ones((kernel_size, kernel_size), np.uint8)  # Create a square structuring element

        ### Erode Each Channel: ###
        eroded_channels = [cv2.erode(channel, kernel, iterations=iterations) for channel in
                           channels]  # Erode each channel

        ### Merge the Eroded Channels Back Together: ###
        eroded_image = cv2.merge(eroded_channels)  # Merge the eroded channels back into an RGB image

        return eroded_image  # Return the image with eroded blob edges

    @staticmethod
    def erode_blob_edges(image, kernel_size=3, iterations=1):
        """
        Erode the edges of blobs in an RGB image.

        Args:
            image (np.ndarray): Input RGB image with shape (H, W, 3).
            kernel_size (int, optional): Size of the structuring element. Default is 3.
            iterations (int, optional): Number of iterations for erosion. Default is 1.

        Returns:
            np.ndarray: Image with eroded blob edges.
        """
        ### Convert RGB Image to Grayscale: ###
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

        ### Threshold the Grayscale Image: ###
        _, binary_image = cv2.threshold(gray_image, 0, 255,
                                        cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Apply Otsu's thresholding

        ### Define Structuring Element for Erosion: ###
        kernel = np.ones((kernel_size, kernel_size), np.uint8)  # Create a square structuring element

        ### Erode the Binary Image: ###
        eroded_binary_image = cv2.erode(binary_image, kernel, iterations=iterations)  # Perform erosion

        ### Create a 3-Channel Mask from the Eroded Binary Image: ###
        eroded_mask = cv2.cvtColor(eroded_binary_image, cv2.COLOR_GRAY2BGR)  # Convert to 3-channel mask

        ### Apply the Eroded Mask to the Original Image: ###
        eroded_image = cv2.bitwise_and(image, eroded_mask)  # Apply the mask

        return eroded_image  # Return the image with eroded blob edges


    @staticmethod
    def stretch_histogram(frames):
        return [scale_array_stretch_hist(frames[i], min_max_values_to_scale_to=(0, 255)).astype(np.uint8) for i in np.arange(len(frames))]

    @staticmethod
    def mean_over_frames(frames):
        return [list_to_numpy(np.mean(frames, axis=0))]

    @staticmethod
    def apply_sharpening_filter_list(frames, sigma=2, strength=3):
        # frames = AlignClass.frames_to_constant_format(frames, dtype_requested='uint8', range_requested=[0, 255], channels_requested=3, threshold=5)
        return [WienerDeconvolution.apply_sharpening_filter(frame, sigma, strength) for frame in frames]

    @staticmethod
    def apply_sharpening_filter(image, sigma=2, strength=3):
        """
        Sharpen image using unsharp masking.

        Args:
            image (numpy.ndarray): Input image (H, W, C) or (H, W).
            sigma (float, optional): Standard deviation for Gaussian blur. Default is 1.0.
            strength (float, optional): Strength of the sharpening effect. Default is 1.5.

        Returns:
            numpy.ndarray: Sharpened image.
        """
        ### Convert to Grayscale If Needed: ###
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale if it is a color image
        else:
            gray_image = image  # Use the image directly if it is already grayscale

        ### Apply Gaussian Blur: ###
        blurred = cv2.GaussianBlur(gray_image, (0, 0), sigma)  # Apply Gaussian blur with specified sigma

        ### Sharpen Image: ###
        sharpened = cv2.addWeighted(gray_image, 1 + strength, blurred, -strength,
                                    0)  # Combine original and blurred images

        return sharpened  # Return the sharpened image



def wiener_filter_demo_final():
    """
        Demonstrate the Wiener filtering on an example image.
        """
    ### Load the Image: ###
    image_path = r'C:\Users\dudyk\Documents\RDND_dudy\SHABAK/default_image_1.jpg'
    original_image_torch = read_image_torch(image_path).to(DEVICE)
    # original_image = WienerDeconvolution.load_grayscale_image(image_path).astype(float)  # Load the grayscale image
    original_image = torch_to_numpy(original_image_torch)[0]

    ### Define a Blur Kernel (Example: Gaussian Blur): ###
    kernel_size = 15  # Size of the blur kernel
    # blur_kernel = cv2.getGaussianKernel(kernel_size, 10)  # Create a 1D Gaussian kernel
    # blur_kernel = np.outer(blur_kernel, blur_kernel)  # Convert 1D kernel to 2D kernel
    blur_size_in_pixels = 10
    blur_kernel = WienerDeconvolution.create_motion_blur_kernel(angle=1 * np.pi / 2 / 3,
                                                                d=blur_size_in_pixels,
                                                                sz=65) / blur_size_in_pixels
    # plt.imshow(blur_kernel); plt.show()

    ### Get Blurred Image: ###
    # blurred_image = convolve2d(original_image, blur_kernel, mode='same')  # Create a blurred version of the original image
    blurred_image = torch_to_numpy(read_image_torch(r"C:\Users\dudyk\Documents\RDND_dudy\SHABAK\blurred_image.png"))[0]
    blurred_image_torch = (numpy_to_torch(blurred_image).cuda())

    ### Get Deblurred Image Using NUBKE: ###
    blur_kernel_torch, K_kernels_basis_tensor, deblurred_image, deblurred_crop = get_blur_kernel_and_deblurred_image_using_NUBKE(
        blurred_image_torch,
        segmentation_mask_torch=None,
        NUBKE_model=None,
        device='cuda',
        n_iters=10,
        SAVE_INTERMIDIATE=True,
        saturation_threshold=0.99,
        K_number_of_base_elements=25, # TODO: probably doesn't need this because it is only necessary for model initialization and visualizations
        reg_factor=1e-3,
        optim_iters=1e-6,
        gamma_correction_factor=2.2,
        apply_dilation=False,
        apply_smoothing=True,
        apply_erosion=True,
        flag_use_avg_kernel_on_everything=False)
    # imshow_torch(blurred_image_torch/255, title_str='Blurred Image')
    # imshow_torch(deblurred_image, title_str='DeBlurred Image')

    ### Get augmented PSF's: ###
    blur_kernel = blur_kernel_torch.detach().cpu().numpy()
    augmented_psfs, augmentations = WienerDeconvolution.augment_psf(blur_kernel)

    ## Try different deconvolution methods: ###
    for current_blur_kernel in augmented_psfs:
        # current_blur_kernel = blur_kernel
        # current_blur_kernel = augmented_psfs[5]
        balance_list = WienerDeconvolution.get_default_balance_list()
        snr_list = WienerDeconvolution.get_default_SNRs()
        blurred_image_BW = RGB2BW(blurred_image)[:,:,0]
        output_dict = WienerDeconvolution.try_deconvolution_methods({},
                                                                    blurred_image_BW,
                                                                    current_blur_kernel,
                                                                    balance_list=balance_list,
                                                                    snr_list=snr_list,
                                                                    num_iter=10,
                                                                    num_psf_iter=5)
        deblurred_images = output_dict['deblurred_images']
        string_explanations = output_dict['string_explanations']

        ### Debug Results: ###
        WienerDeconvolution.plot_deconvolution_results(blurred_image, deblurred_images, string_explanations, single_figure=False)


# wiener_filter_demo_final()


def wiener_filter_demo():
    """
    Demonstrate the Wiener filtering on an example image.
    """
    ### Load the Image: ###
    image_path = r'C:\Users\dudyk\Documents\RDND_dudy\SHABAK/default_image_1.jpg'
    original_image = WienerDeconvolution.load_grayscale_image(image_path).astype(float)  # Load the grayscale image

    ### Define a Blur Kernel (Example: Gaussian Blur): ###
    kernel_size = 15  # Size of the blur kernel
    # blur_kernel = cv2.getGaussianKernel(kernel_size, 10)  # Create a 1D Gaussian kernel
    # blur_kernel = np.outer(blur_kernel, blur_kernel)  # Convert 1D kernel to 2D kernel
    blur_size_in_pixels = 10
    blur_kernel = WienerDeconvolution.create_motion_blur_kernel(angle=1*np.pi/2/3, d=blur_size_in_pixels, sz=65)/blur_size_in_pixels
    # plt.imshow(blur_kernel); plt.show()

    ### Get augmented PSF's: ###
    augmented_psfs, augmentations = WienerDeconvolution.augment_psf(blur_kernel)

    ### Get Blurred Image: ###
    blurred_image = convolve2d(original_image, blur_kernel, mode='same')  # Create a blurred version of the original image
    # imshow_np(blurred_image)

    # ### Estimate the SNR for the Entire Image: ###
    # snr_estimate_for_each_component = WienerDeconvolution.estimate_snr_fourier(original_image, blurred_image)  # Estimate the SNR for each forier component!@
    # snr_estimate = WienerDeconvolution.estimate_snr_real_space(original_image, blurred_image)  # Estimate the SNR
    # snr_estimate = WienerDeconvolution.estimate_snr_using_fourier_psd_analysis(blurred_image)  # Estimate the SNR
    # snr_estimate = WienerDeconvolution.estimate_noise_real_space_skimage(blurred_image)  # Estimate the SNR
    # snr_estimate = WienerDeconvolution.estimate_snr_using_real_space_local_variance(blurred_image, patch_size=7)  # Estimate the SNR

    # ### Estimate SNR using Wrapper Function: ###
    # snr_estimate = WienerDeconvolution.estimate_snr_wrapper(blurred_image,
    #                                                         method=0, # Options: 'real_space_local_variance', 'fourier_psd_analysis', 'noise_real_space', 'real_space'.
    #                                                         original_image=original_image,
    #                                                         patch_size=7,
    #                                                         average_sigmas=False,
    #                                                         channel_axis=None)

    # ### Estimate SNR and Deblurred Image Using Wiener Filtering: ###
    # snr_estimate, deblurred_image = WienerDeconvolution.estimate_snr_and_perform_deconvolution_EM(blurred_image,
    #                                                                              blurred_image,
    #                                                                              blur_kernel,
    #                                                                              max_iter=10,
    #                                                                              method=0, # Options: 'wiener_skimage', 'wiener_opencv', 'wiener_classic', 'unsupervised_wiener', 'richardson_lucy', 'richardson_lucy_skimage'.
    #                                                                              param=3,
    #                                                                              reg=None,
    #                                                                              num_iter=10,
    #                                                                              num_psf_iter=5)

    ## Try different deconvolution methods: ###
    balance_list = WienerDeconvolution.get_default_balance_list()
    snr_list = WienerDeconvolution.get_default_SNRs()
    output_dict = WienerDeconvolution.try_deconvolution_methods({},
                                                                blurred_image,
                                                                blur_kernel,
                                                                balance_list=balance_list,
                                                                snr_list=snr_list,
                                                                num_iter=10,
                                                                num_psf_iter=5,
                                                                deblur_strength=1)
    deblurred_images = output_dict['deblurred_images']
    string_explanations = output_dict['string_explanations']
    deblurred_images = frames_to_constant_format(deblurred_images)
    deblurred_images_stretched = WienerDeconvolution.stretch_histogram(deblurred_images)
    deblurred_images_sharpened = WienerDeconvolution.apply_sharpening_filter_list(deblurred_images)
    imshow_np(deblurred_images[-3]/255)
    # imshow_video(list_to_numpy(deblurred_images), video_title_list=string_explanations)
    # imshow_video(list_to_numpy(deblurred_images_stretched))
    # imshow_video(list_to_numpy(deblurred_images_sharpened))

    ### Debug Results: ###
    WienerDeconvolution.plot_deconvolution_results(blurred_image, deblurred_images, string_explanations, single_figure=False)

    # ### Perform Wiener Filtering Using Fourier and Real Space Methods: ###
    # deblurred_image_fourier, psf = WienerDeconvolution.blind_non_blind_deconvolution_richardson_lucy_skimage(blurred_image, blur_kernel, num_iter=10, num_psf_iter=5)  # Deblur using Fourier method
    # deblurred_image_fourier = WienerDeconvolution.non_blind_deconvolution_wiener_skimage(blurred_image, blur_kernel, balance=0.03, clip=False)  # Deblur using Fourier method
    # deblurred_image_fourier = WienerDeconvolution.non_blind_deconvolution_classic_wiener(blurred_image, blur_kernel, snr=50)  # Deblur using Fourier method
    # deblurred_image_fourier, final_blur_kernel_estimation = WienerDeconvolution.blind_deconvolution_unsupervised_wiener(blurred_image, blur_kernel, clip=False)  # Deblur using Fourier method
    # # deblurred_image_fourier, psf = WienerDeconvolution.blind_non_blind_deconvolution_richardson_lucy(blurred_image, blur_kernel, num_iter=10)  # Deblur using Fourier method
    # # deblurred_image_fourier = WienerDeconvolution.non_blind_deconvolution_classic_wiener_opencv(blurred_image.astype(np.uint8), blur_kernel, noise=3)  # Deblur using Fourier method

    ### Display the Results: ###
    # gain_factor = (deblurred_image_fourier/(blurred_image+1e-3)).mean()
    # wiener_filter_display_images_side_by_side(blurred_image, blurred_image - deblurred_image_fourier)  # Display original and deblurred images
    # plt.imshow(original_image); plt.show()
    # plt.imshow(blurred_image); plt.show()
    # plt.imshow(deblurred_image_fourier); plt.show()
    # plt.imshow(deblurred_image_fourier-blurred_image); plt.colorbar(); plt.show()
#
# wiener_filter_demo()







