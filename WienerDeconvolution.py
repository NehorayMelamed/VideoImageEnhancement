


# from Rapid_Base.import_all import *

# ### Wiener Filtering: ###
# import numpy as np
# import cv2
# from scipy.signal import convolve2d
# from scipy.fft import fft2, ifft2, fftshift
# import matplotlib.pyplot as plt
# import numpy as np
# import cv2
# from skimage import data, restoration
# from skimage.restoration import unsupervised_wiener
# import matplotlib.pyplot as plt
# from scipy.signal import fftconvolve
# from skimage.restoration import estimate_sigma  # For estimating noise standard deviation
# import torch
# from typing import Tuple
# from torch import Tensor
# from skimage.util import img_as_float


### Import NUBKE: ###
from nubke_utils import *
from Utils import *


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
    def blind_deconvolution_unsupervised_wiener(image, psf, reg=None, user_params=None, is_real=True, clip=False, rng=None):
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

        Returns:
        tuple: Tuple containing:
            - deblurred_image (np.ndarray): The deblurred image with shape (H, W) or (H, W, C).
            - psf (np.ndarray): The estimated point spread function (PSF) with shape (H, W).
        """
        ### This Is The Code Block: ###
        if image.ndim == 3 and image.shape[2] == 3:  # If image is RGB
            deblurred_image = np.zeros_like(image)  # Initialize output image
            psf = np.ones((image.shape[0], image.shape[1], 3))  # Initialize PSF
            ### Looping Over Indices: ###
            for c in range(3):  # Process each channel separately
                deblurred_image[:, :, c], psf[:, :, c] = unsupervised_wiener(
                    image[:, :, c], psf[:, :, c], reg=reg, user_params=user_params, is_real=is_real, clip=clip, rng=rng
                )  # Unsupervised Wiener
            return deblurred_image, psf
        elif image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):  # If image is grayscale
            deblurred_image, psf = unsupervised_wiener(
                image, psf, reg=reg, user_params=user_params, is_real=is_real, clip=clip, rng=rng
            )  # Unsupervised Wiener
            return deblurred_image, psf
        else:
            raise ValueError("Input image must be of shape (H, W) or (H, W, C) where C=1 or 3.")

    @staticmethod
    def non_blind_deconvolution_classic_wiener(image, psf, snr):
        """
        Perform Wiener filtering in the Fourier domain.

        Parameters:
        image (np.ndarray): The input image with shape (H, W) or (H, W, C).
        psf (np.ndarray): The blur kernel with shape (kernel_H, kernel_W).
        snr (float): The estimated Signal-to-Noise Ratio (SNR).

        Returns:
        np.ndarray: The deblurred image with shape (H, W) or (H, W, C).
        """

        ### Normalize the PSF: ###
        psf /= psf.sum()  # Normalize the PSF so that its sum is 1

        ### Check Image Dimensionality: ###
        if image.ndim == 3 and image.shape[2] == 3:  # If image is RGB
            deblurred_image = np.zeros_like(image)  # Initialize output image
            ### Looping Over Indices: ###
            for c in range(3):  # Process each channel separately
                deblurred_image[:, :, c] = WienerDeconvolution.wiener_deconvolution_fourier(image[:, :, c], psf, snr)  # Recursive call
            return deblurred_image
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
    def non_blind_deconvolution_classic_wiener_opencv(image, psf, noise=1e-6):
        """
        Perform non-blind Wiener deconvolution using OpenCV for both RGB and grayscale images.

        Parameters:
        image (np.ndarray): Input image with shape (H, W) or (H, W, C).
        psf (np.ndarray): Point spread function (PSF) with shape (kernel_H, kernel_W).
        noise (float): Noise power for Wiener filter. Default is 1e-6.

        Returns:
        np.ndarray: Deblurred image with shape (H, W) or (H, W, C).
        """
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

            ### Looping Over Indices: ###
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

    @staticmethod
    def blind_non_blind_deconvolution_richardson_lucy(image, psf, num_iter=50):
        """
        Perform blind deconvolution using the Richardson-Lucy algorithm.

        Parameters:
        image (np.ndarray): The observed (blurred and noisy) image with shape (H, W) or (H, W, C).
        psf (np.ndarray): Initial estimate of the point spread function (PSF).
        num_iter (int): Number of iterations. Default is 50.

        Returns:
        np.ndarray: Deblurred image.
        np.ndarray: Estimated PSF.
        """
        ### This Is The Code Block: ###
        im_deconv = np.full(image.shape, 0.1, dtype='float')  # Initialize deconvolved image
        ### Looping Over Indices: ###
        for i in range(num_iter):  # Perform iterations
            psf_mirror = np.flip(psf)  # Flip the PSF
            conv = fftconvolve(im_deconv, psf, mode='same')  # Convolve the deconvolved image with the PSF
            relative_blur = image / conv  # Compute the relative blur
            im_deconv *= fftconvolve(relative_blur, psf_mirror, mode='same')  # Update the deconvolved image
            im_deconv_mirror = np.flip(im_deconv)  # Flip the deconvolved image
            psf *= fftconvolve(relative_blur, im_deconv_mirror, mode='same')  # Update the PSF
            psf /= psf.sum()  # Normalize the PSF
        return im_deconv, psf

    @staticmethod
    def non_blind_deconvolution_wiener_skimage(image, psf, balance, reg=None, is_real=True, clip=True):
        """
        Perform Wiener deconvolution using the Wiener filter provided by skimage.

        Parameters:
        image (np.ndarray): The observed (blurred and noisy) image with shape (H, W) or (H, W, C).
        psf (np.ndarray): Point spread function (PSF) with shape (kernel_H, kernel_W).
        balance (float): The regularization parameter for Wiener filtering.
        reg (np.ndarray or float, optional): The regularization parameter. Default is None.
        is_real (bool): If True, assumes the output is a real image. Default is True.
        clip (bool): If True, clips the output image to the range [0, 1]. Default is True.

        Returns:
        np.ndarray: Deblurred image.
        """
        ### This Is The Code Block: ###
        deblurred_image = restoration.wiener(image, psf, balance, reg=reg, is_real=is_real,
                                             clip=clip)  # Perform Wiener deconvolution
        return deblurred_image

    @staticmethod
    def blind_non_blind_deconvolution_richardson_lucy_skimage(image, psf, num_iter=50, num_psf_iter=5):
        """
        Perform blind deconvolution using the Richardson-Lucy algorithm with Wiener filtering.

        Parameters:
        image (np.ndarray): The observed (blurred and noisy) image with shape (H, W) or (H, W, C).
        psf (np.ndarray): Initial estimate of the point spread function (PSF).
        num_iter (int): Number of iterations for image and PSF deconvolution. Default is 50.
        num_psf_iter (int): Number of iterations for PSF estimation in each outer loop. Default is 5.

        Returns:
        np.ndarray: Deblurred image.
        np.ndarray: Estimated PSF.
        """
        image = img_as_float(image)  # Ensure the image is in float format

        for i in range(num_iter):
            # Update the estimated image using Richardson-Lucy algorithm
            image_deconv = restoration.richardson_lucy(image, psf, num_iter=num_psf_iter, clip=False)

            # Update the PSF
            psf_mirror = np.flip(psf)
            relative_blur = image / (fftconvolve(image_deconv, psf, mode='same') + 1e-10)
            psf_update = fftconvolve(relative_blur, np.flip(image_deconv), mode='same')

            # Pad psf to match the dimensions of psf_update
            psf_padded = np.zeros_like(psf_update)
            kh, kw = psf.shape
            psf_padded[:kh, :kw] = psf

            psf_padded *= psf_update
            psf_padded /= psf_padded.sum()  # Normalize the PSF to ensure its sum is 1

            # Extract the updated PSF from the padded PSF
            psf = psf_padded[:kh, :kw]

        return image_deconv, psf

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
    def deconvolution_wrapper(image, psf, method, param=3, reg=None, num_iter=10, num_psf_iter=5):
        """
        Wrapper function to perform various deconvolution methods based on the input string or index.

        Parameters:
        image (np.ndarray): The observed (blurred and noisy) image with shape (H, W) or (H, W, C).
        psf (np.ndarray): Point spread function (PSF) with shape (kernel_H, kernel_W).
        method (str or int): Method to use for deconvolution. Options: 'wiener_skimage', 'wiener_opencv',
                             'wiener_classic', 'unsupervised_wiener', 'richardson_lucy', 'richardson_lucy_skimage'.
        param (float or int): Regularization parameter for Wiener filtering or number of iterations for Richardson-Lucy.
                              Default is 3.
        reg (np.ndarray or float, optional): Regularization parameter for Wiener filtering. Default is None.
        num_iter (int): Number of iterations for Richardson-Lucy deconvolution. Default is 10.
        num_psf_iter (int): Number of iterations for PSF estimation in Richardson-Lucy deconvolution. Default is 5.

        Returns:
        np.ndarray: Deblurred image.
        """
        if method == 'wiener_skimage' or method == 0:
            return WienerDeconvolution.non_blind_deconvolution_wiener_skimage(image, psf, param, reg)
        elif method == 'wiener_opencv' or method == 1:
            return WienerDeconvolution.non_blind_deconvolution_classic_wiener_opencv(image, psf, noise=param)
        elif method == 'wiener_classic' or method == 2:
            return WienerDeconvolution.non_blind_deconvolution_classic_wiener(image, psf, snr=param)
        elif method == 'unsupervised_wiener' or method == 3:
            return WienerDeconvolution.blind_deconvolution_unsupervised_wiener(image, psf)
        elif method == 'richardson_lucy' or method == 4:
            return WienerDeconvolution.blind_non_blind_deconvolution_richardson_lucy(image, psf, num_iter=param)
        elif method == 'richardson_lucy_skimage' or method == 5:
            return WienerDeconvolution.richardson_lucy_blind(image, psf, num_iter=num_iter, num_psf_iter=num_psf_iter)
        else:
            raise ValueError("Invalid method. Choose from 'wiener_skimage', 'wiener_opencv', 'wiener_classic', "
                             "'unsupervised_wiener', 'richardson_lucy', 'richardson_lucy_skimage'.")

    @staticmethod
    def try_deconvolution_methods(blurred_image, blur_kernel, balance_list=None, snr_list=None, num_iter=10, num_psf_iter=5):
        """
        Try different balance values for non_blind_deconvolution_wiener_skimage and different SNR values for
        non_blind_deconvolution_classic_wiener, and return all results along with explanations.
        Also, include the outputs of blind_non_blind_deconvolution_richardson_lucy_skimage and blind_deconvolution_unsupervised_wiener.

        Parameters:
        blurred_image (np.ndarray): The blurred input image with shape (H, W) or (H, W, C).
        blur_kernel (np.ndarray): The blur kernel (PSF) with shape (kernel_H, kernel_W).
        balance_list (list of float): List of balance values to try for non_blind_deconvolution_wiener_skimage.
        snr_list (list of float): List of SNR values to try for non_blind_deconvolution_classic_wiener.
        num_iter (int): Number of iterations for Richardson-Lucy deconvolution. Default is 10.
        num_psf_iter (int): Number of iterations for PSF estimation in Richardson-Lucy deconvolution. Default is 5.

        Returns:
        tuple:
            - list of np.ndarray: List of deblurred images.
            - list of str: List of strings explaining the corresponding deblurred images.
        """
        results = []  # List to store deblurred images
        explanations = []  # List to store explanations

        if balance_list is None:  # If balance_list is None
            balance_list = [0.1, 0.5, 1, 2, 5, 10, 20, 50]
        if snr_list is None:  # If snr_list is None
            snr_list = [0.1, 0.5, 1, 2, 5, 10, 20, 50]

        ### Looping Over Balance Values: ###
        for balance in balance_list:  # Iterate over each balance value
            ### Perform Wiener Deconvolution Using Skimage: ###
            deblurred_image = WienerDeconvolution.non_blind_deconvolution_wiener_skimage(blurred_image, blur_kernel, balance, clip=False)  # Deblur using specified balance
            results.append(deblurred_image)  # Append the deblurred image to results
            explanations.append(f"Deblurred using Wiener Skimage with balance={balance}")  # Append explanation

        ### Looping Over SNR Values: ###
        for snr in snr_list:  # Iterate over each SNR value
            ### Perform Classic Wiener Deconvolution: ###
            deblurred_image = WienerDeconvolution.non_blind_deconvolution_classic_wiener(
                blurred_image, blur_kernel, snr
            )  # Deblur using specified SNR
            results.append(deblurred_image)  # Append the deblurred image to results
            explanations.append(f"Deblurred using Classic Wiener with SNR={snr}")  # Append explanation

        ### Perform Richardson-Lucy Blind Deconvolution: ###
        deblurred_image_rl, psf_rl = WienerDeconvolution.blind_non_blind_deconvolution_richardson_lucy_skimage(
            blurred_image, blur_kernel, num_iter=num_iter, num_psf_iter=num_psf_iter
        )  # Deblur using Richardson-Lucy method
        results.append(deblurred_image_rl)  # Append the deblurred image to results
        explanations.append("Deblurred using Richardson-Lucy Blind Deconvolution")  # Append explanation

        ### Perform Unsupervised Wiener Blind Deconvolution: ###
        deblurred_image_uw, final_blur_kernel_estimation = WienerDeconvolution.blind_deconvolution_unsupervised_wiener(
            blurred_image, blur_kernel, clip=False
        )  # Deblur using Unsupervised Wiener method
        results.append(deblurred_image_uw)  # Append the deblurred image to results
        explanations.append("Deblurred using Unsupervised Wiener Blind Deconvolution")  # Append explanation

        return results, explanations  # Return the deblurred images and their explanations

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
    def get_blur_kernels_and_deblurred_images_using_NUBKE(blurred_images, input_method='BB', user_input=None):
        ### Get Region From User: ###
        #TODO: support input a list BBs, right now assuming user_input
        H, W = blurred_images[0].shape[0:2]
        initial_BB_XYXY, initial_polygon_points, initial_segmentation_mask, grid_points, flag_no_input = user_input_to_all_input_types(
            user_input,
            input_method=input_method,
            input_shape=(H, W))

        ### Loop Over Images And Get Deblurred Images And Blur Kernel Using NUBKE: ###
        average_blur_kernels_list = []
        deblurred_crops_list = []
        for i in np.arange(len(blurred_images)):
            ### Get current blurred image: ###
            current_blurred_image = blurred_images[i]

            ### Get Deblurred Image Using NUBKE: ###
            blurred_image_torch = numpy_to_torch(current_blurred_image)
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

            ### Append to Lists: ###
            average_blur_kernels_list.append(average_blur_kernel)
            deblurred_crops_list.append(torch_to_numpy(deblurred_crop))

        return average_blur_kernels_list, deblurred_crops_list

    @staticmethod
    def get_blur_kernel_on_image_using_NUBKE(blurred_image, input_method='BB', user_input=None):
        ### Get Region From User: ###
        H, W = blurred_image.shape[0:2]
        initial_BB_XYXY, initial_polygon_points, initial_segmentation_mask, grid_points, flag_no_input = user_input_to_all_input_types(
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
    def get_blur_kernel_and_deblurred_image_using_NUBKE(blurred_image, input_method='BB', user_input=None):
        ### Get Region From User: ###
        H,W = blurred_image.shape[0:2]
        initial_BB_XYXY, initial_polygon_points, initial_segmentation_mask, grid_points, flag_no_input = user_input_to_all_input_types(
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
            K_number_of_base_elements=25, # TODO: probably doesn't need this because it is only necessary for model initialization and visualizations
            reg_factor=1e-3,
            optim_iters=1e-6,
            gamma_correction_factor=2.2,
            apply_dilation=False,
            apply_smoothing=True,
            apply_erosion=True,
            flag_use_avg_kernel_on_everything=False)
        average_blur_kernel = torch_to_numpy(blur_kernel_torch)

        return average_blur_kernel, deblurred_crop

    @staticmethod
    def get_blur_kernel_using_NUBKE_and_deblur_using_Wiener_all_options(blurred_image, input_method='BB', user_input=None, flag_plot=False):
        ## Get Region From User: ###
        H, W = blurred_image.shape[0:2]
        initial_BB_XYXY, initial_polygon_points, initial_segmentation_mask, grid_points, flag_no_input = user_input_to_all_input_types(
            user_input,
            input_method=input_method,
            input_shape=(H, W))

        ### Get Average Blur Kernel Using NUBKE: ###
        average_blur_kernel, deblurred_crop = WienerDeconvolution.get_blur_kernel_and_deblurred_image_using_NUBKE(blurred_image, input_method='BB', user_input=None)

        ### Insert Blur Kernel into Wiener Deconvolution: ###
        balance_list = WienerDeconvolution.get_default_balance_list()
        snr_list = WienerDeconvolution.get_default_SNRs()
        blurred_image_BW = RGB2BW(blurred_image)[:, :, 0]
        deblurred_images, string_explanations = WienerDeconvolution.try_deconvolution_methods(blurred_image_BW,  #TODO: enable deblurring RGB frames by deblurring each channel!!!
                                                                                              average_blur_kernel,
                                                                                              balance_list=balance_list,
                                                                                              snr_list=snr_list,
                                                                                              num_iter=10,
                                                                                              num_psf_iter=5)

        ### Get Deblurred Crop: ###
        X0, Y0, X1, Y1 = initial_BB_XYXY
        deblurred_crops_list = []
        blurred_crops_list = []
        for i in np.arange(len(deblurred_images)):
            blurred_crop = blurred_image[Y0:Y1, X0:X1]
            deblurred_crop = deblurred_images[i][Y0:Y1, X0:X1]
            deblurred_crops_list.append(deblurred_crop)
            blurred_crops_list.append(blurred_crop)

        ### Debug Results: ###
        if flag_plot:
            WienerDeconvolution.plot_deconvolution_results(blurred_crops_list, deblurred_crops_list, string_explanations, single_figure=False)

        return deblurred_images, string_explanations

    @staticmethod
    def get_deblurred_image_from_blur_kernel_using_Wiener_all_options(blurred_image, blur_kernel, input_method='BB', user_input=None, flag_plot=False):
        ## Get Region From User: ###
        H, W = blurred_image.shape[0:2]
        initial_BB_XYXY, initial_polygon_points, initial_segmentation_mask, grid_points, flag_no_input = user_input_to_all_input_types(
            user_input,
            input_method=input_method,
            input_shape=(H, W))

        ### Insert Blur Kernel into Wiener Deconvolution: ###
        balance_list = WienerDeconvolution.get_default_balance_list()
        snr_list = WienerDeconvolution.get_default_SNRs()
        blurred_image_BW = RGB2BW(blurred_image)[:, :, 0]
        deblurred_images, string_explanations = WienerDeconvolution.try_deconvolution_methods(blurred_image_BW, # TODO: enable deblurring RGB frames by deblurring each channel!!!
                                                                                              blur_kernel,
                                                                                              balance_list=balance_list,
                                                                                              snr_list=snr_list,
                                                                                              num_iter=10,
                                                                                              num_psf_iter=5)

        ### Get Deblurred Crop: ###
        X0, Y0, X1, Y1 = initial_BB_XYXY.shape
        deblurred_crops_list = []
        blurred_crops_list = []
        for i in np.arange(len(deblurred_images)):
            blurred_crop = blurred_image[Y0:Y1, X0:X1]
            deblurred_crop = deblurred_images[i][Y0:Y1, X0:X1]
            deblurred_crops_list.append(deblurred_crop)
            blurred_crops_list.append(blurred_crop)

        ### Debug Results: ###
        if flag_plot:
            WienerDeconvolution.plot_deconvolution_results(blurred_crops_list, deblurred_crops_list, string_explanations, single_figure=False)

        return deblurred_images, string_explanations

    @staticmethod
    def get_deblurred_image_from_blur_kernel_using_Wiener_specific_option(blurred_image,
                                                                          blur_kernel,
                                                                          wiener_method,
                                                                          wiener_parameter=5, #SNR or balance parameter
                                                                          input_method='BB',
                                                                          user_input=None,
                                                                          flag_plot=False):
        # [wiener_method] = 'wiener_skimage, 'wiener_opencv', 'wiener_classic', 'unsupervised_wiener', 'richardson_lucy', 'richardson_lucy_skimage'
        ## Get Region From User: ###
        H, W = blurred_image.shape[0:2]
        initial_BB_XYXY, initial_polygon_points, initial_segmentation_mask, grid_points, flag_no_input = user_input_to_all_input_types(
            user_input,
            input_method=input_method,
            input_shape=(H, W))

        ### Insert Blur Kernel into Wiener Deconvolution: ###
        blurred_image_BW = RGB2BW(blurred_image)[:, :, 0]
        deblurred_image = WienerDeconvolution.deconvolution_wrapper(blurred_image_BW,
                                                                    blur_kernel,
                                                                    method=wiener_method,
                                                                    param=wiener_parameter,
                                                                    reg=None,
                                                                    num_iter=10,
                                                                    num_psf_iter=5)

        ### Get Deblurred Crop: ###
        X0, Y0, X1, Y1 = initial_BB_XYXY
        deblurred_crop = deblurred_image[Y0:Y1, X0:X1]

        return deblurred_crop

    @staticmethod
    def get_blur_kernel_and_deblurred_image_unsupervised_wiener(blurred_image, blur_kernel=None, input_method='BB', user_input=None):
        ### Get Initial Blur Kernel: ###
        if blur_kernel is None:
            blur_kernel = np.zeros((33,33))

        ### Perform Unsupervised Wiener Deconvolution: ###
        deblurred_image, blur_kernel = WienerDeconvolution.blind_deconvolution_unsupervised_wiener(blurred_image, psf=blur_kernel)

        return deblurred_image, blur_kernel





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
    blur_kernel = WienerDeconvolution.create_motion_blur_kernel(angle=1 * np.pi / 2 / 3, d=blur_size_in_pixels,
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
        deblurred_images, string_explanations = WienerDeconvolution.try_deconvolution_methods(blurred_image_BW,
                                                                                              current_blur_kernel,
                                                                                              balance_list=balance_list,
                                                                                              snr_list=snr_list,
                                                                                              num_iter=10,
                                                                                              num_psf_iter=5)

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
    deblurred_images, string_explanations = WienerDeconvolution.try_deconvolution_methods(blurred_image, blur_kernel, balance_list=balance_list, snr_list=snr_list, num_iter=10, num_psf_iter=5)

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

# wiener_filter_demo()




