from RapidBase.import_all import *
from scipy import signal
# def get_filter_1D(filter_name, filter_type, N, f_sampling, f_low_cutoff, f_high_cutoff, filter_parameter=True,
#                   attenuation=0, beta=14):
#     # Get window type wanted:
#     # if filter_name == 'kaiser':
#     #     window = signal.windows.kaiser(N + 1, beta, filter_parameter)
#     # elif filter_name == 'hann' or filter_name == 'hanning':
#     #     window = signal.windows.hann(N + 1, filter_parameter)
#     # elif filter_name == 'hamming' or filter_name == "hamm":
#     #     window = signal.windows.hamming(N + 1, filter_parameter)
#     # elif filter_name == 'blackmanharris':
#     #     window = signal.windows.blackmanharris(N + 1, filter_parameter)
#     # elif filter_name == 'cheb':
#     #     window = signal.windows.chebwin(N + 1, attenuation, filter_parameter)
#     if filter_name == 'kaiser':
#         filter_name = "kaiser"
#     elif filter_name == 'hann' or filter_name == 'hanning':
#         filter_name = "hann"
#     elif filter_name == 'hamming' or filter_name == "hamm":
#         filter_name = "hamming"
#     elif filter_name == 'blackmanharris':
#         filter_name = "blackmanharris"
#     elif filter_name == 'cheb':
#         filter_name = "chebwin"
#     # Calculate the coefficients using the fir1 function
#     if filter_name in ["chebwin", "kaiser"]:
#         if filter_type in ['bandpass', 'stop']:
#             coefficients = signal.firwin(N, [f_low_cutoff, f_high_cutoff], width=attenuation, window=filter_name,pass_zero="bandpass")
#         elif filter_type in ['low', 'lowpass']:
#             coefficients = signal.firwin(N, f_low_cutoff, window=filter_name, width=attenuation, pass_zero="lowpass")
#         elif filter_type in ['high', 'highpass']:
#             coefficients = signal.firwin(N, f_high_cutoff, window=filter_name, width=attenuation, pass_zero="highpass")
#     else:
#         if filter_type in ['bandpass', 'stop']:
#             coefficients = signal.firwin(N, [f_low_cutoff, f_high_cutoff], fs=f_sampling, window=filter_name, pass_zero="bandpass")
#         elif filter_type in ['low', 'lowpass']:
#             coefficients = signal.firwin(N, f_low_cutoff, fs=f_sampling, window=filter_name, pass_zero="lowpass")
#         elif filter_type in ['high', 'highpass']:
#             coefficients = signal.firwin(N, f_high_cutoff, fs=f_sampling, window=filter_name, pass_zero="highpass")
#     actual_filter = signal.dlti(coefficients, [1], dt=1)
#
#     return actual_filter, coefficients
#
#
# # from RapidBase.Utils.Classical_DSP.FFT_utils import torch_fftshift
# # import torch
# def get_filter_2D(filter_coefficients_1D):
#     ### Get the 1D filter Coefficients: ###
#     n = int((length(filter_coefficients_1D)-1)/2)  #filter_coefficients_1D must be of odd length!!!!
#     filter_coefficients_1D_shifted = torch_fftshift(filter_coefficients_1D.flip(-1)).flip(-1)
#     a = torch.cat([filter_coefficients_1D_shifted[0:1], 2*filter_coefficients_1D_shifted[1:n]], -1)
#
#     ### Use Chebyshev polynomials to compute h: ###
#     t = torch.ones(3,3).to(filter_coefficients_1D.device)
#     t[0,0] = 1
#     t[0,1] = 2
#     t[0,2] = 1
#     t[1,0] = 2
#     t[1,1] = -4
#     t[1,2] = 2
#     t[2,0] = 1
#     t[2,1] = 2
#     t[2,2] = 1
#     t = t/8
#     P0 = 1
#     P1 = t
#     h = a[1]*P1
#     inset0 = int(np.floor((t.shape[-1] - 1)/2))
#     inset1 = int(np.floor((t.shape[-2] - 1)/2))
#     inset = [inset0, inset1]
#     rows = inset[0] + 1
#     cols = inset[1] + 1
#     h[rows-1,cols-1] = h[rows-1,cols-1] + a[0]*P0
#     for i in np.arange(3,n+1):
#         P2 = 2 * filter2D_torch(t, P1)
#         rows = rows + inset(1)
#         cols = cols + inset(2)
#         P2(rows, cols) = P2(rows, cols) - P0
#         rows = inset(1) + (1:size(P1, 1))
#         cols = inset(2) + (1:size(P1, 2))
#         hh = h
#         h = a(i) * P2
#         h(rows, cols) = h(rows, cols) + hh
#         P0 = P1
#         P1 = P2
#
    
# b = rot90(fftshift(rot90(b,2)),2); % Inverse fftshift
# a = [b(1) 2*b(2:n+1)];
#
# inset = floor((size(t)-1)/2);
#
# % Use Chebyshev polynomials to compute h
# P0 = 1; P1 = t;
# h = a(2)*P1;
# rows = inset(1)+1; cols = inset(2)+1;
# h(rows,cols) = h(rows,cols)+a(1)*P0;
# for i=3:n+1,
#   P2 = 2*conv2(t,P1);
#   rows = rows + inset(1); cols = cols + inset(2);
#   P2(rows,cols) = P2(rows,cols) - P0;
#   rows = inset(1) + (1:size(P1,1));
#   cols = inset(2) + (1:size(P1,2));
#   hh = h;
#   h = a(i)*P2; h(rows,cols) = h(rows,cols) + hh;
#   P0 = P1;
#   P1 = P2;
# end
# h = rot90(h,2); % Rotate for use with filter2







