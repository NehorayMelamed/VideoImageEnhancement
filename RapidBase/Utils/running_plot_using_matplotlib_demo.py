import time
from matplotlib import pyplot as plt
import numpy as np
from RapidBase.import_all import *

def live_update_demo(y_generator ,number_of_samples_to_display=128*4, title="", mini_batch_size=10, number_of_samples_to_pause=10):
    ### Default parameters: ###

    ### Initialize stuff: ###   #TODO: understand how to do it only once
    tic()
    blit = True
    x_vec = np.linspace(0, number_of_samples_to_display, num=number_of_samples_to_display)
    colors = ['b', 'g']
    X,Y = np.meshgrid(x_vec,x_vec)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    last_y = [0] * number_of_samples_to_display
    line, = ax.plot([])
    text = ax.text(0.8,0.5, title)
    ax.set_xlim(x_vec.min(), x_vec.max())
    ax.set_ylim(-500,500)
    fig.canvas.draw()   # note that the first draw comes before setting data

    ### cache the background: ###
    axbackground = fig.canvas.copy_from_bbox(ax.bbox)
    plt.show(block=False)
    toc('initialize stuff')
    print('*********************')

    ### Loop over batches in y_generator (in our case only 1 batch so for loop isn't even necessary here): ###
    t_start = time.time()
    i = 0
    for batch_index, y_batch in enumerate(y_generator):
        ### get current batch: ###
        y_batch = y_batch.squeeze()
        color = colors[batch_index % 2]

        ### Loop over current batch samples sample-by-sample: ####
        number_of_mini_batches = len(y_batch) // mini_batch_size
        for sample in range(number_of_mini_batches):
            ### Set current line data: ###
            line.set_data(x_vec, last_y)

            ### Pop most previous and append latest sample: ###
            #TODO: understand if there's a more effective way to do this
            del last_y[0:mini_batch_size]
            last_y.extend(list(y_batch[sample*mini_batch_size:(sample+1)*mini_batch_size].detach().numpy()))

            # last_y.append(y_batch[sample].detach().item())
            # last_y.pop(0)

            ### Get current display_samples last samples values+colors: ###
            tx = 'Mean Frame Rate:\n {fps:.3f}FPS'.format(fps= ((i+1) / (time.time() - t_start)) )
            text.set_text(tx)
            i += mini_batch_size
            # print(i)

            ### restore background: ###
            fig.canvas.restore_region(axbackground)

            ### redraw just the points: ###
            ax.draw_artist(line)
            ax.draw_artist(text)
            # ax.autoscale_view()

            ### fill in the axes rectangle: ###
            # fig.canvas.blit(ax1.bbox)
            fig.canvas.blit(ax.bbox)
            # in this post http://bastibe.de/2013-05-30-speeding-up-matplotlib.html
            # it is mentionned that blit causes strong memory leakage.
            # however, I did not observe that.

            if sample % number_of_samples_to_pause == 0:
                plt.pause(0.000000000001)

        # else:
        #     # redraw everything
        #     fig.canvas.draw()

        #     fig.canvas.flush_events()
        #alternatively you could use
        plt.pause(0.000000000001)
        # however plt.pause calls canvas.draw(), as can be read here:
        #http://bastibe.de/2013-05-30-speeding-up-matplotlib.html


def y_generator():
    ### Get song: ###
    song = scipy.io.loadmat("/home/simteam-j/Downloads/ofra_haza.mat")["input_vec"]
    # sr, song = scipy.io.wavfile.read("/home/simteam-j/Downloads/shibolet_basade.wav")
    # song = song[:, 20000:]
    song = song[:, 20000:] * 5
    song_shape = song.shape
    N_samples, N_channels = song_shape

    ### Get Parameters: ###
    samples_per_frame = 256
    overlap_samples_per_frame = 128
    Fs = 44100 / 2
    F_lowpass = 2000
    F_highpass = 5000
    filter_number_of_samples = overlap_samples_per_frame
    filter_parameter = 8
    # number_of_batches = int(np.floor(N_samples/overlap_samples_per_frame))
    number_of_batches = 80


    ### Take ROI from the video

    ### Run integranlon that


    ### Initialize FFT_OLA layer: ###
    FFT_OLA_layer = FFT_OLA_PerPixel_Layer_Torch(samples_per_frame=samples_per_frame, filter_name='hann',
                                                 filter_type='lowpass', N=filter_number_of_samples,
                                                 Fs=Fs, low_cutoff=F_lowpass, high_cutoff=F_highpass,
                                                 filter_parameter=filter_parameter)
    actual_filter = FFT_OLA_layer.filter

    ### Loop over audio samples and filter: ###
    output_signal = []
    output_signal_no_filter = []
    signal_buffer = torch.zeros(samples_per_frame)
    previous_buffer_samples = torch.zeros(overlap_samples_per_frame)
    song = song.squeeze()
    print(number_of_batches)
    for batch_counter in np.arange(number_of_batches):
        print(batch_counter)
        ### Get current batch: ###
        start_index = batch_counter * overlap_samples_per_frame
        stop_index = (batch_counter + 1) * overlap_samples_per_frame
        current_batch = song[start_index:stop_index].astype(float)
        current_batch = torch.tensor(current_batch)
        output_signal_no_filter.append(current_batch)

        ### Use Buffering ###
        signal_buffer = torch.cat([previous_buffer_samples, current_batch], dim=0)

        ### Get to 5D (the filtering function is general and is able to filter image pixels in time as well, later on a 1D function will be created if needed): ###
        signal_buffer = torch_get_5D(signal_buffer, 'T')

        ### Filter current batch: ###
        filtered_batch = FFT_OLA_layer.forward(signal_buffer)
        yield filtered_batch
        ### Append to output list: ###
        # output_signal.append(filtered_batch)

        ### Assign previous buffer with current output: ###
        # previous_buffer_samples = filtered_batch.squeeze()
        # previous_buffer_samples = current_batch


live_update_demo(y_generator())   # 175 fps
#live_update_demo(False) # 28 fps

