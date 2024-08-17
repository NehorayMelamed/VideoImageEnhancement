from util.video_to_numpy_array import get_video_frames



#### Align_dict_example
#  input_dict (dict): Dictionary containing the following keys:
#    - 'frames' (list): List of frames (numpy arrays) to align.
#    - 'reference_frame' (np.ndarray, optional): Reference frame to align against.
#    - 'input_method' (str, optional): Method for user input.
#    - 'user_input' (dict, optional): User input data.


video_source = r"C:\Users\orior\PycharmProjects\VideoImageEnhancement\data\videos\scene_0_resized.mp4"
list_of_input_source_as_numpy_array = get_video_frames(video_source)


align_dict_example = {"frames": list_of_input_source_as_numpy_array }