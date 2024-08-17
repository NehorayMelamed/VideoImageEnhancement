from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

file_path = r"C:\Users\orior\PycharmProjects\VideoImageEnhancement\data\videos\scene_0_resized.mp4"
start_time = 0
end_time   = 2


ffmpeg_extract_subclip(file_path, start_time, end_time, targetname=r"C:\Users\orior\PycharmProjects\VideoImageEnhancement\data\videos\scene_0_resized_short.mp4")


