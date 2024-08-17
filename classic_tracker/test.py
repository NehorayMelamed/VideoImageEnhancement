import cv2
import imutils
import time

# Define the OpenCV object tracker dictionary with available trackers
OPENCV_OBJECT_TRACKERS = {
    "mil": cv2.TrackerMIL_create,
    "kcf": cv2.TrackerKCF_create,
    "csrt": cv2.TrackerCSRT_create
}

def initialize_tracker(tracker_type="kcf"):
    if tracker_type not in OPENCV_OBJECT_TRACKERS:
        raise ValueError(f"Unsupported tracker type: {tracker_type}")
    return OPENCV_OBJECT_TRACKERS[tracker_type]()

def main(video_path=None, tracker_type="kcf"):
    # Initialize the video stream
    print("[INFO] starting video stream...")
    vs = cv2.VideoCapture(video_path if video_path else 0)
    time.sleep(1.0)

    # Get the FPS of the video
    fps_video = vs.get(cv2.CAP_PROP_FPS)
    if fps_video == 0:
        fps_video = 30  # Fallback to 30 FPS if FPS information is not available
    frame_time = int(1000 / fps_video)

    # Grab the first frame to select the ROI
    ret, frame = vs.read()
    if not ret:
        print("[ERROR] Could not read the video")
        return

    # Resize the frame and grab the frame dimensions
    frame = imutils.resize(frame, width=600)
    (H, W) = frame.shape[:2]

    # Allow the user to select the bounding box of the object we want to track
    initBB = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
    tracker = initialize_tracker(tracker_type)
    tracker.init(frame, initBB)

    while True:
        # Grab the current frame
        ret, frame = vs.read()
        if frame is None:
            break

        # Resize the frame and grab the frame dimensions
        frame = imutils.resize(frame, width=600)
        (H, W) = frame.shape[:2]

        # Check to see if we are currently tracking an object
        if initBB is not None:
            # Grab the new bounding box coordinates of the object
            (success, box) = tracker.update(frame)

            # Check to see if the tracking was a success
            if success:
                (x, y, w, h) = [int(v) for v in box]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Initialize the set of information we'll be displaying on the frame
            info = [
                ("Tracker", tracker_type),
                ("Success", "Yes" if success else "No"),
            ]

            # Loop over the info tuples and draw them on our frame
            for (i, (k, v)) in enumerate(info):
                text = "{}: {}".format(k, v)
                cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(frame_time) & 0xFF

        # If the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # If we are using a webcam, release the pointer
    if not video_path:
        vs.release()

    # Close all windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main(video_path=r"C:\Users\orior\PycharmProjects\VideoImageEnhancement\data\images\videos\DATA\video_example_for_car_detection_1.mp4", tracker_type="mil")
