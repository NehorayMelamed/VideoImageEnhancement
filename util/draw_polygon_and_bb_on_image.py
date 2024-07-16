import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle

class InteractiveDrawing:
    def __init__(self, ax):
        self.ax = ax
        self.canvas = ax.figure.canvas
        self.polygon_points = []
        self.bbox_points = []
        self.polygon = None
        self.bbox = None
        self.drawing_polygon = False
        self.drawing_bbox = False
        self.cid = self.canvas.mpl_connect('button_press_event', self.on_click)

    def on_click(self, event):
        if event.inaxes != self.ax:
            return

        if self.drawing_polygon:
            self.polygon_points.append((event.xdata, event.ydata))
            if len(self.polygon_points) > 1:
                self.update_polygon()
        elif self.drawing_bbox:
            if len(self.bbox_points) == 0:
                self.bbox_points = [(event.xdata, event.ydata)]
            elif len(self.bbox_points) == 1:
                self.bbox_points.append((event.xdata, event.ydata))
                self.update_bbox()

    def update_polygon(self):
        if self.polygon is not None:
            self.polygon.remove()
        self.polygon = Polygon(self.polygon_points, closed=True, edgecolor='g', fill=False)
        self.ax.add_patch(self.polygon)
        self.canvas.draw()

    def update_bbox(self):
        if self.bbox is not None:
            self.bbox.remove()
        x0, y0 = self.bbox_points[0]
        x1, y1 = self.bbox_points[1]
        self.bbox = Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor='r', fill=False)
        self.ax.add_patch(self.bbox)
        self.canvas.draw()

    def reset(self):
        self.polygon_points = []
        self.bbox_points = []
        if self.polygon is not None:
            self.polygon.remove()
            self.polygon = None
        if self.bbox is not None:
            self.bbox.remove()
            self.bbox = None
        self.canvas.draw()

def main(video_path):
    # Load the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Unable to open video.")
        return

    # Read the first frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read the first frame.")
        return
    cap.release()

    # Convert the frame from BGR to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots()
    ax.imshow(frame)
    interactive_drawing = InteractiveDrawing(ax)

    def on_key(event):
        if event.key == 'p':
            interactive_drawing.drawing_polygon = not interactive_drawing.drawing_polygon
            interactive_drawing.drawing_bbox = False
            print(f"Polygon drawing mode: {interactive_drawing.drawing_polygon}")
        elif event.key == 'b':
            interactive_drawing.drawing_bbox = not interactive_drawing.drawing_bbox
            interactive_drawing.drawing_polygon = False
            print(f"BBox drawing mode: {interactive_drawing.drawing_bbox}")
        elif event.key == 'r':
            interactive_drawing.reset()
            print("Reset drawing")
        elif event.key == 'q':
            print("Polygon points:", interactive_drawing.polygon_points)
            print("Bounding box points:", interactive_drawing.bbox_points)
            plt.close()

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()

if __name__ == "__main__":
    video_path = input("Enter the path to the video: ")
    main(video_path)
