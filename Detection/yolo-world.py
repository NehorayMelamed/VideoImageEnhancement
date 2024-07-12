from ultralytics import YOLO
import cv2


def get_bounding_boxes(image_path, model_path, classes, display_image=False):
    # Initialize the YOLO model
    model = YOLO(model_path)

    # Set custom classes
    model.set_classes(classes)

    # Predict on the image
    results = model.predict(image_path)

    # Extract bounding boxes
    bounding_boxes = []
    for result in results:
        for box in result.boxes:
            if box.cls in classes:
                bounding_boxes.append(box.xywh.tolist())  # or box.xyxy.tolist() for (x_min, y_min, x_max, y_max)

    if display_image:
        # Load image
        image = cv2.imread(image_path)

        # Draw bounding boxes on the image
        for box in bounding_boxes:
            x, y, w, h = map(int, box)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the image
        cv2.imshow("Image with Bounding Boxes", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return bounding_boxes


# Example usage:
image_path = "C:/Users/dudyk/PycharmProjects/NehorayWorkSpace/Shaback/models/img.png"
model_path = "C:/Users/dudyk/PycharmProjects/NehorayWorkSpace/Shaback/models/yolov8x-worldv2.pt"
classes = ["person", "car", "license-plate"]
display_image = True

bounding_boxes = get_bounding_boxes(image_path, model_path, classes, display_image)
print(bounding_boxes)
