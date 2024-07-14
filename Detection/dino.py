import os
import sys
import PARAMETER
sys.path.append(os.path.join(PARAMETER.BASE_PROJECT, "Detection"))
from GroundingDINO.groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2

def detect_objects(model, image_path, class_prompt, box_threshold=0.35, text_threshold=0.25):
    image_source, image = load_image(image_path)

    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=class_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold
    )

    return boxes, logits, phrases

def display_results(image_path, boxes, logits, phrases):
    image_source, _ = load_image(image_path)
    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    cv2.imwrite("annotated_image.jpg", annotated_frame)
    cv2.imshow("Annotated Image", annotated_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Example usage
    model = load_model("GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", r"C:/Users/dudyk/PycharmProjects/NehorayWorkSpace/Shaback/models/groundingdino_swint_ogc.pth")
    model = model.to('cuda:0')
    IMAGE_PATH = r"C:/Users/dudyk/PycharmProjects/NehorayWorkSpace/Shaback/models/img_1.png"
    TEXT_PROMPT = "license plate . car ."

    boxes, logits, phrases = detect_objects(model, IMAGE_PATH, TEXT_PROMPT)
    display_results(IMAGE_PATH, boxes, logits, phrases)
