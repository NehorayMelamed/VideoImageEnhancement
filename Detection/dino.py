from Detection.GroundingDINO.groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
from typing import Union
import numpy as np
def detect_objects_dino(model, image_source: Union[str, np.ndarray], class_prompt, box_threshold=0.35, text_threshold=0.25):
    image, image_transformed = load_image(image_source)

    boxes, logits, phrases = predict(
        model=model,
        image=image_transformed,
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

if __name__ == "__main__":
    model = load_model("GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", r"C:/Users/dudyk/PycharmProjects/NehorayWorkSpace/Shaback/models/groundingdino_swint_ogc.pth")
    model = model.to('cuda:0')
    IMAGE_PATH = r"C:/Users/dudyk/PycharmProjects/NehorayWorkSpace/Shaback/models/img_1.png"
    TEXT_PROMPT = "license plate . car ."

    boxes, logits, phrases = detect_objects_dino(model, IMAGE_PATH, TEXT_PROMPT)
    display_results(IMAGE_PATH, boxes, logits, phrases)