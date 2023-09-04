from GroundingDINO.util.inference import load_model, load_image, predict, annotate
import cv2

CONFIG_PATH = "./GroundingDINO/config/GroundingDINO_SwinT_OGC.py"
CHECKPOINT_PATH = "./GroundingDINO/groundingdino_swint_ogc.pth"
DEVICE = "cuda"
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25

dino_model = load_model(CONFIG_PATH, CHECKPOINT_PATH)


def dino_predict(image_path, text_prompt):
    image_source, image = load_image(image_path)

    # Perform prediction
    boxes, logits, phrases = predict(
        model=dino_model,
        image=image,
        caption=text_prompt,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        device=DEVICE,
    )

    # Annotate the image
    annotated_frame = annotate(image_source=image_source, boxes=boxes)

    # Save the annotated image
    output_image_path = "annotated_image.jpg"
    cv2.imwrite(output_image_path, annotated_frame)

    return output_image_path
