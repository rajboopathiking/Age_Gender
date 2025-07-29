def predict_person(image_path):
    """
    Detects persons in the image and returns cropped person images and the original image with boxes drawn.
    
    Args:
        image_path (str): Path to the input image.
    
    Returns:
        Tuple[Dict[str, List], PIL.Image.Image, PIL.Image.Image]: Cropped data, image with boxes, original image
    """
    from object_detection import predict
    from PIL import Image, ImageDraw, ImageFont

    CONFIDENCE_THRESHOLD = 0.5  # only keep detections with score > 0.5

    # Load font
    try:
        font = ImageFont.truetype("arial.ttf", size=50)
    except:
        font = ImageFont.load_default()

    # Run detection
    output = predict(image_path)
    image = output["image"].copy()  # to draw boxes
    image_draw = ImageDraw.Draw(image)

    # Prepare outputs
    cropped_images = {
        "images": [],
        "boxes": []
    }

    for t in output["output"]:
        boxes = t["boxes"].detach().cpu().numpy()
        labels = t["labels"].detach().cpu().numpy()
        scores = t["scores"].detach().cpu().numpy()  # Add this line to access confidence

        for box, label, score in zip(boxes, labels, scores):
            if score < CONFIDENCE_THRESHOLD or label != 0:  # 0 assumed to be 'person'
                continue

            x1, y1, x2, y2 = map(int, box)
            image_draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            image_draw.text((x1, y1), "person", fill="blue", font=font)

            cropped = image.crop((x1, y1, x2, y2))
            cropped_images["boxes"].append([x1, y1, x2, y2])
            cropped_images["images"].append(cropped)

    return cropped_images, image, output["image"]