def inference(image_path: str):
    """
    Final Inference Pipeline:
    - Always detects faces and predicts age/gender.
    - Uses person detection (if available) to also draw person bounding boxes.
    """
    from face import predict_face
    from age_gender import run_inference
    from PIL import Image, ImageDraw, ImageFont

    # Load image and run person+face detection
    org_img = Image.open(image_path).convert("RGB")
    outputs = predict_face(image_path)

    # Extract outputs
    person_boxes = outputs.get("person", {}).get("boxes", [])
    face_boxes = outputs["faces"]["boxes"]
    face_images = outputs["faces"]["face_images"]

    # Run age and gender prediction for all detected faces
    detection = {"gender": [], "age": []}
    for face_img in face_images:
        gender, age = run_inference(face_img)
        detection["gender"].append(gender)
        detection["age"].append(age)

    # Prepare to draw
    draw = ImageDraw.Draw(org_img)
    try:
        font = ImageFont.truetype("arial.ttf", size=50)
    except:
        font = ImageFont.load_default()

    # Draw face bounding boxes with age and gender
    for gender, age, f_box in zip(detection["gender"], detection["age"], face_boxes):
        x1, y1, x2, y2 = map(int, f_box)
        draw.rectangle([x1, y1, x2, y2], outline="red", width=5)
        draw.text((x1, y1 - 60), f"{gender} - {age}y", font=font, fill="blue")

    # Optionally draw person boxes if available
    for p_box in person_boxes:
        x1, y1, x2, y2 = map(int, p_box)
        draw.rectangle([x1, y1, x2, y2], outline="green", width=3)
        draw.text((x1, y1 - 60), "Person", font=font, fill="green")

    return org_img
