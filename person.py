def predict_person(image_path):

    """

     Function Returns Person Cropped Images and Original Images

     args : image_path  >> path of the image

    """
    from object_detection import predict
    from PIL import Image, ImageDraw, ImageFont

    # Try loading Arial, fall back if not available
    try:
        font = ImageFont.truetype("arial.ttf", size=50)
    except:
        font = ImageFont.load_default()

    output = predict(image_path)
    
    croped_images = {
        "images": [],
        "boxes": []
    }

    image = output["image"]
    image_draw = ImageDraw.Draw(image)
    
    
    for t in output["output"]:
        for box, label in zip(t["boxes"].detach().cpu().numpy(), t["labels"].detach().cpu().numpy()):

            x1, y1, x2, y2 = map(int, box)

            image_draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            image_draw.text((x1, y1), "person", fill="blue", font=font)

            crop_box = [x1, y1, x2, y2]
            cropped = image.crop(crop_box)
            
            croped_images["boxes"].append(crop_box)
            croped_images["images"].append(cropped)

    return croped_images,image,output["image"]
