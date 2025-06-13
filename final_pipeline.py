

def inference(image_path:str):
    from face import predict_face
    from age_gender import run_inference
    from PIL import Image,ImageDraw,ImageFont
    org_img = Image.open(image_path)
    outputs = predict_face(
        image_path
    )

    if len(outputs["person"]) < 1:
        detection = {
        "gender":[],
        "age":[]
    }
        faces = outputs["faces"]
        for i in faces["face_images"]:
            gender,age = run_inference(i)
            detection["gender"].append(gender)
            detection["age"].append(age)
        outputs["gender"] = detection["gender"]
        outputs["age"] = detection["age"]

        draw = ImageDraw.Draw(org_img)
        for gender,age,f_box in zip(outputs["gender"],outputs["age"],outputs["faces"]["boxes"]):
            text = f"{gender} - {age}y"
            f_x1,f_y1,f_x2,f_y2 = f_box
            font = ImageFont.truetype("arial.ttf",size=50)
            draw.rectangle([int(f_x1),int(f_y1),int(f_x2),int(f_y2)],width=5,outline="red")
            draw.text((int(f_x1),int(f_y1)-60),text,font=font,fill="blue")
        return org_img
    else:
        detection = {
            "gender":[],
            "age":[]
        }
        faces = outputs["faces"]
        for i in faces["face_images"]:
            gender,age = run_inference(i)
            detection["gender"].append(gender)
            detection["age"].append(age)
        outputs["gender"] = detection["gender"]
        outputs["age"] = detection["age"]

        draw = ImageDraw.Draw(org_img)
        for gender,age,p_box,f_box in zip(outputs["gender"],outputs["age"],outputs["person"]["boxes"],outputs["faces"]["boxes"]):
            text = f"{gender} - {age}y"
            p_x1,p_y1,p_x2,p_y2 = p_box
            f_x1,f_y1,f_x2,f_y2 = f_box
            font = ImageFont.truetype("arial.ttf",size=50)
            draw.rectangle([int(p_x1),int(p_y1),int(p_x2),int(p_y2)],width=5,outline="red")
            draw.text((int(p_x1),int(p_y1)-60),text,font=font,fill="blue")

        return org_img