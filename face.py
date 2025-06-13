from person import predict_person
def predict_face(image_path):
    """
    function Extract Person in image Then Extract Face Of Person then Detect age and gender

    image_path : path of the images
    
      
    """
    import torch
    from facenet_pytorch import MTCNN
    from PIL import Image

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(
        keep_all=True,
        device=device,
        image_size=224,
        thresholds=[0.85, 0.90, 0.95]
    )

    # Get person crops and boxes
    crop_data,_,image = predict_person(image_path)
    person_images = crop_data["images"]
    face_boxes_all = []
    face_images_all = []


    if len(crop_data["images"])<1:
        bboxes, probs = mtcnn.detect(image)
        for box in bboxes:
            x1, y1, x2, y2 = map(int, box)
            face_images_all.append(image.crop((x1, y1, x2, y2)))
            face_boxes_all.append([x1, y1, x2, y2])
        return {
            "person": [],
            "faces": {
                "boxes": face_boxes_all,
                "face_images": face_images_all
            }
        }
    else:


        for cropped_person in person_images:
            bboxes, probs = mtcnn.detect(cropped_person)
            
            if bboxes is None:
                continue  # No face detected in this person crop

            for box in bboxes:
                # Convert float to int for cropping
                x1, y1, x2, y2 = map(int, box)
                face_images_all.append(cropped_person.crop((x1, y1, x2, y2)))
                face_boxes_all.append([x1, y1, x2, y2])

        return {
            "person": crop_data,
            "faces": {
                "boxes": face_boxes_all,
                "face_images": face_images_all
            }
        }