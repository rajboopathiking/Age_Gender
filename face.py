def predict_face(image_path):

    '''
    Detect Person Using pretrained Yolo Model and Detect Face Using MTCNN
    '''
    from person import predict_person
    import torch
    from facenet_pytorch import MTCNN
    from PIL import Image

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(
    min_face_size=20,
    thresholds=[0.7, 0.8, 0.8],
    margin=30,
    keep_all=True,
    image_size=224,
    device=device
)



    crop_data, _, image = predict_person(image_path)

    person_images = crop_data["images"]
    face_boxes_all = []
    face_images_all = []

    if not person_images:
        # Fallback: face detection on whole image
        bboxes, _ = mtcnn.detect(image)
        if bboxes is not None:
            for box in bboxes:
                x1, y1, x2, y2 = map(int, box)
                face_images_all.append(image.crop((x1, y1, x2, y2)))
                face_boxes_all.append([x1, y1, x2, y2])

        return {
            "person": {},  # empty to indicate no person
            "faces": {
                "boxes": face_boxes_all,
                "face_images": face_images_all
            }
        }

    else:
        for cropped_person in person_images:
            bboxes, _ = mtcnn.detect(cropped_person)
            if bboxes is not None:
                for box in bboxes:
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
