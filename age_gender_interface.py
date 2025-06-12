<<<<<<< HEAD
import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms as T
import onnxruntime as ort
from facenet_pytorch import MTCNN

# Initialize MTCNN for face detection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)

# Load ONNX session for age/gender prediction
session = ort.InferenceSession(
    "best.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)

# Image transform for ONNX input
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

# Helper: Run ONNX inference on cropped face image
def run_inference(face_image):
    with torch.no_grad():
        input_tensor = transform(face_image.convert("RGB")).unsqueeze(0)
        input_array = input_tensor.cpu().numpy()
        input_name = session.get_inputs()[0].name
        gender_output, age_output = session.run(None, {input_name: input_array})
        gender = "Female" if gender_output[0][0] > 0.5 else "Male"
        age = int(age_output[0][0])
    return gender, age

# Main prediction function with confidence threshold only
def predict(image):
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()

    # Detect faces
    boxes, probs = mtcnn.detect(image)

    if boxes is None or len(boxes) == 0:
        return image, "No face detected."

    # Confidence threshold filter
    conf_threshold = 0.85

    # Filter boxes by confidence only
    filtered_boxes = []
    filtered_scores = []
    for box, score in zip(boxes, probs):
        if score is not None and score >= conf_threshold:
            filtered_boxes.append(box)
            filtered_scores.append(score)

    if len(filtered_boxes) == 0:
        return image, f"No faces detected above confidence threshold {conf_threshold}."

    results = []
    for box in filtered_boxes:
        box_int = [int(b) for b in box]
        face_crop = image.crop(box_int)
        gender, age = run_inference(face_crop)

        label = f"{gender}, {age}y"
        draw.rectangle(box_int, outline="red", width=10)
        draw.text((box_int[0], box_int[1] - 25), label, fill="white", font=font)
        results.append(label)

    return image

# Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=[gr.Image(label="Annotated Image")],
    title="Age & Gender Prediction with Confidence Threshold"
)

if __name__ == "__main__":
    iface.launch(share=True)
=======
import gradio as gr


def inference_(image,model_path:str):

    import onnxruntime as ort
    from torchvision import transforms as T
    from PIL import Image
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    
    # === Step 1: Load ONNX Model with GPU Support ===
    # Tries to use GPU, falls back to CPU if not available
    session = ort.InferenceSession(
        model_path,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    
    # Print provider info
    print("Available providers:", ort.get_available_providers())
    print("Session using:", session.get_providers())
    
    # === Step 2: Preprocessing Transformation ===
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    
    # === Step 3: Load and Preprocess Image ===
    image = image.convert("RGB")
    image_tensor = transform(image)  # [C, H, W]
    
    # Add batch dimension: [1, C, H, W]
    input_tensor = image_tensor.unsqueeze(0)  # shape: (1, 3, 224, 224)
    
    # Convert to NumPy
    input_array = input_tensor.cpu().numpy()
    
    # === Step 4: Run Inference ===
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_array})
    
    # === Step 5: Output ===
    gender_output, age_output = outputs
    predicted_gender = "FeMale" if gender_output[0][0] > 0.5 else "Male"
    predicted_age = age_output[0][0]

    return predicted_age,predicted_gender



def predict_age_gender(image):

    age,gender = inference_(image=image,
              model_path="best.onnx")
    return age,gender

title = "Age-Gender Prediction"

interface = gr.Interface(
    fn=predict_age_gender,
    inputs=gr.Image(type="pil"),
    outputs=[gr.Text(label="Predicted Age"), gr.Text(label="Predicted Gender")],
    title=title,
    description="Upload an image to predict age and gender"
)

interface.launch(share=True)
>>>>>>> 59fdd0aa2e3de589f3f22e213e00f97045730190
