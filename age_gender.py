import os

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms as T
import onnxruntime as ort
from facenet_pytorch import MTCNN

from final_pipeline import inference

from get_model import model

# Check or download model
if not os.path.exists("best.onnx"):
    print(model())
else:
    print("ONNX model already exists.")


# Setup ONNX session
available_providers = ort.get_available_providers()
session = ort.InferenceSession(
    "best.onnx",
    providers=[p for p in ["CUDAExecutionProvider", "CPUExecutionProvider"] if p in available_providers]
)

# Face detector
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(keep_all=True, device=device)

# Transform for ONNX input
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

# Inference on cropped face
def run_inference(face_image):
    with torch.no_grad():
        input_tensor = transform(face_image.convert("RGB")).unsqueeze(0)
        input_array = input_tensor.cpu().numpy()
        input_name = session.get_inputs()[0].name
        gender_output, age_output = session.run(None, {input_name: input_array})
        gender = "Female" if gender_output[0][0] > 0.5 else "Male"
        age = int(age_output[0][0])
    return gender, age


