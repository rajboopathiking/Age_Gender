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
