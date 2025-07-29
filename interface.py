import gradio as gr
from PIL import Image
import tempfile
import os
from final_pipeline import inference


### Create User Inferface

def predict_with_temp(image: Image.Image):
    '''
     image store as temporary file and predict using model and return final Image
    '''
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
        temp_path = temp_file.name
        image.save(temp_path)

    try:
        # Run inference with the image path
        result_img = inference(temp_path)
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return result_img 

# Gradio UI
iface = gr.Interface(
    fn=predict_with_temp,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=gr.Image(type="pil", label="Annotated Image"),
    title="Age & Gender Prediction (Clean Temp File)"
)

if __name__ == "__main__":
    iface.launch(share=True)