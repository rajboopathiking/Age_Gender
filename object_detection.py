from model import get_model
import torch
from torchvision.transforms import v2 as T
from PIL import Image
import os
from get_ob_model import model


### Preprocessing 

if not os.path.exists("./best_weights.pth"):
    model()
else:
    print("Object Detection Model already !")
model = get_model(
    weight_path="./best_weights.pth"
)

def predict(Image_path):
    transform = T.Compose(
        [
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True)
        ]
    )

    image = Image.open(Image_path)
    tensor_image = transform(image)

    model.eval()
    with torch.no_grad():
        output = model([tensor_image])


    return {
        "image":image,
        "output":output
    }

