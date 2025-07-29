import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


##Load FasterRCNN Model
def get_model(num_classes=2, weight_path="/kaggle/input/model_fasterrcnn/pytorch/default/1/best_weights.pth", freeze_backbone=True):
    # Load model without pretrained weights to avoid mismatch
    model = fasterrcnn_resnet50_fpn(weights=None)

    # Replace the classifier head for your num_classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Load weights
    state_dict = torch.load(weight_path, map_location="cpu")
    model.load_state_dict(state_dict)
    if freeze_backbone:
        # Freeze all parameters except the box_predictor head
        for name, parameter in model.named_parameters():
            if "box_predictor" not in name:
                parameter.requires_grad = False
            else:
                parameter.requires_grad = True

    return model