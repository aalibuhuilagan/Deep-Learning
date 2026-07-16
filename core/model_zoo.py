import torchvision.models as models
from config import DEVICE

def load_model(model_name="resnet50"):
    model = None
    target_layer = None

    if model_name == "resnet50":
        model = models.resnet50(weights="DEFAULT")
        target_layer = [model.layer4]

    elif model_name == "vgg16":
        model = models.vgg16(weights="DEFAULT")
        target_layer = [model.features[-1]]

    elif model_name == "vit_b_16":
        model = models.vit_b_16(weights="DEFAULT")
        target_layer = [model.encoder.layers[-1]]

    model = model.to(DEVICE)
    model.eval()
    return model, target_layer