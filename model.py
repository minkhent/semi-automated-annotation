import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection


# Use GPU if available
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def load_model():

    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(
        device
    )
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    model = torch.compile(model)

    return model, processor
