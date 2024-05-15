import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection


def load_model(device):
    
    if device == "cuda":
        selected_dtype = torch.float16
    else:
        selected_dtype = torch.float32
        
    model = OwlViTForObjectDetection.from_pretrained(
        "google/owlvit-base-patch32",
        torch_dtype=selected_dtype).to(device)
    
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    model = torch.compile(model)

    return model, processor
