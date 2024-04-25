import time
import os
import torch
import numpy as np
from glob import glob
from PIL import Image
from tqdm import tqdm
from model import load_model
from utils import remove_duplicate_boxes, save_prediction

device = torch.device("cuda:0" if torch.cuda.is_available() 
                      else "cpu")
model, processor = load_model(device)


def inference_batch(image, query_images , device):

    inputs = processor(images=image, query_images=query_images, return_tensors="pt").to(
        device
    )
    if device == "cuda":
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
            batch_outputs = model.image_guided_detection(**inputs)
    else:
        with torch.no_grad():
            batch_outputs = model.image_guided_detection(**inputs)

    target_sizes = torch.Tensor([image.size[::-1]] * len(query_images)).to(device)
    results = processor.post_process_image_guided_detection(
        outputs=batch_outputs,
        threshold=0.7,
        nms_threshold=0.5,
        target_sizes=target_sizes,
    )

    return results


def main():

    # RAW images
    raw_images = glob("datasets/kangroo_raw_data/*.jpg")

    # Query images
    query_paths = glob("object_patches/kangroo/*.jpg")
    query_images = [
        Image.fromarray(np.uint8(Image.open(img))).convert("RGB") for img in query_paths
    ]

    from time import gmtime, strftime

    created_time = strftime("%Y-%m-%d_%H:%M:%S", gmtime())
    save_dir = f"exp/runs/{created_time}"
    os.makedirs(save_dir)

    with tqdm(total=len(raw_images)) as pbar:
        for raw_image in raw_images:
            raw_image = Image.fromarray(np.uint8(Image.open(raw_image))).convert("RGB")
            pred_results = inference_batch(raw_image, query_images , device)

            unique_boxes = remove_duplicate_boxes(pred_results)
            if len(unique_boxes):
                save_name = str(round(time.time() * 1000))
                save_prediction(unique_boxes, raw_image, save_name, save_dir)
            pbar.update(1)


if __name__ == "__main__":
    main()
