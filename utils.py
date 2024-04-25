import numpy as np
import cv2


def remove_duplicate_boxes(preds_results):
    "Remove dupplicate bounding box from predictions."

    post_preds = [
        preds["boxes"].cpu().detach().numpy()
        for preds in preds_results
        if preds["boxes"].numel()
    ]
    if post_preds:
        stakced_preds = np.vstack(post_preds)
        unique_boxes = np.unique(stakced_preds, axis=0)
        return unique_boxes
    else:
        return []


def save_prediction(boxes, image, save_name, save_dir):

    img = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    for box in boxes:
        box = [int(i) for i in box.tolist()]
        img = cv2.rectangle(img, box[:2], box[2:], (255, 0, 0), 5)
    cv2.imwrite(f"{save_dir}/{save_name}.jpg", img)
