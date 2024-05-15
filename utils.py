from pathlib import Path
import numpy as np
import cv2
import pybboxes as pbx




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
    


def save_annotation(boxes ,raw_image, raw_image_path , class_name):
    "Save boounding box annotation in YOLO format."
    
    save_path = "datasets/annotated_dataset"
    annotation_file = Path(raw_image_path.replace(".jpg" , ".txt")).name
    annotation_path = f"{save_path}/{annotation_file}"
    width, height = raw_image.size
    raw_image.save(f"{save_path}/{Path(raw_image_path).name}")
    with open(annotation_path, "w") as f:
        for box in boxes:
            # yolo_bbox = pbx.convert_bbox(box, 
            #                  from_type="coco", 
            #                  to_type="yolo",
            #                  image_width=width,
            #                  image_height=height)
            formatted_box = " ".join([str(point) for point in box])
            f.write(f"{class_name} {formatted_box}\n")
