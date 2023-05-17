import cv2
import json
import shutil
import numpy as np
import matplotlib.pyplot as plt

def impaint_img(img_path, saturation_min=40, saturation_max=255):
    input_img = cv2.imread(img_path)
    hsv_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2HSV)

    # Create mask image 
    lower_val = np.array([0, saturation_min, 0])
    upper_val = np.array([360, saturation_max, 255])
    mask = cv2.inRange(hsv_img, lower_val, upper_val)

    # Dilate mask
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Impainting
    dst_img = cv2.inpaint(input_img, mask, 5, cv2.INPAINT_TELEA)

    # Save img
    img_name = img_path.split('.')[0]
    ext = img_path.split('.')[1]
    cv2.imwrite(f"{img_name}_out.{ext}", dst_img)


def manifest_reader(manifest_path):
    for line in open(manifest_path, mode="r"):
        yield json.loads(line)


def convert_bbox_manifest(manifest_path, job_name, output_coco_json_path,
                          dataset_type="train", ignore_s3_path=True):
    """
    Converts a single bounding box manifest file into COCO format.
    :param manifest_path: Path of the GT manifest file
    :param job_name: Name of the GT job
    :param output_coco_json_path: Output path for converted COCO json
    :param dataset_type: Dataset type (Default: train)
    :param ignore_s3_path: Delete s3 path if this parameter is true (Default: True)
    """
    assert(dataset_type in ["train", "valid", "test"])

    image_id = 0
    annotation_id = 0
    annotations = []
    images = []
    category_ids = {}

    for annotation in manifest_reader(manifest_path):
        w = annotation[job_name]["image_size"][0]["width"]
        h = annotation[job_name]["image_size"][0]["height"]

        if ignore_s3_path:
            file_name = annotation["source-ref"].split('/')[-1]
        else:
            file_name = annotation["source-ref"]
        file_name = f"images/{dataset_type}/{file_name}"
        images.append(
            {
                "file_name": file_name,
                "height": h,
                "width": w,
                "id": image_id,
            }
        )

        for bbox in annotation[job_name]["annotations"]:
            coco_bbox = {
                "iscrowd": 0,
                "image_id": image_id,
                "category_id": bbox["class_id"],
                "id": annotation_id,
                "bbox": (bbox["left"], bbox["top"], bbox["width"], bbox["height"]),
                "area": bbox["width"] * bbox["height"],
            }
            class_map = annotation[job_name + "-metadata"]["class-map"]
            class_map_coco = [{"id":int(k), "name":v, "supercategory":"none"} for k,v in class_map.items()]

            annotations.append(coco_bbox)
            annotation_id += 1
            #category_ids.update(class_map_coco)

        image_id += 1

    coco_json = {
        "type": "instances",
        "images": images,
        "categories": class_map_coco,
        "annotations": annotations,
    }

    with open(output_coco_json_path, "w") as f:
        json.dump(coco_json, f)


def show_detection_result(img_path, img_result, conf_threshold=0.5):
    from autogluon.multimodal.utils import Visualizer
    from PIL import Image
    from IPython.display import display

    visualizer = Visualizer(img_path)  # Initialize the Visualizer
    out = visualizer.draw_instance_predictions(img_result, conf_threshold)
    visualized = out.get_image()
    img = Image.fromarray(visualized, 'RGB')
    display(img)