import cocordiais_utils as cocordiais
import datasets
import json
import os

HF_DATASET = "thiagohersan/cordiais-faces"
COCORDIAIS_PATH = "./cocordiais"


def create_cocordiais_from_hf(hf_dataset_name, cocordiais_path):
  hf_dataset_ = datasets.load_dataset(hf_dataset_name)
  hf_dataset = hf_dataset_["train"].train_test_split(test_size=0.2, shuffle=True, seed=101010)

  for split_name, objs in hf_dataset.items():
    split_data_path = os.path.join(cocordiais_path, split_name)
    os.makedirs(split_data_path)

    json_path_out = os.path.join(split_data_path, "cocordiais.json")
    cocordiais_info = cocordiais.get_cocordiais_info()

    for obj in objs:
      img_path = os.path.join(split_data_path, obj["image_filename"])

      if not os.path.isfile(img_path):
        obj["image"].save(img_path, "JPEG")

      cocordiais_info["images"].append({
        "id": obj["image_id"],
        "width": obj["width"],
        "height": obj["height"],
        "file_name": obj["image_filename"],
        "license": obj["license_id"],
        "date_captured": obj["date_captured"],
        "coco_url": "",
        "flickr_url": ""
      })

      ann_objs = obj["objects"]
      for ann_idx in range(len(ann_objs["bbox_id"])):
        cocordiais_info["annotations"].append({
          "id": ann_objs["bbox_id"][ann_idx],
          "image_id": obj["image_id"],
          "category_id": ann_objs["category"][ann_idx],
          "area": ann_objs["area"][ann_idx],
          "bbox": ann_objs["bbox"][ann_idx],
          "iscrowd": 1 if ann_objs["is_crowd"][ann_idx] else 0
        })

    with open(json_path_out, 'w') as json_file_out_write:
      json.dump(cocordiais_info, json_file_out_write)

if __name__ == "__main__":
  create_cocordiais_from_hf(HF_DATASET, COCORDIAIS_PATH)
