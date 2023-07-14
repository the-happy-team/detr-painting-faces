import datasets
import datetime
import json
import os

HF_DATASET = "thiagohersan/cordiais-faces"
COCORDIAIS_PATH = "./cocordiais-hf"


def load_hf_dataset(dataset_name):
  dataset_ = datasets.load_dataset(dataset_name)
  return dataset_["train"].train_test_split(test_size=0.2, shuffle=True, seed=1010)


def get_categories(hf_dataset):
  hf_dataset_categories = hf_dataset["train"].features["objects"].feature["category"].names
  id2label = { i:l for i,l in enumerate(hf_dataset_categories) }
  id2superlabel = { i:l if l == 'N/A' else "face" for i,l in id2label.items() }
  return [
    { "id": i, "name": l, "supercategory": id2superlabel[i] } for i,l in id2label.items()
  ]


def get_cocordiais_info(hf_dataset):
  return {
    "info": {
      "year": 2023,
      "version": "1.0.0",
      "description": "Object Detection dataset to detect female-ish faces in paintings",
      "contributor": "Thiago Hersan",
      "url": "https://huggingface.co/datasets/thiagohersan/cordiais-faces",
      "date_created": "%s" % datetime.datetime.now()
    },
    "categories": get_categories(hf_dataset),
    "licenses": [
      { "id": 1, "name": "CC BY-NC 2.0", "url": "https://creativecommons.org/licenses/by-nc/2.0/" },
      { "id": 2, "name": "CC0 1.0", "url": "https://creativecommons.org/publicdomain/zero/1.0/" },
    ],
    "images": [],
    "annotations": []
  }


def create_cocordiais_from_hf(hf_dataset, cocordiais_path):
  for split_name, objs in hf_dataset.items():
    split_data_path = os.path.join(cocordiais_path, split_name)
    os.makedirs(split_data_path)

    json_path_out = os.path.join(split_data_path, "cocordiais.json")
    cocordiais_obj = get_cocordiais_info(hf_dataset)

    for obj in objs:
      img_path = os.path.join(split_data_path, obj["image_filename"])

      if not os.path.isfile(img_path):
        obj["image"].save(img_path, "JPEG")

      cocordiais_obj["images"].append({
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
        cocordiais_obj["annotations"].append({
          "id": ann_objs["bbox_id"][ann_idx],
          "image_id": obj["image_id"],
          "category_id": ann_objs["category"][ann_idx],
          "area": ann_objs["area"][ann_idx],
          "bbox": ann_objs["bbox"][ann_idx],
          "iscrowd": 1 if ann_objs["is_crowd"][ann_idx] else 0
        })

    with open(json_path_out, 'w') as json_file_out_write:
      json.dump(cocordiais_obj, json_file_out_write)

if __name__ == "__main__":
  hf_dataset = load_hf_dataset(HF_DATASET)
  create_cocordiais_from_hf(hf_dataset, COCORDIAIS_PATH)
