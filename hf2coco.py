import datasets
import datetime
import json
import os

HF_DATASET = "thiagohersan/cordiais-faces"

hf_dataset_ = datasets.load_dataset(HF_DATASET)
hf_dataset = hf_dataset_["train"].train_test_split(test_size=0.2, shuffle=True, seed=1010)
hf_dataset_categories = hf_dataset["train"].features["objects"].feature["category"].names

COCORDIAIS_INFO = {
  "info": {
    "year": 2023,
    "version": "1.0.0",
    "description": "Object Detection dataset to detect female-ish faces in paintings",
    "contributor": "Thiago Hersan",
    "url": "https://huggingface.co/datasets/thiagohersan/cordiais-faces",
    "date_created": "%s" % datetime.datetime.now()
  },
  "categories": [],
  "licenses": [
    { "id": 1, "name": "CC BY-NC 2.0", "url": "https://creativecommons.org/licenses/by-nc/2.0/" },
    { "id": 2, "name": "CC0 1.0", "url": "https://creativecommons.org/publicdomain/zero/1.0/" },
  ],
  "images": [],
  "annotations": []
}

ID2LABEL = { i:l for i,l in enumerate(hf_dataset_categories) }
ID2SUPERLABELS = { i:l if l == 'N/A' else "face" for i,l in ID2LABEL.items() }

COCORDIAIS_INFO["categories"] = [
  { "id": i, "name": l, "supercategory": ID2SUPERLABELS[i] } for i,l in ID2LABEL.items()
]

COCORDIAIS_PATH = "./cocordiais-hf"

for split_name, objs in hf_dataset.items():
  split_data_path = os.path.join(COCORDIAIS_PATH, split_name)
  os.makedirs(split_data_path)

  json_path_out = os.path.join(split_data_path, "cocordiais.json")
  cocordiais_obj = json.loads(json.dumps(COCORDIAIS_INFO))

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

