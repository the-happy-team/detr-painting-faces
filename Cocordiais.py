import datetime
import json

import torch
import torchvision.transforms as T

class CocordiaisUtils():
  COCORDIAIS_LABELS = [
    "female",
    "not-female"
  ]

  COCORDIAIS_SUPERLABELS = [
    "face",
    "face"
  ]

  ID2LABEL = {i:l for i,l in enumerate(COCORDIAIS_LABELS)}
  ID2SUPERLABEL = {i:l for i,l in enumerate(COCORDIAIS_SUPERLABELS)}

  LABEL2ID = {v:int(k) for k,v in ID2LABEL.items()}
  LABEL2SUPERLABEL = {l:sl for l,sl in zip(COCORDIAIS_LABELS, COCORDIAIS_SUPERLABELS)}

  SUPERLABEL2SUPERID = {sl:si for si,sl in enumerate(set([l for l in ID2SUPERLABEL.values()]))}

  COCORDIAIS_DATASET_INFO = {
    "info": {
      "year": 2023,
      "version": "1.0.0",
      "description": "Object Detection dataset to detect female-ish faces in paintings",
      "contributor": "Thiago Hersan",
      "url": "https://huggingface.co/datasets/thiagohersan/cordiais-faces",
      "date_created": "%s" % datetime.datetime.now(),
    },
    "categories": [],
    "licenses": [
      { "id": 0, "name": "CC0 1.0", "url": "https://creativecommons.org/publicdomain/zero/1.0/", },
      { "id": 1, "name": "CC BY-NC 2.0", "url": "https://creativecommons.org/licenses/by-nc/2.0/", }
    ],
    "references": [
      { "id": 0, "name": "Training Generative Adversarial Networks with Limited Data", "url": "https://doi.org/10.48550/arXiv.2006.06676" }
    ],
    "images": [],
    "annotations": [],
  }

  def get_cocordiais_info():
    return json.loads(json.dumps(CocordiaisUtils.COCORDIAIS_DATASET_INFO))

CocordiaisUtils.COCORDIAIS_DATASET_INFO["categories"] = [
  { "id": i, "name": l, "supercategory": CocordiaisUtils.ID2SUPERLABEL[i] } for i,l in CocordiaisUtils.ID2LABEL.items()
]


class CocordiaisDataset():
  def GaussianNoise(sigma=25.0):
    def gauss_noise(img):
      dtype = img.dtype
      if not img.is_floating_point():
        img = img.to(torch.float32)

      out = img + sigma * torch.randn_like(img)

      if out.dtype != dtype:
         out = out.to(dtype)
      return out
    return gauss_noise

  transform = T.Compose([
    T.ColorJitter(brightness=0.5, hue=0.3),
    T.RandomSolarize(threshold=200.0),
    T.RandomEqualize(),
    GaussianNoise(sigma=20.0),
    T.RandomPosterize(bits=4),
    T.GaussianBlur(kernel_size=5, sigma=(0.1, 5))
  ])

  def to_coco_annotation(image_id, category, area, bbox):
    annotations = []
    for i in range(0, len(category)):
      new_ann = {
        "image_id": image_id,
        "category_id": category[i],
        "isCrowd": 0,
        "area": area[i],
        "bbox": list(bbox[i]),
      }
      annotations.append(new_ann)

    return annotations

  def __init__(self, dataset, img_processor, train):
    self.img_processor = img_processor
    self.train = train
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.data = dataset.with_transform(self.to_coco)

  def __len__(self):
      return len(self.data)

  def to_coco(self, examples):
    image_ids = examples["image_id"]
    images, bboxes, area, categories = [], [], [], []

    for image, objects in zip(examples["image"], examples["objects"]):
      image = T.PILToTensor()(image).to(self.device)

      if self.train:
        image = CocordiaisDataset.transform(image)

      area.append(objects["area"])
      images.append(image)
      bboxes.append(objects["bbox"])
      categories.append(objects["category"])

    targets = [
      {"image_id": id_, "annotations": CocordiaisDataset.to_coco_annotation(id_, cat_, ar_, box_)}
      for id_, cat_, ar_, box_ in zip(image_ids, categories, area, bboxes)
    ]

    return self.img_processor(images=images, annotations=targets, return_tensors="pt")

  def collate_batch(self, batch):
    pixel_values = [item["pixel_values"] for item in batch]
    encoding = self.img_processor.pad(pixel_values, return_tensors="pt")
    labels = [item["labels"] for item in batch]
    batch = {}
    batch["pixel_values"] = encoding["pixel_values"]
    batch["pixel_mask"] = encoding["pixel_mask"]
    batch["labels"] = labels
    return batch
