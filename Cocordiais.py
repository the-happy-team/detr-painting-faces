import torch
import torchvision.transforms as T


class CocordiaisData():
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

  def __init__(self, img_processor):
    self.img_processor = img_processor
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  def to_coco(self, train=True):
    def to_coco_fun(examples):
      image_ids = examples["image_id"]
      images, bboxes, area, categories = [], [], [], []

      for image, objects in zip(examples["image"], examples["objects"]):
        if train:
          image = T.PILToTensor()(image).to(self.device)
          image = CocordiaisData.transform(image)

        area.append(objects["area"])
        images.append(image)
        bboxes.append(objects["bbox"])
        categories.append(objects["category"])

      targets = [
        {"image_id": id_, "annotations": CocordiaisData.to_coco_annotation(id_, cat_, ar_, box_)}
        for id_, cat_, ar_, box_ in zip(image_ids, categories, area, bboxes)
      ]

      return self.img_processor(images=images, annotations=targets, return_tensors="pt")
    return to_coco_fun

  def collate_batch(self, batch):
    pixel_values = [item["pixel_values"] for item in batch]
    encoding = self.img_processor.pad(pixel_values, return_tensors="pt")
    labels = [item["labels"] for item in batch]
    batch = {}
    batch["pixel_values"] = encoding["pixel_values"]
    batch["pixel_mask"] = encoding["pixel_mask"]
    batch["labels"] = labels
    return batch
