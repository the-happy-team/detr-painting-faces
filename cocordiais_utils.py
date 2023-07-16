import datetime
import json

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
  "categories": [
    { "id": i, "name": l, "supercategory": ID2SUPERLABEL[i] } for i,l in ID2LABEL.items()
  ],
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
  return json.loads(json.dumps(COCORDIAIS_DATASET_INFO))
