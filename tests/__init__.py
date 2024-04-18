from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
DATASET_FOLDER = DATA_DIR / "VisDrone2019-DET-test-dev-TINY"
LABEL_FILE = DATASET_FOLDER / "annotations/COCO_annotations_VisDrone_TINY.json"
