from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
DATASET_FOLDER = DATA_DIR / "VisDrone2019-DET-test-dev-TINY"
LABEL_FILE = DATASET_FOLDER / "annotations/COCO_annotations_VisDrone_TINY.json"
NRTK_CONFIG = DATA_DIR / "nrtk_config.yaml"
BAD_NRTK_CONFIG = DATA_DIR / "bad_nrtk_config.yaml"
