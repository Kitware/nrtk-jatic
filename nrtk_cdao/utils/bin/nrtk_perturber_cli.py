import click  # type: ignore
from typing import TextIO, List
from pathlib import Path
import logging
import yaml  # type: ignore
import json

from PIL import Image  # type: ignore
import numpy as np
import kwcoco

from nrtk.impls.perturb_image.pybsm.scenario import PybsmScenario  # type: ignore
from nrtk.impls.perturb_image.pybsm.sensor import PybsmSensor  # type: ignore
from nrtk.impls.perturb_image_factory.pybsm import CustomPybsmPerturbImageFactory  # type: ignore
from nrtk_cdao.interop.dataset import COCOJATICObjectDetectionDataset
from nrtk_cdao.utils.nrtk_perturber import nrtk_perturber


@click.command(context_settings={"help_option_names": ['-h', '--help']})
@click.argument('dataset_dir', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path(exists=False))
@click.argument('config_file', type=click.File(mode='r'))
@click.argument('perturb_params_file', type=click.File(mode='r'))
@click.option('--verbose', '-v', count=True, help='print progress messages')
def nrtk_perturber_cli(
    dataset_dir: str,
    output_dir: str,
    config_file: TextIO,
    perturb_params_file: TextIO,
    verbose: bool
) -> None:
    """
    Generate NRTK perturbed images and detections from a given set of source images and
    COCO-format annotations and write them to an output folder in disk. The perturbed
    images are stored in subfolders named after the chosen perturbation parameter keys
    and values.

    \b
    DATASET_DIR - Root directory of dataset.
    OUTPUT_DIR - Directory to write the perturbed images to.
    CONFIG_FILE - Configuration file pertaining to a specific type of
        NRTK perturber.
    PERTURB_PARAMS_FILE - Configuration file containing the parameter value
        combinations to be implemented with the perturber factory.

    \f
    :param dataset_dir: Root directory of dataset.
    :param output_dir: Directory to write the perturbed images to.
    :param config_file: Configuration file pertaining to a specific type of
        NRTK perturber.
    :param perturb_params_file: Configuration file containing the values needed
        for the parameter sweep implemented with the perturber factory.
    :param verbose: Display progress messages. Default is false.
    """

    if verbose:
        logging.basicConfig(level=logging.INFO)

    logging.info(f"Dataset path: {dataset_dir}")

    logging.info("Loading kwcoco annotations")
    annotation_dir = Path(dataset_dir) / "annotations"

    coco_file = list(annotation_dir.glob("*.json"))
    kwcoco_dataset = kwcoco.CocoDataset(coco_file[0])

    # load config
    config = yaml.safe_load(config_file)

    if any(key not in config for key in ["gsd", "sensor", "scenario"]):
        raise ValueError("Invalid Configuration")

    # Load pybsm perturb params
    perturb_factory_config = yaml.safe_load(perturb_params_file)

    for key, value in config["sensor"].items():
        if isinstance(value, List):
            config["sensor"][key] = np.asarray(value)

    image_gsd = config["gsd"]
    sensor = PybsmSensor(**config["sensor"])
    scenario = PybsmScenario(**config["scenario"])

    # Initialize dataset object
    input_dataset = COCOJATICObjectDetectionDataset(
        root=dataset_dir,
        kwcoco_dataset=kwcoco_dataset,
        img_gsd=image_gsd  # A global GSD value is applied to each image
    )

    perturb_factory_keys = list(perturb_factory_config.keys())
    thetas = [perturb_factory_config[key]
              for key in perturb_factory_keys]

    # Set up custom pybsm perturber factory
    perturber_factory = CustomPybsmPerturbImageFactory(
        sensor=sensor,
        scenario=scenario,
        theta_keys=perturb_factory_keys,
        thetas=thetas
    )

    augmented_datasets = nrtk_perturber(
        maite_dataset=input_dataset,
        perturber_factory=perturber_factory
    )

    output_path = Path(output_dir)
    img_paths = input_dataset.get_img_path_list()
    for perturb_params, aug_dataset in augmented_datasets:
        (output_path / perturb_params).mkdir(parents=True, exist_ok=True)
        augmented_annotations = kwcoco.CocoDataset()
        for i in range(len(aug_dataset)):
            image, det, metadata = aug_dataset[i]

            # Setting pybsm config in updated metadata to 'None' since the output of get_config()
            # PyBSM perturber is (as of currently) not JSON serializable
            metadata.update({"pybsm_params": "None"})
            img_path = img_paths[i]
            im = Image.fromarray(image)
            im.save(output_path / perturb_params / (img_path.stem + img_path.suffix))
            labels = np.asarray(det.labels)
            boxes = np.asarray(det.boxes)
            augmented_annotations.add_images([{'id': i, 'file_name': img_path.stem + img_path.suffix}])
            for lbl, bbox in zip(labels, boxes.tolist()):
                augmented_annotations.add_annotation(
                    image_id=i,
                    category_id=int(lbl),
                    bbox=list(map(int, bbox))
                )
        logging.info(f"Saved perturbed images to {output_path / perturb_params}")
        with open(output_path / perturb_params / "image_metadata.json", "w") as f:
            json.dump([aug_dataset[d][2] for d in range(len(aug_dataset))], f)
        logging.info(f"Saved image_metadata to {output_path}/{perturb_params}/image_metadata.json")

        augmented_annotations.dump(output_path / perturb_params / "augmented_detections.json")
        logging.info(f"Saved augmented detections to {output_path}/{perturb_params}/augmented_detections.json")
