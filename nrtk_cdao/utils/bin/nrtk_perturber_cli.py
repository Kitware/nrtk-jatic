import click  # type: ignore
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, TextIO

from smqtk_core.configuration import from_config_dict, make_default_config

from nrtk.interfaces.perturb_image_factory import PerturbImageFactory
from nrtk_cdao.utils.nrtk_perturber import nrtk_perturber

try:
    from nrtk_cdao.interop.object_detection.utils import dataset_to_coco
    from nrtk_cdao.interop.object_detection.dataset import COCOJATICObjectDetectionDataset
    import kwcoco  # type: ignore
    is_usable = True
except ImportError:
    is_usable = False


@click.command(context_settings={"help_option_names": ['-h', '--help']})
@click.argument('dataset_dir', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path(exists=False))
@click.argument('config_file', type=click.File(mode='r'))
@click.option('-g', '--generate-config-file', help='write default config to specified file', type=click.File(mode='w'))
@click.option('--verbose', '-v', count=True, help='print progress messages')
def nrtk_perturber_cli(
    dataset_dir: str,
    output_dir: str,
    config_file: TextIO,
    generate_config_file: TextIO,
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
    CONFIG_FILE - Configuration file specifying the PerturbImageFactory configuration.

    \f
    :param dataset_dir: Root directory of dataset.
    :param output_dir: Directory to write the perturbed images to.
    :param config_file: Configuration file specifying the PerturbImageFactory configuration.
    :param generate_config_file: File to write default config file, only written
        if specified.
    :param verbose: Display progress messages. Default is false.
    """

    if generate_config_file:
        config: Dict[str, Any] = dict()
        config["PerturberFactory"] = make_default_config(PerturbImageFactory.get_impls())
        json.dump(config, generate_config_file, indent=4)

        exit()

    if verbose:
        logging.basicConfig(level=logging.INFO)

    logging.info(f"Dataset path: {dataset_dir}")

    # Load COCO dataset
    coco_file = Path(dataset_dir) / "annotations.json"
    if not coco_file.is_file():
        raise ValueError("Could not identify annotations file. Expected at '[dataset_dir]/annotations.json'")
    logging.info(f"Loading kwcoco annotations from {coco_file}")
    if not is_usable:
        print("This tool requires additional dependencies, please install `nrtk-cdao[tools]`")
        exit(-1)
    kwcoco_dataset = kwcoco.CocoDataset(coco_file)

    # Load metadata, if it exists
    metadata_file = Path(dataset_dir) / "image_metadata.json"
    if not metadata_file.is_file():
        logging.warn("Could not identify metadata file, assuming no metadata. "
                     "Expected at '[dataset_dir]/image_metadata.json'")
        metadata: List[Dict[str, Any]] = [dict() for _ in range(len(kwcoco_dataset.imgs))]
    else:
        logging.info(f"Loading metadata from {metadata_file}")
        with open(metadata_file) as f:
            metadata = json.load(f)

    # Load config
    config = json.load(config_file)
    perturber_factory = from_config_dict(config["PerturberFactory"], PerturbImageFactory.get_impls())

    # Initialize dataset object
    input_dataset = COCOJATICObjectDetectionDataset(
        root=dataset_dir,
        kwcoco_dataset=kwcoco_dataset,
        image_metadata=metadata
    )

    # Augment input dataset
    augmented_datasets = nrtk_perturber(
        maite_dataset=input_dataset,
        perturber_factory=perturber_factory
    )

    # Save each augmented dataset to its own directory
    output_path = Path(output_dir)
    img_filenames = [Path(img_path.name) for img_path in input_dataset.get_img_path_list()]
    for perturb_params, aug_dataset in augmented_datasets:
        dataset_to_coco(
            dataset=aug_dataset,
            output_dir=output_path / perturb_params,
            img_filenames=img_filenames,
            dataset_categories=input_dataset.get_categories()
        )
