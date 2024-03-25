import click  # type: ignore
from typing import TextIO, List
from pathlib import Path
import logging
import yaml  # type: ignore
import itertools
from importlib import find_loader

import numpy as np

from nrtk.impls.perturb_image.pybsm.scenario import PybsmScenario  # type: ignore
from nrtk.impls.perturb_image.pybsm.sensor import PybsmSensor  # type: ignore
from nrtk.impls.perturb_image_factory.pybsm import CustomPybsmPerturbImageFactory  # type: ignore
from nrtk_cdao.interop.dataset import JATICObjectDetectionDataset
from nrtk_cdao.utils.nrtk_perturber import nrtk_perturber


kwcoco_loader = find_loader('kwcoco')
kwcoco_is_usable = kwcoco_loader is not None

if kwcoco_is_usable:
    import kwcoco  # type: ignore


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
    Generate NRTK perturbed images from a given set of source images and write
    them to an output folder in disk. The perturbed images are stored in subfolders
    named after the chosen perturbation parameter keys and values.

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

    if not kwcoco_is_usable:
        raise ImportError("This tool requires additional dependencies, please install 'nrtk-cdao[tools]'")

    if verbose:
        logging.basicConfig(level=logging.INFO)

    logging.info(f"Dataset path: {dataset_dir}")

    logging.info("Loading kwcoco annotations")
    annotation_dir = Path(dataset_dir) / "annotations"

    coco_file = list(annotation_dir.glob("*.json"))
    kwcoco_dataset = kwcoco.CocoDataset(coco_file[0])

    # Initialize dataset object
    dataset = JATICObjectDetectionDataset(root=dataset_dir,
                                          kwcoco_dataset=kwcoco_dataset)

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

    # Set up custom pybsm perturber factory
    perturb_factory_keys = list(perturb_factory_config.keys())
    thetas = [perturb_factory_config[key]
              for key in perturb_factory_keys]
    perturber_combinations = [dict(zip(perturb_factory_keys, v))
                              for v in itertools.product(*thetas)]

    logging.info(f"Perturber sweep values: {perturber_combinations}")

    perturber_factory = CustomPybsmPerturbImageFactory(
        sensor=sensor,
        scenario=scenario,
        theta_keys=perturb_factory_keys,
        thetas=thetas
    )

    nrtk_perturber(
        maite_dataset=dataset,  # type: ignore
        output_dir=output_dir,
        perturber_combinations=perturber_combinations,
        perturber_factory=perturber_factory,
        image_gsd=image_gsd
    )
