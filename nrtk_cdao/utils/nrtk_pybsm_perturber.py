from typing import List, Dict
from pathlib import Path
import glob
import logging
import itertools

import numpy as np
from PIL import Image  # type: ignore

from nrtk.impls.perturb_image.pybsm.sensor import PybsmSensor  # type: ignore
from nrtk.impls.perturb_image.pybsm.scenario import PybsmScenario  # type: ignore
from nrtk.impls.perturb_image_factory.pybsm import CustomPybsmPerturbImageFactory  # type: ignore
from nrtk_cdao.interop.augmentation import JATICAugmentation
from nrtk_cdao.interop.dataset import CustomMAITEDataset
from maite.protocols import HasDataImage


def nrtk_pybsm_perturber(
    dataset_img_dir: str,
    output_dir: str,
    config: Dict,
    perturb_factory_config: Dict,
    verbose: bool
) -> None:
    """
    Generate NRTK perturbed images from a given set of source images and write
    them to an output folder in disk. The perturbed images are stored in subfolders
    named after the chosen perturbation parameter keys and values.

    \b
    DATASET_IMG_DIR - Directory where the source images are located.
    OUTPUT_DIR - Directory to write the perturbed images to.

    \f
    :param dataset_img_dir: Directory where the source images are located.
    :param output_dir: Directory to write the perturbed images to.
    :param config: PyBSM config dictionary.
    :param perturb_factory_config: Perturber factory parameter dictionary.
    :param verbose: Display progress messages. Default is false.
    """

    if verbose:
        logging.basicConfig(level=logging.INFO)

    logging.info(f"Dataset path: {dataset_img_dir}")

    # Checking if all essential pybsm configurations exist
    if any(key not in config for key in ["gsd", "sensor", "scenario"]):
        raise ValueError("Invalid Configuration")

    # Convert list values to numpy arrays in sensor config
    for key, value in config["sensor"].items():
        if isinstance(value, List):
            config["sensor"][key] = np.asarray(value)

    # Initialize pybsm sensor and scenario objects from config
    image_gsd = config["gsd"]
    sensor = PybsmSensor(**config["sensor"])
    scenario = PybsmScenario(**config["scenario"])

    file_exts = ['jpg', 'JPG', 'png', 'PNG']
    img_path_list = sorted([filename for ext in file_exts
                            for filename in glob.glob(dataset_img_dir
                                                      + '/*.' + ext)])
    img_paths: List[Path] = [Path(im_path) for im_path in img_path_list]

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

    perturbed_images: HasDataImage = {'image': []}
    # Iterate through the different perturber factory parameter combinations and
    # save the perturbed images to disk
    logging.info("Starting perturber sweep")
    for perturber_combo, perturber in zip(perturber_combinations, perturber_factory):
        output_perturb_params = ''.join('_' + str(k) + '-' + str(v)
                                        for k, v in perturber_combo.items())
        logging.info(f"Starting perturbation for {output_perturb_params}")
        perturbed_images = JATICAugmentation(augment=[perturber],
                                             gsd=image_gsd).__call__(
                                             CustomMAITEDataset(img_paths=img_paths))

        Path(output_dir + '/' + output_perturb_params).mkdir(parents=True, exist_ok=True)

        logging.info(f"Saving perturbed images to {output_dir + '/' + output_perturb_params}")
        for perturbed_image, img_path in zip(perturbed_images["image"], img_paths):  # type: ignore
            im = Image.fromarray(perturbed_image)
            im.save(output_dir + '/' +
                    output_perturb_params + '/' +
                    img_path.stem + img_path.suffix)
