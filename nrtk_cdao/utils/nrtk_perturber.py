from typing import List, Dict, Any, Tuple, Union
from pathlib import Path
import logging
import json

from PIL import Image  # type: ignore

from nrtk.interfaces.perturb_image_factory import PerturbImageFactory  # type: ignore
from nrtk_cdao.interop.augmentation import JATICAugmentation
from nrtk_cdao.interop.dataset import JATICObjectDetectionDataset
from maite.protocols import ArrayLike
from maite.protocols.object_detection import ObjectDetectionTarget


def nrtk_perturber(
    maite_dataset: Union[JATICObjectDetectionDataset,
                         Tuple[ArrayLike, ObjectDetectionTarget,
                               Dict[str, Any]]],
    output_dir: str,
    perturber_combinations: List[Dict],
    perturber_factory: PerturbImageFactory,
    **kwargs: Any
) -> None:
    """
    Generate NRTK perturbed images from a given set of source images and write
    them to an output folder in disk. The perturbed images are stored in subfolders
    named after the chosen perturbation parameter keys and values.

    \b
    OUTPUT_DIR - Directory to write the perturbed images to.

    \f
    :param dataset_dir: Root directory of dataset.
    :param output_dir: Directory to write the perturbed images to.
    :param perturber_combinations: Perturber parameter sweep combinations.
    :param perturb_factory_config: PerturbImageFactory implementation.
    """

    img_data: List[ArrayLike] = []
    img_metadata: List[Dict] = []
    for img, _, metadata in maite_dataset:  # type: ignore
        logging.info(metadata)
        img_metadata.append(metadata)
        img_data.append(img.detach().cpu())

    # Iterate through the different perturber factory parameter combinations and
    # save the perturbed images to disk
    logging.info("Starting perturber sweep")
    for perturber_combo, perturber in zip(perturber_combinations, perturber_factory):
        output_perturb_params = ''.join('_' + str(k) + '-' + str(v)
                                        for k, v in perturber_combo.items())
        Path(output_dir + '/' + output_perturb_params).mkdir(parents=True, exist_ok=True)

        logging.info(f"Starting perturbation for {output_perturb_params}")

        for img, img_path in zip(img_data, maite_dataset.get_img_path_list()):  # type: ignore
            augmented_img = JATICAugmentation(augment=[perturber],
                                              gsd=kwargs["image_gsd"]).__call__(
                                                            data=img)
            im = Image.fromarray(augmented_img)
            im.save(output_dir + '/' +
                    output_perturb_params + '/' +
                    img_path.stem + img_path.suffix)

        logging.info(f"Saved perturbed images to {output_dir + '/' + output_perturb_params}")

    with open((Path(output_dir) / "image_metadata.json"), "w") as f:
        json.dump(img_metadata, f)

    logging.info(f"Saved image_metadata to {output_dir}/image_metadata.json")
