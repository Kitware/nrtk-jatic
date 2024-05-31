import kwcoco
import json
import os

from smqtk_core.configuration import from_config_dict

from nrtk.interfaces.perturb_image_factory import PerturbImageFactory
from nrtk_cdao.interop.object_detection.dataset import COCOJATICObjectDetectionDataset
from nrtk_cdao.api.schema import NrtkPerturbInputSchema


def build_factory(data: NrtkPerturbInputSchema) -> PerturbImageFactory:
    """
    Returns a PerturbImageFactory based on scenario and sensor parameters in data

    :param data: dictionary of Schema from schema.py
    """

    if not os.path.isfile(data.config_file):
        raise FileNotFoundError(f"Config file at {data.config_file} was not found")
    with open(data.config_file) as config_file:
        config = json.load(config_file)
        if "PerturberFactory" not in config.keys():
            raise ValueError(f"Config file at {data.config_file} does not have \"PerturberFactory\" key")
        perturber_factory = from_config_dict(config["PerturberFactory"], PerturbImageFactory.get_impls())

    return perturber_factory


def load_COCOJATIC_dataset(data: NrtkPerturbInputSchema) -> COCOJATICObjectDetectionDataset:
    """
    Returns a COCOJATICObjectDetectionDataset based on dataset parameters in data

    :param data: dictionary of Schema from schema.py
    """
    kwcoco_dataset = kwcoco.CocoDataset(data.label_file)

    dataset = COCOJATICObjectDetectionDataset(
        root=data.dataset_dir,
        kwcoco_dataset=kwcoco_dataset,
        image_metadata=data.image_metadata,
    )

    return dataset
