import kwcoco
import json
import os

from smqtk_core.configuration import from_config_dict

from nrtk.interfaces.perturb_image_factory import PerturbImageFactory
from nrtk_cdao.interop.dataset import COCOJATICObjectDetectionDataset
from nrtk_cdao.api.schema import NrtkPerturbInputSchema


def build_factory(data: NrtkPerturbInputSchema) -> PerturbImageFactory:
    """
    Returns a CustomPybsmPerturbImageFactory based on scenario and sensor parameters in data

    :param data: dictionary of Schema from schema.py
    """

    if not os.path.isfile(data.config_file):
        raise FileNotFoundError("Config file at {0} was not found".format(data.config_file))
    with open(data.config_file) as config_file:
        config = json.load(config_file)
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
        image_metadata=[{"img_gsd": gsd} for gsd in data.image_metadata["gsds"]],
    )

    return dataset
