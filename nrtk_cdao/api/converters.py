import numpy as np
import kwcoco
from typing import List

from nrtk.impls.perturb_image.pybsm.scenario import PybsmScenario  # type: ignore
from nrtk.impls.perturb_image.pybsm.sensor import PybsmSensor  # type: ignore
from nrtk.impls.perturb_image_factory.pybsm import CustomPybsmPerturbImageFactory  # type: ignore
from nrtk_cdao.interop.dataset import COCOJATICObjectDetectionDataset
from nrtk_cdao.api.schema import NrtkPybsmPerturbInputSchema


def build_pybsm_factory(data: NrtkPybsmPerturbInputSchema) -> CustomPybsmPerturbImageFactory:
    """
    Returns a CustomPybsmPerturbImageFactory based on scenario and sensor parameters in data

    :param data: dictionary of Schema from schema.py
    """
    data_dict = dict(data)
    scenario_params = PybsmScenario.get_default_config()
    sensor_params = PybsmSensor.get_default_config()

    scenario_params.update(
        {key: data_dict[key] for key in scenario_params if key in data_dict and data_dict[key] is not None}
    )
    sensor_params.update(
        {key: data_dict[key] for key in sensor_params if key in data_dict and data_dict[key] is not None}
    )

    # Convert list to np.arrays. Should change to from_config_dict
    # (https://github.com/Kitware/SMQTK-Core/blob/master/smqtk_core/configuration.py#L443) when possible
    for key in sensor_params:
        if isinstance(sensor_params[key], List):
            sensor_params[key] = np.asarray(sensor_params[key])

    sensor = PybsmSensor(**sensor_params)
    scenario = PybsmScenario(**scenario_params)

    perturber_factory = CustomPybsmPerturbImageFactory(
        sensor=sensor, scenario=scenario, theta_keys=data.theta_keys, thetas=data.thetas
    )

    return perturber_factory


def load_COCOJAITIC_dataset(data: NrtkPybsmPerturbInputSchema) -> COCOJATICObjectDetectionDataset:
    """
    Returns a COCOJATICObjectDetectionDataset based on dataset parameters in data

    :param data: dictionary of Schema from schema.py
    """
    kwcoco_dataset = kwcoco.CocoDataset(data.label_file)

    dataset = COCOJATICObjectDetectionDataset(
        root=data.dataset_dir,
        kwcoco_dataset=kwcoco_dataset,
        image_metadata=[{"img_gsd": gsd} for gsd in data.gsds],
    )

    return dataset
