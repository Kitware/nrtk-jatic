import kwcoco
from typing import Dict, TYPE_CHECKING

from smqtk_core.configuration import from_config_dict, make_default_config

from nrtk.impls.perturb_image.pybsm.scenario import PybsmScenario
from nrtk.impls.perturb_image.pybsm.sensor import PybsmSensor
from nrtk.impls.perturb_image_factory.pybsm import CustomPybsmPerturbImageFactory
from nrtk_cdao.interop.dataset import COCOJATICObjectDetectionDataset
from nrtk_cdao.api.schema import NrtkPybsmPerturbInputSchema


def build_pybsm_factory(data: NrtkPybsmPerturbInputSchema) -> CustomPybsmPerturbImageFactory:
    """
    Returns a CustomPybsmPerturbImageFactory based on scenario and sensor parameters in data

    :param data: dictionary of Schema from schema.py
    """
    data_dict = dict(data)
    scenario_config = make_default_config([PybsmScenario])
    scenario_impl = "nrtk.impls.perturb_image.pybsm.scenario.PybsmScenario"
    scenario_config["type"] = scenario_impl
    sensor_config = make_default_config([PybsmSensor])
    sensor_impl = "nrtk.impls.perturb_image.pybsm.sensor.PybsmSensor"
    sensor_config["type"] = sensor_impl

    scenario_params = scenario_config[scenario_impl]
    if TYPE_CHECKING:
        assert isinstance(scenario_params, Dict)
    scenario_params.update(
        {
            key: data_dict[key]
            for key in scenario_params
            if key in data_dict and data_dict[key] is not None
        }
    )
    sensor_params = sensor_config[sensor_impl]
    if TYPE_CHECKING:
        assert isinstance(sensor_params, Dict)
    sensor_params.update(
        {
            key: data_dict[key]
            for key in sensor_params
            if key in data_dict and data_dict[key] is not None
        }
    )

    sensor = from_config_dict(sensor_config, [PybsmSensor])
    scenario = from_config_dict(scenario_config, [PybsmScenario])

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
