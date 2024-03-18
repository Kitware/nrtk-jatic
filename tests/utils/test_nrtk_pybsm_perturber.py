import os
import py  # type: ignore
import yaml  # type: ignore
import pytest

from tests import DATA_DIR

from nrtk_cdao.utils.nrtk_pybsm_perturber import nrtk_pybsm_perturber


dataset_image_folder = os.path.join(DATA_DIR, 'images', 'VisDrone2019-DET-test-dev-TINY')
config_file = os.path.join(DATA_DIR, 'pybsm_config.yaml')
perturb_params_file = os.path.join(DATA_DIR, 'pybsm_perturb_params.yaml')


class TestNRTKPybsmPerturber:
    """
    These tests make use of the `tmpdir` fixture from `pytest`. Find more
    information here: https://docs.pytest.org/en/6.2.x/tmpdir.html
    """

    def test_nrtk_pybsm_perturber_invalid_config(self, tmpdir: py.path.local) -> None:
        """
        Test if an invalid pybsm config raises ValueError through the CliRunner
        """
        output_dir = tmpdir.join('out')
        random_config = {'gsd': 0.5, 'sensor': {'name': 'pybsm_sensor'}}

        with open(perturb_params_file, 'r') as cfg_file:
            perturb_params = yaml.safe_load(cfg_file)

        with pytest.raises(ValueError, match=r"Invalid Configuration"):
            nrtk_pybsm_perturber(
                str(dataset_image_folder),
                str(output_dir),
                random_config,
                perturb_params,
                True
            )

        assert not output_dir.check(dir=1)

    def test_nrtk_pybsm_perturber(self, tmpdir: py.path.local) -> None:

        output_dir = tmpdir.join('out')

        with open(config_file, 'r') as cfg_file:
            config = yaml.safe_load(cfg_file)

        with open(perturb_params_file, 'r') as cfg_file:
            perturb_params = yaml.safe_load(cfg_file)

        nrtk_pybsm_perturber(
                str(dataset_image_folder),
                str(output_dir),
                config,
                perturb_params,
                True
            )
        # expected created directories for the perturber sweep combinations
        img_dirs = [output_dir.join(d) for d in ["_f-0.012_D-0.001_px-2e-05",
                                                 "_f-0.012_D-0.003_px-2e-05",
                                                 "_f-0.014_D-0.001_px-2e-05",
                                                 "_f-0.014_D-0.003_px-2e-05"]]
        # image ids that belong to each perturber sweep combination
        img_ids = ['0000006_02616_d_0000007.jpg', '0000006_03636_d_0000009.jpg',
                   '0000006_00159_d_0000001.jpg', '0000006_01659_d_0000004.jpg',
                   '0000161_01584_d_0000158.jpg', '0000006_01111_d_0000003.jpg',
                   '0000006_04050_d_0000010.jpg', '0000006_04309_d_0000011.jpg',
                   '0000006_01275_d_0000004.jpg', '0000006_00611_d_0000002.jpg',
                   '0000006_02138_d_0000006.jpg']

        assert sorted(output_dir.listdir()) == sorted(img_dirs)
        for img_dir in img_dirs:
            assert len(img_dir.listdir()) > 0
            img_filenames = [img.basename for img in img_dir.listdir()]
            assert sorted(img_filenames) == sorted(img_ids)
