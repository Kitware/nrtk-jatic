from typing import List, ContextManager
from contextlib import nullcontext as does_not_raise
from click.testing import CliRunner
import pytest
import os
import py  # type: ignore
import yaml  # type: ignore

from tests import DATA_DIR

from nrtk_cdao.utils.bin.nrtk_perturber_cli import nrtk_perturber_cli

dataset_folder = os.path.join(DATA_DIR, 'VisDrone2019-DET-test-dev-TINY')
config = yaml.safe_load(os.path.join(DATA_DIR, 'pybsm_config.yaml'))
perturb_params = yaml.safe_load(os.path.join(DATA_DIR, 'pybsm_perturb_params.yaml'))


class TestNRTKPerturber:
    """
    These tests make use of the `tmpdir` fixture from `pytest`. Find more
    information here: https://docs.pytest.org/en/6.2.x/tmpdir.html
    """

    def test_nrtk_perturber(self, tmpdir: py.path.local) -> None:
        """
        Test if the NRTK Perturber CLI works end-to-end and returns the
        appropriate exit code
        """
        output_dir = tmpdir.join('out')

        runner = CliRunner()
        result = runner.invoke(nrtk_perturber_cli,
                               [str(dataset_folder),
                                str(output_dir),
                                config,
                                perturb_params, "-v"])

        assert result.exit_code == 0

    @pytest.mark.parametrize("output_subfolders, expectation", [
        (["_f-0.012_D-0.001_px-2e-05",
          "_f-0.012_D-0.003_px-2e-05",
          "_f-0.014_D-0.001_px-2e-05",
          "_f-0.014_D-0.003_px-2e-05"], does_not_raise())
    ])
    def test_perturber_output_files_exist(self, tmpdir: py.path.local,
                                          output_subfolders: List[str],
                                          expectation: ContextManager) -> None:
        """
        Test if the required output folders and files exist when the CLI
        runs successfully
        """
        output_dir = tmpdir.join('out')

        runner = CliRunner()
        result = runner.invoke(nrtk_perturber_cli,
                               [str(dataset_folder),
                                str(output_dir),
                                config,
                                perturb_params, "-v"])

        assert result.exit_code == 0

        with expectation:
            assert output_dir.check(dir=1)
            for img_dir in output_subfolders:
                assert output_dir.join(img_dir).check(dir=1)
                # image metadata json file
                img_metadata = output_dir.join(img_dir).join("image_metadata.json")
                # resized detections after augmentations
                augmented_detections = output_dir.join(img_dir).join("augmented_detections.json")
                assert img_metadata.check(exists=1)
                assert augmented_detections.check(exists=1)

    def test_nrtk_perturber_invalid_config(self, tmpdir: py.path.local) -> None:
        """
        Test if an invalid pybsm config raises ValueError through the CliRunner
        """
        data = dict(gsd='0.1', sensor=dict(name='test_sensor'))
        output_dir = tmpdir.join('out')
        random_config_file = tmpdir.join('random_config.yml')

        with open(random_config_file, 'w') as outfile:
            yaml.dump(data, outfile, default_flow_style=False)

        random_config = yaml.safe_load(os.path.join(random_config_file))

        runner = CliRunner()
        result = runner.invoke(nrtk_perturber_cli,
                               [str(dataset_folder),
                                str(output_dir),
                                random_config,
                                perturb_params, "-v"])

        result.exception == "Invalid Configuration"
        assert not output_dir.check(dir=1)
