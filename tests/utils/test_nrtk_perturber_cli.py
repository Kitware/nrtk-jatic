from typing import List, ContextManager
from contextlib import nullcontext as does_not_raise
from click.testing import CliRunner
import pytest
import os
import py  # type: ignore

from tests import DATA_DIR

from nrtk_cdao.utils.bin.nrtk_perturber_cli import nrtk_perturber_cli

dataset_folder = os.path.join(DATA_DIR, 'VisDrone2019-DET-test-dev-TINY')
config_file = os.path.join(DATA_DIR, 'nrtk_config.json')


class TestNRTKPerturber:
    """
    These tests make use of the `tmpdir` fixture from `pytest`. Find more
    information here: https://docs.pytest.org/en/6.2.x/tmpdir.html
    """

    @pytest.mark.parametrize("output_subfolders, expectation", [
        (["_f-0.012_D-0.001_px-2e-05",
          "_f-0.012_D-0.003_px-2e-05",
          "_f-0.014_D-0.001_px-2e-05",
          "_f-0.014_D-0.003_px-2e-05"], does_not_raise())
    ])
    def test_nrtk_perturber(
        self,
        tmpdir: py.path.local,
        output_subfolders: List[str],
        expectation: ContextManager
    ) -> None:
        """
        Test if the required output folders and files exist when the CLI
        runs successfully
        """
        output_dir = tmpdir.join('out')

        runner = CliRunner()
        result = runner.invoke(
            nrtk_perturber_cli,
            [
                str(dataset_folder),
                str(output_dir),
                str(config_file),
                "-v"
            ]
        )

        assert result.exit_code == 0

        with expectation:
            assert output_dir.check(dir=1)
            for img_dir in output_subfolders:
                assert output_dir.join(img_dir).check(dir=1)
                # image metadata json file
                img_metadata = output_dir.join(img_dir).join("image_metadata.json")
                # resized detections after augmentations
                augmented_detections = output_dir.join(img_dir).join("annotations.json")
                assert img_metadata.check(exists=1)
                assert augmented_detections.check(exists=1)

    def test_config_gen(self, tmpdir: py.path.local) -> None:
        """
        Test the generate configuration file option.
        """
        output_dir = tmpdir.join('out')

        output_config = tmpdir.join('gen_conf.json')

        runner = CliRunner()
        runner.invoke(
            nrtk_perturber_cli,
            [
                str(dataset_folder),
                str(output_dir),
                str(config_file),
                "-g", str(output_config)
            ]
        )

        # check that config file was created
        assert output_config.check(file=1)
        # check that no output was generated
        assert not output_dir.check(dir=1)
