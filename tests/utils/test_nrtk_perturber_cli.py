from click.testing import CliRunner
import os
import py  # type: ignore
import yaml  # type: ignore

from tests import DATA_DIR

from nrtk_cdao.utils.bin.nrtk_perturber_cli import nrtk_perturber_cli


dataset_image_folder = os.path.join(DATA_DIR, 'images', 'VisDrone2019-DET-test-dev-TINY')
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
                               [str(dataset_image_folder),
                                str(output_dir),
                                config,
                                perturb_params, "-v"])

        assert result.exit_code == 0
        assert output_dir.check(dir=1)
