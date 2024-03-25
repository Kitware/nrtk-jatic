from click.testing import CliRunner
import unittest.mock as mock
import pytest
import os
import py  # type: ignore
import yaml  # type: ignore

from tests import DATA_DIR

from nrtk_cdao.utils.bin.nrtk_perturber_cli import nrtk_perturber_cli

from importlib.util import find_spec

deps = ['kwcoco']
specs = [find_spec(dep) for dep in deps]
is_usable = all([spec is not None for spec in specs])

dataset_folder = os.path.join(DATA_DIR, 'VisDrone2019-DET-test-dev-TINY')
config = yaml.safe_load(os.path.join(DATA_DIR, 'pybsm_config.yaml'))
perturb_params = yaml.safe_load(os.path.join(DATA_DIR, 'pybsm_perturb_params.yaml'))


class TestNRTKPerturberNotUsable:
    """
    These tests make use of the `tmpdir` fixture from `pytest`. Find more
    information here: https://docs.pytest.org/en/6.2.x/tmpdir.html
    """

    @mock.patch("nrtk_cdao.utils.bin.nrtk_perturber_cli.kwcoco_is_usable", False)
    def test_warning(self, tmpdir: py.path.local) -> None:
        """
        Test that proper warning is displayed when required dependencies are
        not installed.
        """
        output_dir = tmpdir.join('out')

        runner = CliRunner()

        result = runner.invoke(nrtk_perturber_cli,
                               [str(dataset_folder),
                                str(output_dir),
                                config,
                                perturb_params, "-v"])

        result.exception == "This tool requires additional dependencies, please install 'nrtk-cdao[tools]'"
        assert not output_dir.check(dir=1)


@pytest.mark.skipif(not is_usable, reason="Extra 'nrtk-cdao[tools]' not installed.")
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
        assert output_dir.check(dir=1)

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
