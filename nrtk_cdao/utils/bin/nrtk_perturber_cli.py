import click  # type: ignore
from typing import TextIO

import yaml  # type: ignore

from nrtk_cdao.utils.nrtk_pybsm_perturber import nrtk_pybsm_perturber


@click.command(context_settings={"help_option_names": ['-h', '--help']})
@click.argument('dataset_img_dir', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path(exists=False))
@click.argument('config_file', type=click.File(mode='r'))
@click.argument('perturb_params_file', type=click.File(mode='r'))
@click.option('--verbose', '-v', count=True, help='print progress messages')
def nrtk_perturber_cli(
    dataset_img_dir: str,
    output_dir: str,
    config_file: TextIO,
    perturb_params_file: TextIO,
    verbose: bool
) -> None:
    """
    Generate NRTK perturbed images from a given set of source images and write
    them to an output folder in disk. The perturbed images are stored in subfolders
    named after the chosen perturbation parameter keys and values.

    \b
    DATASET_IMG_DIR - Directory where the source images are located.
    OUTPUT_DIR - Directory to write the perturbed images to.
    CONFIG_FILE - Configuration file pertaining to a specific type of
        NRTK perturber.
    PERTURB_PARAMS_FILE - Configuration file containing the parameter value
        combinations to be implemented with the perturber factory.

    \f
    :param dataset_img_dir: Directory where the source images are located.
    :param output_dir: Directory to write the perturbed images to.
    :param config_file: Configuration file pertaining to a specific type of
        NRTK perturber.
    :param perturb_params_file: Configuration file containing the values needed
        for the parameter sweep implemented with the perturber factory.
    :param verbose: Display progress messages. Default is false.
    """

    # load config
    config = yaml.safe_load(config_file)

    # Load pybsm perturb params
    perturb_factory_config = yaml.safe_load(perturb_params_file)

    nrtk_pybsm_perturber(
        dataset_img_dir,
        output_dir,
        config,
        perturb_factory_config,
        verbose
    )
