from fastapi import FastAPI, HTTPException
from pathlib import Path

from nrtk_cdao.api.converters import build_factory
from nrtk_cdao.api.schema import NrtkPerturbInputSchema, NrtkPerturbOutputSchema, DatasetSchema
from nrtk_cdao.interop.object_detection.utils import dataset_to_coco
from nrtk_cdao.utils.nrtk_perturber import nrtk_perturber

try:
    from nrtk_cdao.api.converters import load_COCOJATIC_dataset
    is_usable = True
except ImportError:
    is_usable = False


app = FastAPI()


# Define a route for handling POST requests
@app.post("/")
def handle_post(data: NrtkPerturbInputSchema) -> NrtkPerturbOutputSchema:
    """
    Returns a collection of augmented datasets based parameters in data

    :param data: NrtkPybsmPerturbInputSchema from schema.py

    :returns: NrtkPybsmPerturberOutputSchema from schema.py

    :raises: HTTPException upon failure
    """
    try:
        # Build pybsm factory
        perturber_factory = build_factory(data)

        # Load dataset
        if not is_usable:
            raise ImportError("This tool requires additional dependencies, please install `nrtk-cdao[tools]`")
        input_dataset = load_COCOJATIC_dataset(data)

        # Call nrtk_perturber
        augmented_datasets = nrtk_perturber(
            maite_dataset=input_dataset, perturber_factory=perturber_factory
        )

        # Format output
        datasets_out = list()
        img_filenames = [Path("images") / img_path.name for img_path in input_dataset.get_img_path_list()]
        for perturb_params, aug_dataset in augmented_datasets:
            full_output_dir = Path(data.output_dir) / perturb_params
            dataset_to_coco(
                dataset=aug_dataset,
                output_dir=full_output_dir,
                img_filenames=img_filenames,
                dataset_categories=input_dataset.get_categories()
            )
            datasets_out.append(DatasetSchema(
                root_dir=str(full_output_dir),
                label_file="annotations.json",
                metadata_file="image_metadata.json"
            ))

        return NrtkPerturbOutputSchema(
            message="Data received successfully",
            datasets=datasets_out,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
