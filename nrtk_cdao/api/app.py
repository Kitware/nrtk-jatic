import numpy as np
from fastapi import FastAPI, HTTPException
from typing import Dict, Any

from nrtk_cdao.api.converters import build_pybsm_factory, load_COCOJAITIC_dataset
from nrtk_cdao.api.schema import NrtkPybsmPerturbInputSchema


app = FastAPI()


# Define a route for handling POST requests
@app.post("/")
def handle_post(data: NrtkPybsmPerturbInputSchema) -> Dict[str, Any]:
    """
    Returns a collection of augmented datasets based parameters in data

    :param data: NrtkPybsmPerturbInputSchema from schema.py
    """

    # Validate input data if needed
    if not data:
        raise HTTPException(status_code=400, detail="No data provided")

    # Build pybsm factory
    perturber_factory = build_pybsm_factory(data)

    # Load dataset
    input_dataset = load_COCOJAITIC_dataset(data)

    # Call nrtk_perturber
    # augmented_datasets = nrtk_perturber(
    #     maite_dataset=input_dataset, perturber_factory=perturber_factory
    # )

    factory_config = perturber_factory.get_config()

    # Convert np.ndarry to List for easier serialization
    for key in factory_config["sensor"]:
        if isinstance(factory_config["sensor"][key], np.ndarray):
            factory_config["sensor"][key] = factory_config["sensor"][key].tolist()

    processed_data = {
        "dataset_len": len(input_dataset),
        "factory_config": factory_config,
    }

    # Prepare a response
    response_data = {
        "message": "Data received successfully",
        "processed_data": processed_data,  # Echo back the received data
    }

    return response_data
