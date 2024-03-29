from fastapi import FastAPI, HTTPException
from typing import Dict, Any

app = FastAPI()


# Define a route for handling POST requests
@app.post('/')
def handle_post(data: Dict[str, Any]) -> Dict[str, Any]:
    # Validate input data if needed
    if not data:
        raise HTTPException(status_code=400, detail="No data provided")

    # Process the received data (stub implementation)
    processed_data = {"received_data": data}

    # Prepare a stub response
    response_data = {
        "message": "Data received successfully",
        "processed_data": processed_data  # Echo back the received data
    }

    return response_data
