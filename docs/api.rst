Dependencies
============

In addition to the dependencies in Poetry, the user will need to install the uvicorn package.

Usage
=====

To run the app, open a command prompt and navigate to `nrtk-cdao/api/`, then run the command::

    uvicorn app:app --reload

This command starts the server with the API accessible at `https://127.0.0.1:8000` by default.

To invoke the service with `curl`, use the following command::

    curl -X POST http://127.0.0.1:8000/ -H "Content-Type: application/json" -d '{"key": "value"}'

This command sends a POST request with JSON data `{"key": "value"}` to the REST server. Replace `http://127.0.0.1:8000` with the appropriate URL if you have specified a different host or port. If successful, you should receive a response containing a message indicating the success of the operation and the JSON stub.
