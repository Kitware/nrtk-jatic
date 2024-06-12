Pending Release Notes
=====================

This is the initial release of this repository which hosts integration
documentation, examples, and code related to integrating nrtk
components with the CDAO needs and use-cases.


Updates / New Features
----------------------

Documentation

* Added reference to original NRTK documentation.

* Added ability to render documentation on Gitlab Pages.

* Update README.md

* Added Sphinx document rendering for MRs. The docs pages can be accessed by clicking the "View App"
  button located in the merge request page under the test pipeline section.

* Added sphinx auto-documentation on JATIC interoperability for object detection.

* Added sphinx auto-documentation on JATIC interoperability for image classification.

* Added email to pyproject.toml

Examples

* Added an example notebook exploring the current state of several augmentation
  tools as well as the usability of the JATIC augmentation protocol.

* Added an example notebook exploring the DPDivergence metric using the DAML
  tools developed by the ARIA team.

* Added an example notebook exploring using Gradio as an interface for applying
  pyBSM perturbations.

API

* Added in REST API server which accepts a POST request with a JSON data
  payload and returns a JSON payload

* Added in unit-test for POST request with REST API server

* Added JSON schema pydantic file and corresponding documentation for how it
  should be used

* Added additional POST request specifically to accept the AUKUS json schema
  and call our main API with that data

* Added `Dockerfile` and `compose.yaml` to containerize the `nrtk-cdao` package
  and host the base REST API and AUKUS REST API.

* Added config file support for loading pyBSM factories for NRTKaaS.

License

* Add Apache 2.0 license

Interoperability

* Added ``JATICDetectionAugmentation`` which performs perturbations on MAITE-protocol
  based object detection datasets.

* Added ``COCOJATICObjectDetectionDataset``` which loads a COCO detection dataset from
  disk and converts it to a MAITE-protocol compliant detection dataset.

* Added ``JATICObjectDetectionDataset`` which is a custom I/O bridging dataset wrapper
  for MAITE compliant detection datasets.

* Added a utility function to save a MAITE object detection dataset to file as a COCO
  dataset.

* Added ``JATICClassificationAugmentation`` which performs perturbations on MAITE-protocol
  based image classification datasets.

Utils

* Moved kwcoco to poetry tools dependency

* Added a CLI script to perform perturbations on a sample image set.

Python Version Support

* Added support for `py3.12`

CI/CD

* Updated CI test matrix to support `py3.12`

* Added additional entrypoint testing.

Fixes
-----

* Updated to use `nrtk>=0.5.3` which patched an issue with `numpy` dependency resolution.

* Numpy is used by the package but was never added to the list of dependencies and
  was instead installed indirectly from `nrtk`. Added `numpy` as a required
  dependency, which also has the side effect of solving resolution issues.

* Updated `numpy` hinge for `Python 3.12`
