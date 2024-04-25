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

* Added conversion from JSON schema to inputs for ``nrtk_pertuber`` entrypoint.

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

Utils

* Added a CLI script to perform PyBSM perturbations on a sample image set.

Fixes
-----
