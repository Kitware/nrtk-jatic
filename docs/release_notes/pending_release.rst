Pending Release Notes
=====================

Updates / New Features
----------------------

Features

* Added a test to ``test_nrtk_perturber_cli`` that uses a generic factory config file.

Interoperability

* Added ``JATICClassificationAugmentationWithMetric`` workflow which performs
  image-metric computations in a JATIC Augmentation workflow for MAITE-compliant
  image classification datasets.

* Added ``JATICDetectionAugmentationWithMetric`` workflow which performs image-metric
  computations in a JATIC Augmentation workflow for MAITE-compliant object detection
  datasets.

* Added ``compute_image_metric.ipynb`` notebook showcasing how image-metric computations
  can be of use to a JATIC T&E engineer in their workflow.

Dependencies

* Update ``maite`` to ^0.6.0.

* Update ``nrtk`` to >=0.11.0.

CI/CD

* Optimized to not run anything but ``publish`` when ``tag``.

* Created a shared ``python-version`` job for ``python`` version matrices.

* Updated scanning to properly report the vulnerabilities.

* Updated scanning to properly scan used packages

* Added caching of packages to pipeline.

* Changed check release notes to only fetch last commit from main.

* Added examples to ``black`` scan.

* Added ``jupyter`` notebook extra to ``black``.

* Renamed ``linting`` job to ``flake8``.

* Renamed ``typing`` job to ``mypy``.

* Modified all code to be compliant with all ``ruff`` and ``black`` checks besides missing docstrings.

* Added additional entrypoint testing.

* Swapped out pipeline to use a shared pipeline.

* Added a mirroring job to replace builtin gitlab mirroring due to LFS issue.

* Numerous changes to help automated the CI/CD process.

* ``poetry.lock`` file updated for the dev environment.

* Updates to dependencies to support the new CI/CD.

* Updated config for ``black`` to set max line length to 120

Other

* Refactored package into 'src/nrtk_jatic instead of 'nrtk_jatic'

Documentation

* Added ReadTheDocs configuration files

* Added a ``Containers`` section to documentation

* Added ``AUKUS.rst`` to Containers documentations

* Updated JATICDetectionAugmentation docstrings to clarify ground truth behavior

* Added sphinx's ``autosummary`` template for recursively populating
  docstrings from the module level down to the class method level.

* Added support for ``sphinx-click`` to generate documentation for python
  ``click`` functions.

* Updated README to include a section on ``Interoperability``.

* Restored and improved review process documentation.

* Fixed sphinx linting errors.

Fixes
-----

* Fixed an issue where if ``opencv-python`` was missing then the pipeline would fail.
