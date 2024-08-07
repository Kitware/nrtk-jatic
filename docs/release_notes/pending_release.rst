Pending Release Notes
=====================

Updates / New Features
----------------------

Features

* Added a test to `test_nrtk_perturber_cli` that uses a generic factory config file.

Dependencies1

* Update `maite` to ^0.6.0.

* Update `nrtk` to >=0.10.0.

CI/CD

* Optimized to not run anything but `publish` when `tag`.

* Created a shared `python-version` job for `python` version matrices.

* Updated scanning to properly report the vulnerabilities.

* Updated scanning to properly scan used packages

* Added caching of packages to pipeline.

* Changed check release notes to only fetch last commit from main.

* Added examples to `black` scan.

* Added `jupyter` notebook extra to `black`.

* Renamed `linting` job to `flake8`.

* Renamed `typing` job to `mypy`.

* Modified all code to be compliant with all `ruff` and `black` checks besides missing docstrings.

* Added additional entrypoint testing.

* Swapped out pipeline to use a shared pipeline.

Other

* Refactored package into 'src/nrtk_jatic instead of 'nrtk_jatic'

Fixes
-----

* Fixed an issue where if `opencv-python` was missing then the pipeline would fail.