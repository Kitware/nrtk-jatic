# nrtk-jatic

## Description
The `nrtk-jatic` package is an extension of the Natural Robustness Toolkit 
(NRTK) for JATIC. For more information, checkout the base [NRTK](https://gitlab.jatic.net/jatic/kitware/nrtk) 
package.

## Installation
The following steps assumes the source tree has been acquired locally.

Install the current version via pip:
```bash
pip install .
```

Alternatively, [Poetry](https://python-poetry.org/) can also be used:
```bash
poetry install --sync --with dev-linting,dev-testing,dev-docs
```

## Getting Started
We provide a number of examples based on Jupyter notebooks in the `./examples/` directory to show usage
of the `nrtk-jatic` package in a number of different contexts. For general examples, checkout NRTK\'s 
[examples](https://gitlab.jatic.net/jatic/kitware/nrtk/-/tree/main/examples?ref_type=heads) directory.

## Documentation
Documentation snapshots for releases as well as the latest master are hosted on
ReadTheDocs.

The sphinx-based documentation may also be built locally for the most
up-to-date reference:
```bash
# Install dependencies
poetry install
# Navigate to the documentation root.
cd docs
# Build the docs.
poetry run make html
# Open in your favorite browser!
firefox _build/html/index.html
```

## Contributing

- We follow the general guidelines outlined in the
[JATIC Software Development Plan](https://gitlab.jatic.net/jatic/docs/sdp/-/blob/main/Branch,%20Merge,%20Release%20Strategy.md).
- The Git Flow branching strategy is used.
- See `docs/releasing/release_process.rst` for detailed release information.
- See `CONTRIBUTING.md` for additional contributing information.

## License
Apache 2.0

**POC**: Brian Hu @brian.hu
**DPOC**: Brandon RichardWebster @b.richardwebster
