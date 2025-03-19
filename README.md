<!-- :auto badges: -->
[![PyPI - Python Version](https://img.shields.io/pypi/v/nrtk-jatic)](https://pypi.org/project/nrtk-jatic/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/nrtk-jatic)
[![Documentation Status](https://readthedocs.org/projects/nrtk-jatic/badge/?version=latest)](https://nrtk-jatic.readthedocs.io/en/latest/?badge=latest)
<!-- :auto badges: -->

# DEPRECATED

The `nrtk-jatic` package is deprecated and will fail to import on 2025/05/01.
All functionality has been integrated into the core [`nrtk` package](https://gitlab.jatic.net/jatic/kitware/nrtk)
(v0.20.0) under the `nrtk.interop.maite` module. Please migrate to the `nrtk` package as soon as possible.

# nrtk-jatic

The `nrtk-jatic` package is an extension of the Natural Robustness Toolkit
([NRTK](https://github.com/Kitware/nrtk)) containing implementations
and examples in compliance with protocols from the Modular AI Trustworthy Engineering
([MAITE](https://github.com/mit-ll-ai-technology/maite)) library.
These packages (among others) are developed under the
[Joint AI Test Infrastructure Capability (JATIC) program](https://cdao.pages.jatic.net/public/)
for AI Test & Evaluation (T&E) and AI Assurance.

## Interoperability - Implementations and Examples

The `nrtk-jatic` package consists of implementations and utilities that ensure
interoperability of `nrtk` functionality with `maite`. The scripts under
`src/nrtk_jatic/interop` consist of protocol implementations that are compliant
with `maite`'s dataset and augmentation protocols. The `src/nrtk_jatic/utils`
folder houses generic util scripts and the NRTK CLI entrypoint script.
Finally, the `examples` folder consists of Jupyter notebooks showing
end-to-end ML T&E workflows demonstrating natural robustness testing of computer vision models with `nrtk`,
and integrations of `nrtk` with other JATIC tools,
by using the interoperability standards provided by `maite`

Additional information about JATIC and its design principles can be found
[here](https://cdao.pages.jatic.net/public/program/design-principles/).

<!-- :auto installation: -->
## Installation
Ensure the source tree is acquired locally before proceeding.

To install the current version via `pip`:
```bash
pip install nrtk-jatic[<extra1>,<extra2>,...]
```

Alternatively, you can use [Poetry](https://python-poetry.org/):
```bash
poetry install --with main,linting,tests,docs --extras "<extra1> <extra2> ..."
```

Certain plugins may require additional runtime dependencies. Details on these requirements can be found [here](https://nrtk-jatic.readthedocs.io/en/latest/implementations.html).

For more detailed installation instructions, visit the [installation documentation](https://nrtk-jatic.readthedocs.io/en/latest/installation.html).
<!-- :auto installation: -->

<!-- :auto getting-started: -->
## Getting Started
Explore usage examples of the `nrtk-jatic` package in various contexts using the Jupyter notebooks provided in the `./examples/` directory.

Contributions are encouraged! For more details, refer to the [CONTRIBUTING.md](./CONTRIBUTING.md) file.
<!-- :auto getting-started: -->

<!-- :auto documentation: -->
## Documentation
Documentation for both release snapshots and the latest master branch is available on [ReadTheDocs](https://nrtk-jatic.readthedocs.io/en/latest/).

To build the Sphinx-based documentation locally for the latest reference:
```bash
# Install dependencies
poetry install --sync --with main,linting,tests,docs
# Navigate to the documentation root
cd docs
# Build the documentation
poetry run make html
# Open the generated documentation in your browser
firefox _build/html/index.html
```
<!-- :auto documentation: -->

<!-- :auto developer-tools: -->
## Developer Tools

### Pre-commit Hooks
Pre-commit hooks ensure that code complies with required linting and formatting guidelines. These hooks run automatically before commits but can also be executed manually. To bypass checks during a commit, use the `--no-verify` flag.

To install and use pre-commit hooks:
```bash
# Install required dependencies
poetry install --sync --with main,linting,tests,docs
# Initialize pre-commit hooks for the repository
poetry run pre-commit install
# Run pre-commit checks on all files
poetry run pre-commit run --all-files
```
<!-- :auto developer-tools: -->

<!-- :auto contributing: -->
## Contributing
- Follow the [JATIC Design Principles](https://cdao.pages.jatic.net/public/program/design-principles/).
- Adopt the Git Flow branching strategy.
- Detailed release information is available in [docs/release_process.rst](./docs/release_process.rst).
- Additional contribution guidelines and issue reporting steps can be found in [CONTRIBUTING.md](./CONTRIBUTING.md).
<!-- :auto contributing: -->

<!-- :auto license: -->
## License
[Apache 2.0](./LICENSE)
<!-- :auto license: -->

<!-- :auto contacts: -->
## Contacts

**Principal Investigator**: Brian Hu (Kitware) @brian.hu

**Product Owner**: Austin Whitesell (MITRE) @awhitesell

**Scrum Master / Tech Lead**: Brandon RichardWebster (Kitware) @b.richardwebster

**Deputy Tech Lead**: Emily Veenhuis (Kitware) @emily.veenhuis
<!-- :auto contacts: -->
