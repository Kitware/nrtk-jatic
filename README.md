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
The following steps assume the source tree has been acquired locally.

Install the current version via pip:
```bash
pip install nrtk-jatic
```

Alternatively, you can also use [Poetry](https://python-poetry.org/):
```bash
poetry install --sync --with dev-linting,dev-testing,dev-docs
```

See [here for more installation documentation](
https://nrtk-jatic.readthedocs.io/en/latest/installation.html).
<!-- :auto installation: -->

<!-- :auto getting-started: -->
## Getting Started
We provide a number of examples based on Jupyter notebooks in the
`./examples/` directory to show usage of the `nrtk-jatic` package in a number
of different contexts.
<!-- :auto getting-started: -->

<!-- :auto documentation: -->
## Documentation
Documentation snapshots for releases as well as the latest master are hosted
on [ReadTheDocs](https://nrtk-jatic.readthedocs.io/en/latest/).

The sphinx-based documentation may also be built locally for the most
up-to-date reference:
```bash
# Install dependencies
poetry install --sync --with dev-linting,dev-testing,dev-docs
# Navigate to the documentation root.
cd docs
# Build the docs.
poetry run make html
# Open in your favorite browser!
firefox _build/html/index.html
```
<!-- :auto documentation: -->

<!-- :auto developer-tools: -->
## Developer tools

**pre-commit hooks**
pre-commit hooks are used to ensure that any code meets all linting and
formatting guidelines required. After installing, this will always run before
 committing to ensure that any commits are following the standards, but you
 can also manually run the check without committing. If you want to commit
 despite there being errors, you can add `--no-verify` to your commit command.

Installing pre-commit hooks:
```bash
# Ensure that all dependencies are installed
poetry install --sync --with dev-linting,dev-testing,dev-docs
# Initialize pre-commit for the repository
poetry run pre-commit install
# Run pre-commit check on all files
poetry run pre-commit run --all-files
```
<!-- :auto developer-tools: -->

<!-- :auto contributing: -->
## Contributing
- We follow the general guidelines outlined in the
[JATIC Software Development Plan](https://gitlab.jatic.net/jatic/docs/sdp/-/blob/main/Branch,%20Merge,%20Release%20Strategy.md).
- We use the Git Flow branching strategy.
- See [docs/release_process.rst](./docs/release_process.rst) for detailed release information.
- See [CONTRIBUTING.md](./CONTRIBUTING.md) for additional contributing information.
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
