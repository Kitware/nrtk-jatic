import pytest

from jatic_toolbox.testing.pyright import list_error_messages, pyright_analyze


@pytest.mark.filterwarnings("ignore:Jupyter is migrating its paths")
@pytest.mark.parametrize("filepath, expected_num_errors", [
    ("examples/jatic_toolbox/augmentations.ipynb", 0),
    ("examples/daml/daml_example_notebook.ipynb", 0)
])
def test_pyright_nb(filepath: str, expected_num_errors: int) -> None:
    results = pyright_analyze(filepath)[0]
    assert results["summary"]["errorCount"] <= expected_num_errors, list_error_messages(
        results
    )
