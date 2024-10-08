import pytest
from maite.testing.pyright import list_error_messages, pyright_analyze


@pytest.mark.filterwarnings("ignore:Jupyter is migrating its paths")
@pytest.mark.parametrize(
    ("filepath", "expected_num_errors"),
    [
        ("examples/augmentations.ipynb", 0),
        ("examples/jatic-perturbations-saliency.ipynb", 0),
        # ("examples/daml/daml_example_notebook.ipynb", 0), Broken notebook
        ("examples/gradio/nrtk-gradio.ipynb", 0),
    ],
)
def test_pyright_nb(filepath: str, expected_num_errors: int) -> None:
    results = pyright_analyze(filepath)[0]
    assert results["summary"]["errorCount"] <= expected_num_errors, list_error_messages(results)
