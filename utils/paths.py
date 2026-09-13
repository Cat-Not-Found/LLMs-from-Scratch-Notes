"""Small path helpers so the notebooks and scripts work from any directory.

The notebooks in `chXX/` are usually executed from the repository root or from
their own chapter folder. These helpers resolve the repository layout defined
in `README.md`:

    LLMs/
    ├── ch01/ ... ch06/     notebooks
    ├── data/               datasets and generated artifacts
    ├── gpt2/               downloaded OpenAI GPT-2 weights
    ├── scripts/            standalone scripts
    └── utils/              shared modules

`data/` is not stored in git (its contents are downloaded or generated), so
both helpers make sure the directory they point into exists. That way calls
such as `torch.save(..., data_path("model.pth"))` or
`open(data_path("the-verdict.txt"), "wb")` work on a fresh clone too.

Only the standard library is used here, so importing this module is cheap.
"""

from pathlib import Path

# utils/paths.py -> utils/ -> repository root
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"


def repo_path(*parts):
    """Return an absolute path inside the repository root.

    The parent directory of the requested path is created if needed, so
    `repo_path("ch05", "review_classifier.pth")` can be written directly.
    """
    path = REPO_ROOT.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def data_path(*parts):
    """Return an absolute path inside `<repo>/data`, creating `data/` if needed."""
    path = DATA_DIR.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
