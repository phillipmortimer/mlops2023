from pathlib import Path

import pytest

from preprocess.html_to_text import html_to_text


@pytest.fixture
def test_file():
    test_path = Path(__file__).parent / "test_data/18563939_11.html"
    with test_path.open() as fp:
        text = fp.read()
    return text


def test_html_to_text(test_file):
    target = "\n\n\nCash flows from operating activities\n\nNet income\n$839,534\n"

    assert html_to_text(test_file)[:len(target)] == target
