from preprocess.html_to_text import html_to_text


def test_html_to_text():
    assert html_to_text("a") == "a"
