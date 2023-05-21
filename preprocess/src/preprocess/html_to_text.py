from bs4 import BeautifulSoup


def html_to_text(html: str) -> str:
    parser = BeautifulSoup(html, features="html.parser")
    return parser.get_text()
