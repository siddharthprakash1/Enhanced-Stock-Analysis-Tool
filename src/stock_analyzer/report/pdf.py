from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML

_TPL = Path(__file__).parent / "templates"


def render_pdf(context: dict, out_path) -> None:
    env = Environment(loader=FileSystemLoader(str(_TPL)))
    css = (_TPL / "styles.css").read_text()
    html = env.get_template("report.html.j2").render(css=css, **context)
    HTML(string=html, base_url=str(_TPL)).write_pdf(str(out_path))
