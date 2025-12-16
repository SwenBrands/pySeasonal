from datetime import date
from pathlib import Path

from pyseasonal.products.seas2ipe import swen_seas2ipe
from pyseasonal.utils.config import load_config_argo


def main_ipe(
    config_file: str | Path,
    year: int = date.today().year,
    month: int = date.today().month,
):
    config = load_config_argo(config_file)

    swen_seas2ipe(config, str(year), f"{month:02d}",)


if __name__ == "__main__":
    import fire

    fire.Fire(main_ipe)
