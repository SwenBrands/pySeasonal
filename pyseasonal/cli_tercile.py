from datetime import date
from pathlib import Path

from pyseasonal.pred2tercile_operational import swen_pred2tercile_operational
from pyseasonal.utils.config import load_config


def main_pred2tercile(
    config_file: str | Path,
    year: int = date.today().year,
    month: int = date.today().month,
) -> None:
    """
    CLI entry point for operational tercile prediction processing.

    Loads configuration and calls swen_pred2tercile_operational to generate
    tercile probability forecasts for the specified year and month.

    Parameters:
    -----------
    config_file : str or Path
        Path to YAML configuration file
    year : int, optional
        Forecast year (default: current year)
    month : int, optional
        Forecast month (default: current month)
    """
    config = load_config(config_file)

    swen_pred2tercile_operational(config, str(year), f"{month:02d}")


if __name__ == "__main__":
    import fire

    fire.Fire(main_pred2tercile)
