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
        Forecast month (default: current month, 1 to 12)

    Calling example (note that month format is 1 to 12, i.e. not 01):
    run pyseasonal/cli_tercile.py config/config_for_pred2tercile_operational_Iberia.yaml 2026 2

    """
    config = load_config(config_file)

    swen_pred2tercile_operational(config, str(year), f"{month:02d}")


def main():
    import fire

    fire.Fire(main_pred2tercile)


if __name__ == "__main__":
    main()
