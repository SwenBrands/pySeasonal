from pyseasonal.cli_tercile import main_pred2tercile

if __name__ == "__main__":
    for domain in ["medcof", "Iberia", "Canarias"]:
        main_pred2tercile(f"config/config_for_seas2ipe_{domain}.yaml", 2024, 6)
