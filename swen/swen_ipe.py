from pyseasonal.cli_product_ipe import main_ipe


if __name__ == "__main__":
    for domain in ["medcof", "Iberia", "Canarias"]:
        main_ipe(f"config/config_for_seas2ipe_{domain}.yaml", 2024, 6)
