from simuscan import syntheticAFM_COCO


def main():

    syntheticAFM_COCO.generate_data_for_training(r"data\config\estrella_config.yaml")

if __name__ == "__main__":
    main()