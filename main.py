import argparse
import pandas as pd
from config import cfg, set_config
from src.segmentation_step import run_segmentation



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_default", help="Config to use", required=False)
    args = parser.parse_args()
    set_config(args.config)

    segments_file = run_segmentation()



