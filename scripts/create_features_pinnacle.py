import os
import argparse
from typing import List

import pandas as pd

from data.pinnacle.pull_data import load_ticker_prices
from data.pinnacle.constants import (
    PINNACLE_DATA_FOLDER,
    PINNACLE_CPD_FOLDER,
    PINNACLE_ASSETS
)

# Define the features path here since it's specific to the feature creation process
PINNACLE_FEATURES_PATH = os.path.join("dataset", "pinnacle", "CPD", "features.csv")

from features.creation import (
    create_momentum_features,
    merge_with_changepoint_features
)


def main(
    tickers: List[str],
    cpd_module_folder: str,
    lookback_window_length: int,
    output_file_path: str,
    extra_lbw: List[int],
):
    features = pd.concat(
        [
            create_momentum_features(load_ticker_prices(ticker)).assign(
                ticker=ticker
            )
            for ticker in tickers
        ]
    )

    features.date = features.index
    features.index.name = "Date"

    if lookback_window_length:
        features_w_cpd = merge_with_changepoint_features(
            features, cpd_module_folder, lookback_window_length
        )

        if extra_lbw:
            for extra in extra_lbw:
                extra_data = pd.read_csv(
                    output_file_path.replace(
                        f"pinnacle_cpd_{lookback_window_length}lbw.csv",
                        f"pinnacle_cpd_{extra}lbw.csv",
                    ),
                    index_col=0,
                    parse_dates=True,
                ).reset_index()[
                    ["date", "ticker", f"cp_rl_{extra}", f"cp_score_{extra}"]
                ]
                extra_data["date"] = pd.to_datetime(extra_data["date"])

                features_w_cpd = pd.merge(
                    features_w_cpd.set_index(["date", "ticker"]),
                    extra_data.set_index(["date", "ticker"]),
                    left_index=True,
                    right_index=True,
                ).reset_index()
                features_w_cpd.index = features_w_cpd["date"]
                features_w_cpd.index.name = "Date"
        else:
            features_w_cpd.index.name = "Date"
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        features_w_cpd.to_csv(output_file_path)
    else:
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        features.to_csv(output_file_path)


if __name__ == "__main__":

    def get_args():
        """Returns settings from command line."""

        parser = argparse.ArgumentParser(description="Run changepoint detection module")
        # TODO add ticker options
        parser.add_argument(
            "cpd_module_folder",
            metavar="c",
            type=str,
            nargs="?",
            default=PINNACLE_CPD_FOLDER,
            # choices=[],
            help="Input folder for CPD outputs.",
        )
        parser.add_argument(
            "lookback_window_length",
            metavar="l",
            type=int,
            nargs="?",
            default=None,
            # choices=[],
            help="Input folder for CPD outputs.",
        )
        parser.add_argument(
            "output_file_path",
            metavar="f",
            type=str,
            nargs="?",
            default=PINNACLE_FEATURES_PATH,
            # choices=[],
            help="Output file location for csv.",
        )
        parser.add_argument(
            "extra_lbw",
            metavar="-e",
            type=int,
            nargs="*",
            default=[],
            # choices=[],
            help="Fill missing prices.",
        )

        args = parser.parse_known_args()[0]

        return (
            PINNACLE_ASSETS,
            cpd_pinnacle_output_folder(args.lookback_window_length),
            args.lookback_window_length,
            features_pinnacle_file_path(args.lookback_window_length),
            args.extra_lbw,
        )

    main(*get_args())
