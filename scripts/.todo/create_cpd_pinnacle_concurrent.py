import multiprocessing
import argparse
import os

from mlmomentum.settings.default import (
    PINNACLE_ASSETS,
    CPD_PINNACLE_OUTPUT_FOLDER,
    CPD_DEFAULT_LBW,
)

N_WORKERS = len(PINNACLE_ASSETS)


def main(lookback_window_length: int):
    os.makedirs(CPD_PINNACLE_OUTPUT_FOLDER(lookback_window_length), exist_ok=True)

    all_processes = [
        f'python -m examples.cpd_pinnacle "{ticker}" "{os.path.join(CPD_PINNACLE_OUTPUT_FOLDER(lookback_window_length), ticker + ".csv")}" "1990-01-01" "2021-12-31" "{lookback_window_length}"'
        for ticker in PINNACLE_ASSETS
    ]
    process_pool = multiprocessing.Pool(processes=N_WORKERS)
    process_pool.map(os.system, all_processes)


if __name__ == "__main__":

    def get_args():
        """Returns settings from command line."""

        parser = argparse.ArgumentParser(
            description="Run changepoint detection module for all tickers"
        )
        parser.add_argument(
            "lookback_window_length",
            metavar="l",
            type=int,
            nargs="?",
            default=CPD_DEFAULT_LBW,
            help="CPD lookback window length",
        )
        return [
            parser.parse_known_args()[0].lookback_window_length,
        ]

    main(*get_args())
