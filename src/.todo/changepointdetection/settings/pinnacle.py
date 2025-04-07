import os

from . import CPD_LBW_DEFAULT


cpd_pinnacle_output_folder = lambda lbw: os.path.join(
    "features", f"pinnacle_cpd_{(lbw if lbw else 'none')}lbw"
)

features_pinnacle_file_path = lambda lbw: os.path.join(
    "features", f"pinnacle_cpd_{(lbw if lbw else 'none')}lbw.csv"
)

CPD_PINNACLE_OUTPUT_FOLDER_DEFAULT: str = cpd_pinnacle_output_folder(CPD_LBW_DEFAULT)
FEATURES_PINNACLE_FILE_PATH_DEFAULT: str = features_pinnacle_file_path(CPD_LBW_DEFAULT)