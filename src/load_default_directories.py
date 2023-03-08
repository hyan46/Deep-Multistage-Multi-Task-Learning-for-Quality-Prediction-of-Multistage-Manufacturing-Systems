import os
from pathlib import Path

# this file sits under name_of_proj_directory/src after someone clones it
# thus we first get CWD which is that folder and it's parent is name_of_proj_directory
# from there we can navigate other directories

CWD = Path(os.path.realpath(__file__))
PROJ_HOME_DIR = CWD.parent.parent
HOME_DIR = Path(os.getenv('HOME'))
DATA_DIR = HOME_DIR.parent / "haoyan" / "data" / "data" / "PG_data" / "Data_v2"
OUTPUT_DIR = PROJ_HOME_DIR / "output"
