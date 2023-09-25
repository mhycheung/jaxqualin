from postprocess import *
from pathlib import Path
import argparse
import configparser
import os

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(ROOT_PATH, "config")

parser = argparse.ArgumentParser()

parser.add_argument('setting_name')
parser.add_argument('batch_runname')

args = parser.parse_args()

setting_name = args.setting_name
batch_runname = args.batch_runname

eff_num_len = 10
kwargs_in = {}

config_file_path = os.path.join(CONFIG_PATH, f"{setting_name}.ini")
config = configparser.ConfigParser()
config.optionxform = str

config.read(config_file_path)
config_sections = config._sections
kwargs = config_sections['basic']
flatness_checker_kwargs = config_sections['flatness_checker']
mode_searcher_kwargs = config_sections['mode_searcher']
for key in kwargs:
    kwargs[key] = eval(kwargs[key])
for key in flatness_checker_kwargs:
    flatness_checker_kwargs[key] = eval(flatness_checker_kwargs[key])
for key in mode_searcher_kwargs:
    mode_searcher_kwargs[key] = eval(mode_searcher_kwargs[key])


kwargs.update(kwargs_in, flatness_checker_kwargs = flatness_checker_kwargs,
               mode_searcher_kwargs = mode_searcher_kwargs)

eff_num_list = list(range(eff_num_len))
df_path = os.path.join(ROOT_PATH, f"pickle/data_frame/{batch_runname}.csv")

df = create_data_frame_eff(eff_num_list, batch_runname, df_save_prefix = batch_runname,
                           l = 2, m = 2, **kwargs)

