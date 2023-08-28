from postprocess import *
import argparse
import configparser
import os

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(ROOT_PATH, "config")

parser = argparse.ArgumentParser()

parser.add_argument('setting_name')
parser.add_argument('runname')
parser.add_argument('SXS_list_file_name')
parser.add_argument('-s', '--skip', dest = 'SXS_skip', nargs='+')

args = parser.parse_args()

setting_name = args.setting_name
runname = args.runname
SXS_list_file_name = args.SXS_list_file_name
SXS_skip = args.SXS_skip

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

kwargs.update(flatness_checker_kwargs = flatness_checker_kwargs,
               mode_searcher_kwargs = mode_searcher_kwargs)

with open(SXS_list_file_name, "r") as f:
    SXS_nums_raw = f.readlines()

SXS_num_list = [SXS_num.strip() for SXS_num in SXS_nums_raw if SXS_num.strip() not in SXS_skip]

# SXS_num_list_1 = [str(SXS_num) for SXS_num in range(1419,1510)]
# SXS_num_list_2 = ["0" + str(SXS_num) for SXS_num in range(209,306)]
# SXS_num_list = SXS_num_list_1 + SXS_num_list_2
create_data_frame(SXS_num_list, 
                  df_save_prefix = runname, 
                  postfix_string = setting_name,
                  pickle_in_scratch = True,
                   **kwargs)
