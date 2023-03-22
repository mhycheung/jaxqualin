import os
import configparser
import argparse
import pandas as pd

from ModeSelection import *
from postprocess import *

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(ROOT_PATH, "config")
DF_SAVE_PATH = os.path.join(ROOT_PATH, "pickle/data_frame")

parser = argparse.ArgumentParser()

parser.add_argument('batch_runname')
parser.add_argument('i', type = int)
parser.add_argument('setting_name')
parser.add_argument("-nd", "--no_delay", dest = "delay", action='store_false')
parser.add_argument("-l", "--load-pickle", dest = "load_pickle", 
                    action='store_true')

args = parser.parse_args()

batch_runname = args.batch_runname
i = args.i 
setting_name = args.setting_name
delay = args.delay
load_pickle = args.load_pickle

DF_SAVE_PATH_INDIV = os.path.join(DF_SAVE_PATH, f"{batch_runname}")

os.makedirs(DF_SAVE_PATH_INDIV, exist_ok=True)

runname = f"{batch_runname}_{i:03d}"

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

mode_searcher = read_json_eff_mode_search(i, batch_runname,  
                                          load_pickle = load_pickle, 
                                          delay = delay,
                                          **kwargs)

df = pd.DataFrame(columns=["eff_num", "M_rem", "chi_rem",  "l", "m", "mode_string",
                               "A_med", "A_hi", "A_low", "phi_med", "phi_hi", "phi_low"])

l = mode_searcher.l
m = mode_searcher.m

df = append_A_and_phis(mode_searcher, df, l=l, m=m, eff_num = i)

file_path = os.path.join(DF_SAVE_PATH_INDIV, f"{batch_runname}_{i:03d}.csv")
df.to_csv(file_path)