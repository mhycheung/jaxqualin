import sys
import os
import configparser

from ModeSelection import *

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(ROOT_PATH, "config")

batch_runname = sys.argv[1]
i = int(sys.argv[2])
runname = f"{batch_runname}_{i:03d}"

setting_name = sys.argv[3]

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

if len(sys.argv) > 4:
    kwargs_in = eval(sys.argv[4])
    kwargs.update(kwargs_in, flatness_checker_kwargs = flatness_checker_kwargs,
               mode_searcher_kwargs = mode_searcher_kwargs)
else:
    kwargs.update(flatness_checker_kwargs = flatness_checker_kwargs,
               mode_searcher_kwargs = mode_searcher_kwargs)

mode_searcher = read_json_eff_mode_search(i, batch_runname,  
                                          load_pickle = True, 
                                          **kwargs)