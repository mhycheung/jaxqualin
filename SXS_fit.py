import QuasinormalMode
from QuasinormalMode import *
import Waveforms
from Waveforms import *
import Fit
from Fit import *
import utils
from utils import *
import plot
from plot import *
import ModeSelection
from ModeSelection import *
import configparser
import argparse

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(ROOT_PATH, "config")

parser = argparse.ArgumentParser()

parser.add_argument('SXS_num')
parser.add_argument('setting_name')
parser.add_argument('plot_name')
parser.add_argument("-l", "--load-pickle", dest = "load_pickle", help="load pickle",
                    action='store_true')
parser.add_argument("-ml", "--mode-searcher-load-pickle", dest = "mode_searcher_load_pickle", help="load pickle for mode searcher",
                    action='store_true')
parser.add_argument("-cce", dest = "CCE", help="use CCE", action = 'store_true')

args = parser.parse_args()

SXS_num = args.SXS_num
setting_name = args.setting_name
plot_name = args.plot_name
load_pickle = args.load_pickle
mode_searcher_load_pickle = args.mode_searcher_load_pickle
CCE = args.CCE

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
               mode_searcher_kwargs = mode_searcher_kwargs,
               CCE = CCE)
    
mode_search_complete = ModeSearchAllFreeVaryingNSXSAllRelevant(
                                                    SXS_num, 
                                                    load_pickle = load_pickle, 
                                                    mode_searcher_load_pickle = mode_searcher_load_pickle,
                                                    postfix_string = setting_name,
                                                    pickle_in_scratch = True,
                                                    **kwargs
                                                              )
mode_search_complete.do_all_searches()
Mf = mode_search_complete.relevant_lm_mode_searcher_varying_N[0].M
af = mode_search_complete.relevant_lm_mode_searcher_varying_N[0].a
predicted_qnm_list = []
plot_relevant_mode_search_full(mode_search_complete, predicted_qnm_list = predicted_qnm_list, postfix_string=plot_name)
