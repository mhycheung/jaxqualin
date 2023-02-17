import json
from Waveforms import *

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
JSON_SAVE_PATH = os.path.join(ROOT_PATH, "json")

inject_params_base = {
    'l' : 2,
    'm' : 2,
    'Mf' : 1,
    'af' : 0.7,
    'inj_dict' : {
        "2.2.0": (1, 1),
        "2.2.1": (3, 4),
        "2.1.0": (8e-4, 3.5),
        "3.3.0": (0.1, 1),
        "3.2.0": (0.05, 1.5),
        "4.2.0": (4e-4, 0.5),
        "4.4.0": (1e-3, 2),
        "2.-2.0": (1e-5, 3),
        "3.-2.0": (5e-6, 2.5),
        "constant": (4e-6, 3),
    },
    'relevant_lm_list' : [(2, 2), (3, 2), (3, 3), (4, 4), (2, 1)],
    'noise_sig' : 5e-7
}

randomize_params = {
    # 'af' : (0.3, 0.9),
    # 'noise_sig' : (5e-7, 1e-5)
}

phase_random_range = (0, 2*np.pi)
overtone_random_range = (1, 3)
spheroidal_random_range = (1e-3, 1e-2)
recoil_random_range = (1e-4, 1e-2)
retro_random_range = (1e-5, 1e-3)
inj_dict_randomize = {
    "2.2.1": (overtone_random_range, phase_random_range),
    "2.1.0": (recoil_random_range, phase_random_range),
    "3.3.0": (recoil_random_range, phase_random_range),
    "3.2.0": (spheroidal_random_range, phase_random_range),
    "4.2.0": (spheroidal_random_range, phase_random_range),
    "4.4.0": (recoil_random_range, phase_random_range),
    "2.-2.0": (retro_random_range, phase_random_range),
    "3.-2.0": (retro_random_range, phase_random_range),
    "constant": ((1e-6, 5e-5), phase_random_range),
}

amp_order = [["3.2.0", "4.2.0"], ["2.-2.0", "3.-2.0"]]

inject_params_full = {}

for i in range(10):
    inject_params = make_random_inject_params(inject_params_base, randomize_params, inj_dict_randomize, amp_order)
    runname = "eff1"
    inject_params_full[f"{runname}_{i:03d}"] = inject_params
    
with open(f"{JSON_SAVE_PATH}/{runname}.json", 'w') as f:
    json.dump(inject_params_full, f, indent = 4)
