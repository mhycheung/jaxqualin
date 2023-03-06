from postprocess import *
from pathlib import Path
import os

batch_runname = "eff1"
N_list = [5, 6, 7, 8, 9, 10]
eff_num_len = 10

eff_num_list = list(range(eff_num_len))
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
df_path = os.path.join(ROOT_PATH, f"pickle/{batch_runname}.csv")

df_file = Path(df_path)
if df_file.is_file():
    df = pd.read_csv(df_path)
else:
    df = create_data_frame_eff(eff_num_list, batch_runname, N_list, df_save_prefix = batch_runname,
                           l = 2, m = 2)