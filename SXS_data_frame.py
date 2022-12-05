from postprocess import *

SXS_num_list_1 = [str(SXS_num) for SXS_num in range(1419,1510)]
SXS_num_list_2 = ["0" + str(SXS_num) for SXS_num in range(209,306)]
SXS_num_list = SXS_num_list_1 + SXS_num_list_2
create_data_frame(SXS_num_list, [5, 6, 7, 8, 9, 10], "2_run1")
