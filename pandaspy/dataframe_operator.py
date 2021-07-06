# -*- coding: utf-8 -*-

import pandas as pd
		
def cast_row_into_list(a_row, stop_at_none=False):
	if not a_row.isnull().all():
		result_list=[]
		for i, eachcell in a_row.iteritems():
			if stop_at_none and pd.isnull(eachcell):
				break
			else:
				result_list.append(eachcell)
		return result_list

		
def acquire_wanted_columns_from_dataframe(input_df, wanted_column_list, rename_setting_dict=None):
	sub_df=input_df[wanted_column_list]
	
	if rename_setting_dict:
		sub_df.rename(columns=rename_setting_dict, inplace = True)
	
	return sub_df