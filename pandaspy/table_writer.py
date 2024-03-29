# -*- coding: utf-8 -*- 

import pandas as pd
import os
import openpyxl
	
	
#Can put different dfs into the same sheet.	
def append_to_excel(input_df, output_path, target_sheet_name, top, left, **kwargs):
	if os.path.isfile(output_path):
		workbook = openpyxl.load_workbook(output_path)
		writer = pd.ExcelWriter(output_path, engine='openpyxl')
		writer.book = workbook
		writer.sheets = dict((ws.title, ws) for ws in workbook.worksheets)
		input_df.to_excel(writer, sheet_name=target_sheet_name,startrow=top, startcol=left, index=False, index_label=None, **kwargs)
		writer.save()
		writer.close()
	else:
		input_df.to_excel(output_path,sheet_name=target_sheet_name,startrow=top, startcol=left, index=False, index_label=None, **kwargs)
