# -*- coding: utf-8 -*-  
#Provides methods to read data from CSV, xls, xlsx into a pandas dataframe.


import os
import pandas as pd

log_1='Reading from file {}. File_extension is {}'

def get_all_sheet_names_in_excel(file_path):
	sheet_list=[]
	df = pd.read_excel(file_path, sheet_name=None)
	for eachsheet in df.keys():
		sheet_list.append(eachsheet)
	return sheet_list

def read_file_into_dataframe_default(file_path):
	file_extension=os.path.splitext(file_path)[-1]
	#print(log_1.format(file_path, file_extension))
	if file_extension in ['.csv']:
		return read_csv_into_dataframe(file_path, 0, 0, 0)
	elif file_extension in ['.xls', '.xlsx']:
		return read_excel_into_dataframe(file_path,0, 0, 0, 0)

def read_csv_into_dataframe(file_path, has_header, startFromRow, footerRowNum, **kwargs):
	#print('read_csv_into_dataframe is called.')	
	df=pd.read_csv(file_path, header=has_header, index_col=None, skiprows=startFromRow, skipfooter=footerRowNum, **kwargs);
	return df
	
def read_excel_into_dataframe(file_path, has_header, sheet, startFromRow, footerRowNum, **kwargs):
	#print('read_excel_into_dataframe is called.')
	df=pd.read_excel(file_path, sheet_name=sheet, header=has_header, index_col=None, skiprows=startFromRow, skipfooter=footerRowNum, **kwargs);
	return df
