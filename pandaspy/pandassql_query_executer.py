#Rename the dataframes and execute the queries.
# -*- coding: utf-8 -*-  
from pandasql import sqldf

def execute_query_upon_dataframes(sql_query, dataframe_map):
	for eachkey in dataframe_map:
		#print('currentkey: '+eachkey)
		globals()[eachkey]=dataframe_map[eachkey]
	
	result_df= sqldf(sql_query, globals()).head()
	return result_df