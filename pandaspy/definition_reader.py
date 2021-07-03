#Read the definition from the dashboard definition file.
#The target is a json. Current aim is to support some data cards.
#{'data cards':{'1':{'card no':1,'type':'compound','sql query':'...', 'sources':[{'source no':1,'table name':'supplier','file path':'test.xls','sheet':'datasheet', 'start row':0},{...},...], 'current_card_end_row':10},'2':{}...}}
# -*- coding: utf-8 -*- 

import re
import pandas as pd
import pandasapy.table_reader as table_reader

format_data_card='data card ([0-9]+)'
format_source='source ([0-9]+)'

log_definition_dataframe_length='There are {} rows of definitions found.'

txt_current_card_end_row='current_card_end_row'
txt_current_source_end_row='current_source_end_row'
txt_type='type'
txt_sql_query='sql query'
txt_source_no='source no'
txt_file_path='file path'
txt_sheet='sheet'
txt_start_row='start row'
txt_table_name='table name'


def get_config(definition_file_path):
	df=table_reader.read_excel_into_dataframe(definition_file_path, None, 'data_cards', 0, 0)
	#print(df)
	output=convert_dataframe_into_json(df)
	
	return output
	
def convert_dataframe_into_json(input_dataframe):
	result_output={}
	data_cards={}
	
	max_row_number=input_dataframe.shape[0]
	print(log_definition_dataframe_length.format(max_row_number))
	current_rownumber=0
	while current_rownumber<max_row_number:
		card_cell_value=input_dataframe.iloc[current_rownumber, 0]
		
		if pd.notnull(card_cell_value) and re.match(format_data_card, card_cell_value):
			card_number=re.findall(format_data_card, card_cell_value)[0]
			#print('card_number: '+str(card_number))
			current_card_dict=collect_a_card_info(current_rownumber, input_dataframe, max_row_number)
			current_rownumber=current_card_dict[txt_current_card_end_row]
			data_cards[card_number]=current_card_dict
		else:	
			current_rownumber=current_rownumber+1
	
	result_output['data cards']=data_cards
	return result_output
	
def collect_a_card_info(start_row, input_dataframe, max_row_number):
	card_dict={}
	sources=[]
	current_rownumber=start_row+1

	while pd.isnull(input_dataframe.iloc[current_rownumber, 0]):
		#print(current_rownumber)
		card_attr_cell_value=input_dataframe.iloc[current_rownumber, 1]
		
		if pd.notnull(card_attr_cell_value):
			if card_attr_cell_value.lower()==txt_type:
				card_dict[txt_type]=input_dataframe.iloc[current_rownumber, 2]
			elif card_attr_cell_value.lower()==txt_sql_query:
				card_dict[txt_sql_query]=input_dataframe.iloc[current_rownumber, 2]
			elif re.match(format_source, card_attr_cell_value):
				source_number=re.findall(format_source, card_attr_cell_value)[0]
				current_source=collect_a_source_info(current_rownumber, input_dataframe, max_row_number)
				current_source[txt_source_no]=source_number
				sources.append(current_source)
				current_rownumber=current_source[txt_current_source_end_row]-1
			
			
		current_rownumber=current_rownumber+1
		if current_rownumber>=max_row_number:
			break
	
	card_dict['sources']=sources
	card_dict[txt_current_card_end_row]=current_rownumber
	return card_dict
		
def collect_a_source_info(start_row, input_dataframe, max_row_number):
	source_dict={}
	current_rownumber=start_row+1
	while pd.isnull(input_dataframe.iloc[current_rownumber, 0]) and pd.isnull(input_dataframe.iloc[current_rownumber, 1]):
		source_attr_cell_value=input_dataframe.iloc[current_rownumber, 2]
		
		if pd.notnull(source_attr_cell_value):
			if source_attr_cell_value.lower()==txt_file_path:
				source_dict[txt_file_path]=input_dataframe.iloc[current_rownumber, 3]
			elif source_attr_cell_value.lower()==txt_sheet:
				source_dict[txt_sheet]=input_dataframe.iloc[current_rownumber, 3]
			elif source_attr_cell_value.lower()==txt_start_row:
				source_dict[txt_start_row]=input_dataframe.iloc[current_rownumber, 3]
			elif source_attr_cell_value.lower()==txt_table_name:
				source_dict[txt_table_name]=input_dataframe.iloc[current_rownumber, 3]
			
		current_rownumber=current_rownumber+1
		if current_rownumber>=max_row_number:
			break
	
	source_dict[txt_current_source_end_row]=current_rownumber
	return source_dict
		
	
	