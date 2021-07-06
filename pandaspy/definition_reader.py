# -*- coding: utf-8 -*- 
#Read the definition from the dashboard definition file.
#The target is a json. Current aim is to support some data cards.
#{'data cards':{'1':{'card no':1,'type':'compound','sql query':'...', 'sources':[{'source no':1,'table name':'supplier','file path':'test.xls','sheet':'datasheet', 'start row':0},{...},...], 'current_card_end_row':10},'2':{}...}
#'display cards':{'1':{'card no':1, 'data card':'1', 'sheet':'sheet_1', 'top':2, 'left':3, 'header setting':[]}, '2':{...}, ...}}


import re
import pandas as pd
import table_reader as table_reader
import dataframe_operator as df_operator

format_data_card='data card ([0-9]+)'
format_source='source ([0-9]+)'

format_display_card='display card ([0-9]+) based on data card ([0-9]+)'
format_display_direct_column='column *: *([a-zA-Z0-9_]+) *as *([a-zA-Z0-9_]+) *'

log_definition_dataframe_length='There are {} rows of data cards definitions found.'

txt_current_card_end_row='current_card_end_row'
txt_current_source_end_row='current_source_end_row'
txt_type='type'
txt_sql_query='sql query'
txt_source_no='source no'
txt_file_path='file path'
txt_sheet='sheet'
txt_start_row='start row'
txt_table_name='table name'
txt_data_cards='data_cards'
txt_message='message'
txt_definition_parse_success='definition parse success'
txt_card_number='card no'
txt_data_card='data card'
txt_top='top'
txt_left='left'
txt_up_distance='up distance'
txt_header_setting='header setting'
txt_last_row='last row'

def get_config(definition_file_path):
	output={}
	
	#get all the sheets here:
	all_sheets=table_reader.get_all_sheet_names_in_excel(definition_file_path);
	print(all_sheets)
	
	if all_sheets and txt_data_cards in all_sheets:
		df_datacards=table_reader.read_excel_into_dataframe(definition_file_path, None, txt_data_cards, 0, 0)
		data_cards_dict=convert_datacards_into_json(df_datacards)
		output['data cards']=data_cards_dict
		
		#collect all display cards:
		
		all_sheets.remove(txt_data_cards)
		
		display_cards_dict={}
		for eachsheet in all_sheets:
			df_each_display_card=table_reader.read_excel_into_dataframe(definition_file_path, None, eachsheet, 0, 0)
			display_cards_dict.update(convert_display_cards_into_json(eachsheet, df_each_display_card))
		
		output['display cards']=display_cards_dict
	else:
		output[txt_definition_parse_success]=0
		output[txt_message]='Cannot find a "data_cards" sheet.'
		return output
	
	
	
	return output

#Now it is not permited to put 2 display cards into the same row. 
def convert_display_cards_into_json(sheet_name,input_dataframe):
	display_cards_json={}
	previous_row=0
	for i,eachrow in input_dataframe.iterrows():
		for j, eachcell in eachrow.iteritems():
			if pd.notnull(eachcell) and re.match(format_display_card, eachcell):
				#print(eachcell)
				#print(re.findall(format_display_card, eachcell))
				#use an empty card dict
				each_display_card_definition={}
				display_card_number=re.findall(format_display_card, eachcell)[0][0]
				data_card_number=re.findall(format_display_card, eachcell)[0][1]
				each_display_card_definition[txt_card_number]=display_card_number
				each_display_card_definition[txt_data_card]=data_card_number
				each_display_card_definition[txt_sheet]=sheet_name
				each_display_card_definition[txt_top]=i
				each_display_card_definition[txt_left]=j
				each_display_card_definition[txt_up_distance]=i-previous_row
				
				#extract the display settings here.
				header_setting=extract_header_setting(i,j,input_dataframe)
				if len(header_setting)>0:
					each_display_card_definition[txt_header_setting]=header_setting
					previous_row=i+1
				
				display_cards_json[display_card_number]=each_display_card_definition
	
	return display_cards_json


#extract the display settings under a display card cell
def extract_header_setting(top, left,input_dataframe):
	header_setting=[]
	max_row_number=input_dataframe.shape[0]
	current_rownum=top+1
	if current_rownum<max_row_number:
		current_row=input_dataframe.loc[current_rownum, left:]
		if current_row.notnull().any():
			content_in_list=df_operator.cast_row_into_list(current_row, True)
			if content_in_list and len(header_setting)==0:
				header_setting=content_in_list
	
	return header_setting


	
def convert_datacards_into_json(input_dataframe):
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
	
	return data_cards
	
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
		
	
	