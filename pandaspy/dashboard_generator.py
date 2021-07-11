# -*- coding: utf-8 -*- 
#Generate an xls/xlsx according to the definition file.
#command sample: python dashboard_generator.py "E:\testfield\python\pandasapy\testdata\dashboard_definition.xlsx" "E:\testfield\python\pandasapy\testoutput\dashboardresult.xlsx"
#command sample: python dashboard_generator.py "testdata\dashboard_definition.xlsx" "testoutput\dashboardresult.xlsx"

import sys
import re
import definition_reader
import table_reader as table_reader
import pandassql_query_executer as query_executer
import dataframe_operator as df_operator
import table_writer as df_writer

log_incorrect_arguments='Please enter 2 parameters, one is the dashboard definition excel, another is the output path.'
log_input_parameter1='The definition file is "{}".'
log_input_parameter2='The destination file is "{}".'

txt_type='type'
txt_compound='compound query'
txt_sources='sources'
txt_file_path='file path'
txt_other_card='other card'
txt_sheet='sheet'
txt_start_row='start row'
txt_table_name='table name'
txt_source_no='source no'
txt_sql_query='sql query'
txt_data_cards='data cards'
txt_display_cards='display cards'
txt_data_card='data card'
txt_header_setting='header setting'
txt_top='top'
txt_left='left'
txt_up_distance='up distance'


format_header_setting='column:([a-zA-Z0-9_]+) as ([a-zA-Z0-9_ ]+)'
format_data_card='data card ([0-9]+)'

#Needs 2 parameters: 
def main_process(argv):
	if len(argv) !=2:
		print(log_incorrect_arguments)
		return
	
	print(log_input_parameter1.format(argv[0]))
	print(log_input_parameter2.format(argv[1]))
	
	#get definitions
	dashboard_definition=definition_reader.get_config(argv[0])
	print(dashboard_definition)
	
	#prepare the dataframes
	dataframes=get_dataframes_from_definition(dashboard_definition[txt_data_cards])
	
	#print dataframes according to the display settings
	print_data_frames_according(argv[1], dataframes, dashboard_definition[txt_display_cards])
	

def print_data_frames_according(output_path, dataframes, display_cards):
	
	#the last row of each sheet
	cursor_row_eachsheet={}
	#sheet name:card no:top in design: top real 
	previous_sheet='';
	previous_design_top=-1;
	previous_real_top=-1;
	previous_rownum=0;
	
	for card_no, each_display_setting in display_cards.items():
		corresponding_df=dataframes[each_display_setting[txt_data_card]]
		
		#Now only supports the header settings. 
		if txt_header_setting in each_display_setting:
			header_setting_list=each_display_setting[txt_header_setting]
			
			wanted_columns=[]
			rename_settings={}
			for each_header in header_setting_list:
				source_target_name_couple=re.findall(format_header_setting, each_header)[0]
				wanted_columns.append(source_target_name_couple[0])
				rename_settings[source_target_name_couple[0]]=source_target_name_couple[1]
			
			corresponding_df=df_operator.acquire_wanted_columns_from_dataframe(corresponding_df, wanted_columns, rename_settings)
			#print(corresponding_df)
			
		#print data frame()
		
		#calculate the row number
		current_row_cursor=0
		#If at the same row as the previous displaycard.
		if each_display_setting[txt_sheet]==previous_sheet and each_display_setting[txt_top]==previous_design_top:
			current_row_cursor=previous_real_top
		elif each_display_setting[txt_sheet] in cursor_row_eachsheet:
			current_row_cursor=cursor_row_eachsheet[each_display_setting[txt_sheet]]
			current_row_cursor=current_row_cursor+each_display_setting[txt_up_distance]
		else:
			current_row_cursor=each_display_setting[txt_top]
		
		#df_writer.add_to_excel(corresponding_df, output_path, each_display_setting[txt_sheet], current_row_cursor, each_display_setting[txt_left])
		df_writer.append_to_excel(corresponding_df, output_path, each_display_setting[txt_sheet], current_row_cursor, each_display_setting[txt_left])
		
		previous_sheet=each_display_setting[txt_sheet]
		previous_design_top=each_display_setting[txt_top]
		previous_real_top=current_row_cursor
		if corresponding_df.shape[0]>previous_rownum:
			previous_rownum=corresponding_df.shape[0]
		
		if each_display_setting[txt_sheet]==previous_sheet and each_display_setting[txt_top]==previous_design_top:
			current_row_cursor=current_row_cursor+previous_rownum
		else:
			current_row_cursor=current_row_cursor+corresponding_df.shape[0]
		cursor_row_eachsheet[each_display_setting[txt_sheet]]=current_row_cursor
	
	print(cursor_row_eachsheet)

#input is the data cards definition	
def get_dataframes_from_definition(data_cards):
	dataframes={}
	
	for card_no, card_def in data_cards.items():
		#dealing with all data card types
		if card_def[txt_type]==txt_compound:
			dataframes[card_no]=get_compound_df(dataframes, card_def)
		#There can be other types more 
		
		print(dataframes[card_no])
	
	return dataframes

#In the future, support using the existing dfs as the inputs.
def get_compound_df(existing_dataframes, card_def):
	dataframe_map={}
	sql_query=card_def[txt_sql_query]
	for each_source in card_def[txt_sources]:
		each_df={}
		#if a file path is given
		if txt_file_path in each_source:
			each_df=table_reader.read_excel_into_dataframe(each_source[txt_file_path], 0, each_source[txt_sheet], each_source[txt_start_row], 0)
		elif txt_other_card in each_source:
			existing_card_no=re.findall(format_data_card, each_source[txt_other_card])[0]
			each_df=existing_dataframes[existing_card_no]
		dataframe_map[each_source[txt_table_name]]=each_df
	
	result_df=query_executer.execute_query_upon_dataframes(sql_query, dataframe_map);
	
	return result_df;

if __name__=="__main__":
	main_process(sys.argv[1:])
