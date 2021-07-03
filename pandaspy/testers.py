import sys
sys.path.append("..")
import pandasapy.table_reader as table_reader
import pandasapy.pandassql_query_executer as query_executer
import pandasapy.definition_reader as definition_reader

def main_process():
	read_definition()

def get_join_result():
	path_supplier='E:\\testfield\\python\\pandasapy\\testdata\\supplier_info.xlsx'
	path_buy='E:\\testfield\\python\\pandasapy\\testdata\\list_to_buy.xlsx'
	path_product='E:\\testfield\\python\\pandasapy\\testdata\\product_info.xlsx'

	df_supplier=table_reader.read_file_into_dataframe_default(path_supplier)
	df_buy=table_reader.read_file_into_dataframe_default(path_buy)
	df_product=table_reader.read_file_into_dataframe_default(path_product)

	dataframe_map={'supplier':df_supplier, 'buy':df_buy, 'product':df_product}
	sql_query='select buy.num, buy.product_num, buy.amount, product.product_name, supplier.supplier_id, supplier.supplier_name, supplier.email_address from buy left join product on buy.product_num=product.product_num left join supplier on product.supplier_id=supplier.supplier_id;';

	result_df=query_executer.execute_query_upon_dataframes(sql_query, dataframe_map);

	print(result_df)

def read_definition():
	config_df=definition_reader.get_config('E:\\testfield\\python\\pandasapy\\testdata\\dashboard_definition.xlsx')
	print(config_df)

if __name__=="__main__":
	main_process()
