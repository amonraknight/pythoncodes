import os

import numpy as np
import pandas as pd

import common.Config as cfg
from preprocessing.SampleFileReader import get_designated_sample


class ParameterGenerator:
    def __init__(self):
        self.samples_universe = pd.DataFrame()
        self.charset = {}
        self.charset_in_scope = []

        print('ParameterGenerator started')

    def append_all_samples(self):
        # Only list those useful here.
        all_dfs = [
            get_designated_sample(cfg.FILE_PATH_DATE_1, 'date', 'DATE', 'yyyy-MM-dd'),
            get_designated_sample(cfg.FILE_PATH_DATE_2, 'date', 'DATE', 'yyyy/M/d'),
            get_designated_sample(cfg.FILE_PATH_DATE_3, 'date', 'DATE', 'MM/dd/yyyy'),
            get_designated_sample(cfg.FILE_PATH_DESCRIPTION, 'description', 'TEXT', '*'),
            get_designated_sample(cfg.FILE_PATH_BOOKASIN, 'book_asin', 'TEXT', '*'),
            get_designated_sample(cfg.FILE_PATH_COUNTRYCODE, 'country_code', 'TEXT', 'LLL'),
            get_designated_sample(cfg.FILE_PATH_EMAILADDRESS, 'email_address', 'TEXT', '*'),
            get_designated_sample(cfg.FILE_PATH_GROWTHRATE, 'growth_rate', 'NUMBER', 'float'),
            get_designated_sample(cfg.FILE_PATH_LOCATION, 'location', 'TEXT', '*'),
            get_designated_sample(cfg.FILE_PATH_NUMBEROFDEATH, 'number_of_death', 'NUMBER', 'integer'),
            get_designated_sample(cfg.FILE_PATH_PERSONNAMES, 'person_names', 'TEXT', '*'),
            get_designated_sample(cfg.FILE_PATH_PHONENUMBER, 'phone_number', 'TEXT', '*'),
            get_designated_sample(cfg.FILE_PATH_PRICE, 'price', 'NUMBER', 'float'),
            get_designated_sample(cfg.FILE_PATH_TRADEVALUE, 'trade_value', 'NUMBER', 'float'),
            get_designated_sample(cfg.FILE_PATH_UNIQUEIDS, 'unique_ids', 'TEXT', '*'),
            get_designated_sample(cfg.FILE_PATH_URL, 'url', 'TEXT', '*')
        ]
        self.samples_universe = pd.concat(all_dfs)

    def collect_charset(self):
        for each_text in self.samples_universe['sample']:
            for each_char in set(each_text):
                if each_char in self.charset:
                    self.charset[each_char] = self.charset[each_char] + each_text.count(each_char)
                else:
                    self.charset[each_char] = each_text.count(each_char)

        char_count_df = pd.DataFrame.from_dict(self.charset, orient='index', columns=['count']).reset_index()
        char_count_df.sort_values(by=['count'], ascending=False)
        self.charset_in_scope = list(char_count_df[0:round(char_count_df.shape[0] * cfg.CHARSET_APPLY_RATIO)]['index'])

    def add_parameters(self):
        self.collect_charset()
        self.samples_universe['length'] = self.samples_universe[['sample']].apply(lambda x: len(x['sample']), axis=1)

        # Till here, col 1 is the original text, 2 is desc, 3 is the data type, 5 is the length of the text.

        for each_char in self.charset_in_scope:
            print('Calculating char frequency: "{0}"...'.format(each_char))
            self.samples_universe[each_char] = self.samples_universe[['sample', 'length']].apply(
                lambda x: x['sample'].count(each_char) / x['length'], axis=1)

        # 84 columns in total. 5 is length, 6-84 are frequencies.
        print(self.samples_universe.shape)
        self.samples_universe.to_csv(cfg.OUTPUT_DIMENSION_DATAFRAME, sep=',', index=False, header=True, mode='w')

    def read_existing_samples_universe(self):
        if os.path.exists(cfg.OUTPUT_DIMENSION_DATAFRAME):
            self.samples_universe = pd.read_csv(cfg.OUTPUT_DIMENSION_DATAFRAME, sep=',', header=0)
            print(self.samples_universe.shape)
        else:
            print("Data not found at {}!".format(cfg.OUTPUT_DIMENSION_DATAFRAME))

    def convert_predict_input_to_np(self, input_str):
        if self.samples_universe.shape[0] == 0:
            self.read_existing_samples_universe()
        if self.samples_universe.shape[0] == 0:
            print('No training sample. Cannot predict.')

        self.charset_in_scope = self.samples_universe.columns.values
        self.charset_in_scope = self.charset_in_scope[5:]
        x = [len(input_str)]
        for each_char in self.charset_in_scope:
            x.append(input_str.count(each_char) / len(input_str))

        x = np.array(x)

        x = x[:cfg.FREQUENCY_PARAMETER_AMOUNT]
        return x



    def convert_to_np(self):
        if self.samples_universe.shape[0] == 0:
            self.read_existing_samples_universe()

        if self.samples_universe.shape[0] == 0:
            self.append_all_samples()
            self.add_parameters()

        if self.samples_universe.shape[0] == 0:
            return None, None
        else:
            max_col = cfg.FREQUENCY_PARAMETER_AMOUNT + 4
            if self.samples_universe.shape[1] < max_col:
                max_col = self.samples_universe.shape[1]

            dfvalues = self.samples_universe.values
            y_cat_name = dfvalues[:, 1]
            y = np.array(list(map(cfg.DESC_DICT_TO.get, y_cat_name)))

            x = dfvalues[:, 4:max_col]
            return x, y


pg = ParameterGenerator()
# pg.append_all_samples()
# pg.add_parameters()
# print(pg.samples_universe)
# pg.samples_universe.to_csv(cg.OUTPUT_DIMENSION_DATAFRAME, sep=',', index=False, header=True, mode='w')

# pg.read_existing_samples_universe()
# print(pg.samples_universe)

# x, y = pg.convert_to_np()

pg.convert_predict_input_to_np('aaaaaaa')
