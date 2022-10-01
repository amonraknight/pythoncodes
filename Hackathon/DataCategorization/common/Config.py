# Provide all the configurations

BASE_PATH = r'E:\\development\\Hackathon\\datasets\\datatype_samples\\'
# Date in YYYY-MM-DD
FILE_PATH_DATE_1 = BASE_PATH + 'DATE_YYYY-MM-DD.CSV'
# Date in YYYY/M/D
FILE_PATH_DATE_2 = BASE_PATH + 'DATE_YYYY_M_D.CSV'
# Date in MM/DD/YYYY
FILE_PATH_DATE_3 = BASE_PATH + 'DATE_DD_MM_YYYY.CSV'
# Description, long text
FILE_PATH_DESCRIPTION = BASE_PATH + 'DESCRIPTIONS_TEXT.CSV'
# Book ASIN code, text
FILE_PATH_BOOKASIN = BASE_PATH + 'BOOKASIN_TEXT_ID.CSV'
# Country code in 3 chars
FILE_PATH_COUNTRYCODE = BASE_PATH + 'COUNTRYCODE_TEXT_3C.CSV'
# Email addresses, Text.
FILE_PATH_EMAILADDRESS = BASE_PATH + 'EMAILADDRESSES_TEXT.CSV'
# Growth rate, float, can be negative
FILE_PATH_GROWTHRATE = BASE_PATH + 'GROWTHRATE_NUMBER_FLOAT.CSV'
# Location, text
FILE_PATH_LOCATION = BASE_PATH + 'LOCATION_TEXT.CSV'
# Integer
FILE_PATH_NUMBEROFDEATH = BASE_PATH + 'NUMBEROFDEATH_NUMBER_INTEGER.CSV'
# Person names, text
FILE_PATH_PERSONNAMES = BASE_PATH + 'PERSONNAMES_TEXT.CSV'
# Phone numbers, text
FILE_PATH_PHONENUMBER = BASE_PATH + 'PHONENUMBER_TEXT.CSV'
# Price, float, has thousand separators
FILE_PATH_PRICE = BASE_PATH + 'PRICE_NUMBER_THOUSANDSEPARATOR.CSV'
# Trade value, float
FILE_PATH_TRADEVALUE = BASE_PATH + 'TRADEVALUE_NUMBER_FLOAT.CSV'
# IDs, text
FILE_PATH_UNIQUEIDS = BASE_PATH + 'UNIQUEIDS_TEXT_ID.CSV'
# URL, text
FILE_PATH_URL = BASE_PATH + 'URL_TEXT_HTTPLINK.CSV'

# Base path for lists with dirty data
DIRTY_DATA_BASE_PATH = r'E:\\development\\Hackathon\\datasets\\ill_formed_data\\'

DIRTY_DATA_IN_DATE_1 = DIRTY_DATA_BASE_PATH + 'DIRTY_DATA_IN_YYYY-MM-DD.CSV'
DIRTY_DATA_IN_ID = DIRTY_DATA_BASE_PATH + 'DIRTY_DATA_UNIQUEIDS_TEXT_ID.CSV'


MAX_SAMPLE_FROM_EACH_FILE = 3000

# Ratio of charset applied
CHARSET_APPLY_RATIO = 0.95

FREQUENCY_PARAMETER_AMOUNT = 45

OUTPUT_PATH = r'E:\\development\\Hackathon\\outputs\\'
OUTPUT_DIMENSION_DATAFRAME = OUTPUT_PATH+'calculated_dimensions'

OUTPUT_MODEL_SDGCLASSIFIER = OUTPUT_PATH+'SDGCLASSIFIER'
OUTPUT_MODEL_SDGCLASSIFIER_SCALER = OUTPUT_PATH+'SDGCLASSIFIER_SCALER'

DESC_DICT_TO = {
    'date': 1,
    'description': 2,
    'book_asin': 3,
    'unique_ids': 3,
    'country_code': 4,
    'email_address': 5,
    'growth_rate': 6,
    'price': 6,
    'trade_value': 6,
    'number_of_death': 7,
    'url': 8,
    'phone_number': 9,
    'location': 10,
    'person_names': 10
}

