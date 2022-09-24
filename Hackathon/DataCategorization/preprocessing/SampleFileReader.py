import pandas as pd
import common.Config as cfg

def get_designated_sample(filepath, desc, datatype, pattern):
    sample_list = pd.read_csv(filepath, sep='\t', dtype='str', header=None, nrows=cfg.MAX_SAMPLE_FROM_EACH_FILE)
    sample_list.columns = {'sample'}
    sample_list.insert(1, 'desc', desc)
    sample_list.insert(2, 'datatype', datatype)
    sample_list.insert(3, 'pattern', pattern)

    return sample_list
