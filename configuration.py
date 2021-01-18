import pandas as pd
from numpy import str


def config():

    # CONFIGURATION DATA
    # --------------------------

    # Twitter
    tw = pd.read_excel('ConfigurationDataSa.xlsx',
                       'Twitter',
                       header=None,
                       index_col=0).to_dict(orient='dict')[1]

    # News
    news = pd.read_excel('ConfigurationDataSa.xlsx',
                         'News',
                         header=None,
                         index_col=0).to_dict(orient='dict')[1]


    return tw, news
