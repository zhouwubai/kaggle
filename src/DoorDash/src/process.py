import numpy as np
import pandas as pd
import copy
import re


class PreProcess(object):

    def __init__(self):
        self.df = None

    def _standardize_string(self, a_str):
        """Replace whitespace with underscore
           remove non-alphanumeric characters
        """
        if isinstance(a_str, str) or isinstance(a_str, unicode):
            a_str = re.sub(r'\s+', '_', a_str)
            a_str = re.sub(r'\W+', '_', a_str)
            return a_str.lower()
        else:
            return ''

    feature2categorizer = {
        "market_id": _standardize_string,
        # "store_id",
        'store_primary_category': _standardize_string,
        'order_protocol': _standardize_string
    }

    def _categorize_features(self):
        if type(self.df) is dict:
            pass
        else:
            columns_to_dummify = []
            for feature in self.feature2categorizer.keys():
                categorizer = self.feature2categorizer[feature]
                if feature in self.df:
                    # first apply categorizer/replace
                    self.df.loc[:, feature] = self.df[feature].apply(lambda x: categorizer(self, x))
                    # add the column to be dummified
                    columns_to_dummify.append(feature)
            self.df = pd.get_dummies(
                self.df,
                columns=columns_to_dummify).copy(deep=True)

    def preprocess(self, df):
        """
        Returns:
            preprocess dataframe of features, model ready
        """

        if df is None or len(df) == 0:
            raise Exception("Dataframe in Preprocessing is not initilized")
        else:
            if type(df) is dict:
                self.df = copy.deepcopy(df)
            else:
                self.df = df  # this is for offline training, reference is OK

        self._categorize_features()

        return self.df
