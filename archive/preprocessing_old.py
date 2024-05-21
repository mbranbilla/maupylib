import pandas as pd


class preprocessing:
    def __init__(self):
        pass

    def is_binary(self, series, allow_na=False):
        if allow_na:
            series.dropna(inplace=True)

        evaluation = (series.isin([0, 1]) |
                      series.isin([-1, 1]) |
                      series.isin(['S', 'N']) |
                      series.isin(['Y', 'N']) |
                      series.isin([True, False]) |
                      series.isin(['True', 'False']) |
                      series.isin(['true', 'talse']) |
                      series.isin(['TRUE', 'FALSE']))

        return all(evaluation)

    def create_metadata(self, df, id_cols=[], target_cols=[], numeric_categorical_cols=[]):
        '''
            Function create_metadata(self, df, id_cols=[], target_cols=[], numeric_categorical_cols=[])
            ------------------------

            Creates table with metadata to assist in future manipulations.

            ARGUMENTS:

                - df: is a pandas dataframe which you want create a metadata;

                - id_cols: specify columns that keep id`s;

                - target_cols: In case of prediction or another use that have a target variable, is necessary specify that variable names;

                - numeric_categorical_cols: in case categoric variables keeped with numeric values, is necessary specify that variable names;

        '''

        meta_item = dict()
        metadata = []
        cols = df.columns

        for c in cols:

            if c in id_cols:
                role = 'id'
                keep = 'false'

            elif df[c].isnull().all():
                role = 'empty'
                keep = 'false'

            elif c in target_cols:
                role = 'target'
                keep = 'false'
            else:
                role = 'input'
                keep = True

            dtype = df[c].dropna().dtype

            if self.is_binary(df[c].dropna(), allow_na=True):
                level = 'binary'

            elif (dtype == float) & (c not in numeric_categorical_cols):
                condition_1 = ~self.is_binary(df[c], allow_na=True)
                condition_2 = all((x % 1) == 0 for x in df[c])

                if (condition_1) & (condition_2):
                    level = 'ordinal'
                else:
                    level = 'interval'

            elif (dtype == int) & (c not in numeric_categorical_cols):
                level = 'ordinal'

            elif (dtype == 'object') | (c in numeric_categorical_cols):
                level = 'categoric'

            meta_item = {
                'name': c,
                'role': role,
                'level': level,
                'keep': keep,
                'dtype': dtype
            }
            metadata.append(meta_item)

        metadata = pd.DataFrame(
            metadata, columns=['name', 'role', 'level', 'keep', 'dtype']
        ).set_index('name')

        return metadata
