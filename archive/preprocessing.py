import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
import platform
from sklearn.feature_selection import VarianceThreshold

class preprocessing_tools:
    variable_name_col = 'variable_name'
    dtype_col = 'dtype'
    qty_distinct_values_col = 'qty_distinct_values'
    qty_missing_values_col = 'qty_missing_values'
    missing_rate_col = 'missing_rate'
    auto_remove_col = 'auto_remove'
    is_target_col = 'is_target'
    is_id_col = 'is_id'
    variance_col = 'variance'
    low_variance_flag_col = 'low_variance_flag'
    default_id_substring = '_id'
    default_categorical_reducer_threshold = 0.05
    qty_alread_dummy_var = 2
    default_variance_threshold = 0.01
    iqr_value_col = 'iqr_value'
    lower_limits_col = 'lower_limits'
    upper_limits_col = 'upper_limits'
    default_norm_mean = 0.0
    default_norm_std = 0.1
    default_target_substring = 'y_'

    def _get_dataframe_overview(self, df, target_cols=[], id_cols=[]):

        cols = df.columns.tolist()

        if len(id_cols) == 0:
            warnings.warn(
                '\nid_cols not provided. Auto-identifying using `{0}` substring.'
                    .format(self.default_id_substring)
            )

            id_cols = [s for s in cols if self.default_id_substring in s]

        overview_dict = {
            self.variable_name_col: [],
            self.dtype_col: [],
            self.qty_distinct_values_col: [],
            self.qty_missing_values_col: [],
            self.missing_rate_col: [],
            self.auto_remove_col: [],
            self.is_target_col: [],
            self.is_id_col: []
        }

        qty_lines = len(df)

        for c in cols:
            qty_unique = df[c].nunique()
            qty_na = df[c].isna().sum()

            overview_dict[self.variable_name_col].append(c)
            overview_dict[self.dtype_col].append(df[c].dtypes)
            overview_dict[self.qty_distinct_values_col].append(qty_unique)
            overview_dict[self.qty_missing_values_col].append(qty_na)
            overview_dict[self.missing_rate_col].append(qty_na / qty_lines)
            overview_dict[self.auto_remove_col].append(qty_unique <= 1)
            overview_dict[self.is_target_col].append(
                any([x in c for x in target_cols]))
            overview_dict[self.is_id_col].append(
                any([x in c for x in id_cols]))

        return pd.DataFrame(overview_dict).set_index(self.variable_name_col)

    def _categorical_reducer(self, df, id_cols=[], cat_vars=[], threshold=None, n_tolerance=2):
        '''
            Agrupa categorias que representam menos que uma porcentagem (threshold) de cada
            variável categórica.
        '''

        # Check inputs and generate warnings in case of default behaviour

        if threshold is None:
            warnings.warn(
                '\n`threshold` not provided to assess the proportion of categories. Using default value {:.1%}'
                    .format(self.default_categorical_reducer_threshold)
            )

            threshold = self.default_categorical_reducer_threshold

        if len(cat_vars) < 1:
            warnings.warn(
                '\n`cat_vars` not provided. Using `dtype=\'object\' to identify categorical variables.')

            cat_vars = df.select_dtypes(include=['object']).columns

        if len(id_cols) < 1:
            warnings.warn(
                '\n`id_cols` not provided. Using `{0}\' substring to identify ids variables.'
                    .format(self.default_id_substring)
            )

            id_cols = [s for s in df.columns if self.default_id_substring in s]

        for c in tqdm(cat_vars):
            if df[c].nunique() <= n_tolerance:
                pass

            else:
                aux_df = pd.DataFrame(df[c].value_counts())
                aux_df['proportion'] = aux_df[c] / aux_df[c].sum()
                aux_df['is_not_significant'] = aux_df['proportion'] < threshold

                not_sig_cats = list(
                    aux_df[aux_df['is_not_significant'] == True].index)

                if len(not_sig_cats) > 1:
                    df[c] = [x if x not in not_sig_cats else 'others' for x in df[c]]

                    print_msg = 'Column: {0} | {1} of {2} categories were replaced by \'others\' ' \
                        .format(c, len(not_sig_cats), len(aux_df.index))

                    print_msg = print_msg + 'as not represent more than {:.1%} of the total' \
                        .format(threshold)

                    print(print_msg)

        return df

    def _categorical_string_normalizer(self, df, id_cols=[], cat_vars=[]):

        # Check inputs and generate warnings in case of default behaviour
        if len(cat_vars) < 1:
            warnings.warn(
                '\n`cat_vars` not provided. Using `dtype=\'object\' to identify categorical variables.')

            cat_vars = df.select_dtypes(include=['object']).columns

        if len(id_cols) < 1:
            warnings.warn(
                '\n`id_cols` not provided. Using `{0}\' substring to identify ids variables.'
                    .format(self.default_id_substring)
            )

            id_cols = [s for s in df.columns if self.default_id_substring in s]

        for c in tqdm(cat_vars):
            df[c] = df[c].str.replace(' ', '_') \
                         .str.replace(r"[^a-zA-Z0-9_.]", "") \
                         .str.lower()

        return(df)

    def _categorical_get_dummies(self, df, cat_vars=[]):
        '''
            Aplica `get_dummies` (pandas) para representação binária das categorias
            de cada variável categórica. 
            Se a variável possuir apenas dois valores, esta não será transformada. 
            As colunas criadas são nomeadas com o prefixo `dv_`, que indica
            'dummy variable', para identificação das variáveis criadas. 
        '''
        cat_vars = list(cat_vars)

        # Check inputs and generate warnings in case of default behaviour
        if len(cat_vars) < 1:
            warnings.warn(
                '\n`cat_vars` not provided. Using `dtype=\'object\' to identify categorical variables.')

            cat_vars = df.select_dtypes(include=['object']).columns

        for c in tqdm(cat_vars):
            if df[c].nunique() <= self.qty_alread_dummy_var:
                cat_vars.remove(c)

        prefix_names = ['dv_' + str(x) + '_is' for x in cat_vars]

        df = pd.get_dummies(data=df, columns=cat_vars, prefix=prefix_names,
                            prefix_sep='_', dummy_na=True)

        dummies_cols = [x for x in df.columns if 'dv_' in x]

        df[dummies_cols] = df[dummies_cols].astype('int64')

        return df

    def _drop_low_variance_var(self, df, numeric_vars=[], id_cols=[], threshold=None):

        if len(numeric_vars) < 1:
            warnings.warn(
                '\n`numeric_vars` not provided. Using `dtype in [\'float64\', \'int64\'] to identify categorical variables.')

            numeric_vars = df.select_dtypes(
                include=['float64', 'int64']).columns

        if len(id_cols) < 1:
            warnings.warn(
                '\n`id_cols` not provided. Using `{0}\' substring to identify ids variables.'
                    .format(self.default_id_substring)
            )

            id_cols = [s for s in df.columns if self.default_id_substring in s]

        if threshold is None:
            warnings.warn(
                '\n`threshold` not provided to flag variables with low variance. Using default value {:.1%}'
                    .format(self.default_variance_threshold)
            )

            threshold = self.default_variance_threshold

        cols = [x for x in numeric_vars if x not in id_cols]

        vt = VarianceThreshold().fit(df[cols])

        df_variances = pd.DataFrame(
            {self.variable_name_col: cols,
             self.variance_col: vt.variances_,
             self.low_variance_flag_col: [1 if x <= threshold else 0 for x in vt.variances_]}
        )

        mask = df_variances.loc[df_variances[self.low_variance_flag_col]
                                == 1, self.variable_name_col].values

        return df.drop(mask, axis=1), df_variances

    def _interquartile_outlier_detection(self, df, numeric_vars=[], id_cols=[]):
        if len(numeric_vars) < 1:
            warnings.warn(
                '\n`numeric_vars` not provided. Using `dtype=\'float64\' to identify categorical variables.')

            numeric_vars = df.select_dtypes(include=['float64']).columns

        if len(id_cols) < 1:
            warnings.warn(
                '\n`id_cols` not provided. Using `{0}\' substring to identify ids variables.'
                    .format(self.default_id_substring)
            )

            id_cols = [s for s in df.columns if self.default_id_substring in s]

        cols = [x for x in numeric_vars if x not in id_cols]

        iqr_values = []
        lower_limits = []
        upper_limits = []

        for c in tqdm(cols):
            iqr = df[c].quantile(q=.75) - df[c].quantile(q=.25)

            iqr_values.append(iqr)
            lower_limits.append(iqr - 0.15*iqr)
            upper_limits.append(iqr + 0.15*iqr)

        return pd.DataFrame({self.variable_name_col: cols,
                             self.iqr_value_col: iqr_values,
                             self.lower_limits_col: lower_limits,
                             self.upper_limits_col: upper_limits})

    def _interquartile_outlier_replacements(self, df, numeric_vars=[], id_cols=[]):
        if len(numeric_vars) < 1:
            warnings.warn(
                '\n`numeric_vars` not provided. Using `dtype=\'float64\' to identify categorical variables.')

            numeric_vars = df.select_dtypes(include=['float64']).columns

        if len(id_cols) < 1:
            warnings.warn(
                '\n`id_cols` not provided. Using `{0}\' substring to identify ids variables.'
                    .format(self.default_id_substring)
            )

            id_cols = [s for s in df.columns if self.default_id_substring in s]

        cols = [x for x in numeric_vars if x not in id_cols]

        df_outliers_info = self._interquartile_outlier_detection(
            df, numeric_vars, id_cols)

        for c in tqdm(cols):
            # mask
            mask = df_outliers_info[self.variable_name_col] == c
            # Get limits
            lower_limit = df_outliers_info.loc[mask,
                                               self.lower_limits_col].values[0]
            upper_limit = df_outliers_info.loc[mask,
                                               self.upper_limits_col].values[0]

            # Lower outliers
            df[c] = [x if x >= lower_limit
                     else lower_limit * (1 + np.random.normal(
                         loc=self.default_norm_mean,
                         scale=self.default_norm_std
                     )) for x in df[c]]

            # Upper outliers
            df[c] = [x if x <= upper_limit
                     else upper_limit * (1 + np.random.normal(
                         loc=self.default_norm_mean,
                         scale=self.default_norm_std
                     )) for x in df[c]]

        return df
