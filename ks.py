import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings


class k_s:
    qty_deciles = 10
    labels = None
    ks_stats_col = 'ks_stats'
    decile_col = 'decile'
    prob_col = 'prob'
    target_cols = 'target'
    non_target_col = 'non-target'
    qty_goods_col = 'qty_goods'
    min_score_col = 'min_score'
    max_score_col = 'max_score'
    qty_bads_col = 'qty_bads'
    total_goods_plus_bads_col = 'total_goods_plus_bads'
    feature_name_col = 'feature_name'    
    
    
    def _set_labels(self, probs=[]):
        
        self.labels = np.unique(
            list(pd.qcut(probs, self.qty_deciles,
                         duplicates='drop', labels=False))
        )
        
    
    def _check_unique_values(self, probs=[]):
        if len(np.unique(probs)) == 1:
            return True
        else:
            return False
    
    
    def _make_gain_table(self, probs=[], target=[]):
        
        self._set_labels(probs)
        gain_table = pd.DataFrame(
            {
                self.target_cols: target,
                self.non_target_col: 1 - target,
                self.prob_col: probs,
                self.decile_col: pd.qcut(probs,self.qty_deciles,
                                         duplicates='drop', 
                                         labels=self.labels)
            }
        )
        
        gain_table = pd.pivot_table(
            data=gain_table,
            index=[self.decile_col],
            values=[self.target_cols,self.non_target_col,self.prob_col],
            aggfunc={
                self.target_cols:[np.sum],
                self.non_target_col:[np.sum],
                self.prob_col : [np.min,np.max]
            }
        ).reset_index()
        
        gain_table.columns = [self.decile_col, self.qty_goods_col,
                              self.max_score_col, self.min_score_col,
                              self.qty_bads_col]
        
        gain_table = gain_table[[self.decile_col, self.qty_goods_col, 
                                 self.qty_bads_col, self.min_score_col, 
                                 self.max_score_col]]
        
        gain_table = gain_table.sort_values(by=self.min_score_col,ascending=False)
        
        gain_table[self.total_goods_plus_bads_col] = gain_table[self.qty_goods_col] + gain_table[self.qty_bads_col]
        
        gain_table['good_rate'] = gain_table[self.qty_goods_col] / gain_table[self.total_goods_plus_bads_col]
        
        gain_table['bad_rate'] = gain_table[self.qty_bads_col] / gain_table[self.total_goods_plus_bads_col]
        
        total_goods = gain_table[self.qty_goods_col].sum()
        
        gain_table['good_percentage'] = gain_table[self.qty_goods_col] / total_goods
                                          
        total_bads = gain_table[self.qty_bads_col].sum()
        
        gain_table['bad_percentage'] = gain_table[self.qty_bads_col] / total_bads
        
        
        gain_table[self.ks_stats_col] = ((gain_table[self.qty_goods_col] / gain_table[self.qty_goods_col].sum()) \
                 .cumsum() - (gain_table[self.qty_bads_col] / gain_table[self.qty_bads_col].sum()) \
                     .cumsum())
        
        return gain_table
    
    
    def _get_ks_of_features(self, df, id_cols=['id'], target_col=None):
        
        feature_cols = [x for x in df.columns if x not in id_cols + [target_col]]
        
        ks_dict = {self.feature_name_col: [], 
                   self.ks_stats_col: [], 
                   self.decile_col: [],
                   self.min_score_col: [],
                   self.max_score_col: []
                  }
        
        target = df[target_col].values
        
        for f in feature_cols:
            
            probs = df[f].values
            
            if self._check_unique_values(probs):
                warnings.warn("\nAll values finded in variable {0} is the same.".format(f))
            else:
                gain_table = self._make_gain_table(
                    probs=probs,
                    target=target
                )

                ks_argmax = gain_table[self.ks_stats_col].argmax()

                ks_dict[self.feature_name_col].append(f)

                ks_dict[self.ks_stats_col].append(
                    gain_table.iloc[ks_argmax][self.ks_stats_col]
                )

                ks_dict[self.decile_col].append(
                    gain_table.iloc[ks_argmax][self.decile_col]
                )

                ks_dict[self.min_score_col].append(
                    gain_table.iloc[ks_argmax][self.min_score_col]
                )

                ks_dict[self.max_score_col].append(
                    gain_table.iloc[ks_argmax][self.max_score_col]
                )
        
        return pd.DataFrame(ks_dict).sort_values(by=self.ks_stats_col,
                                                 ascending=False)
    
    
    def _format_ks_table(self, df, id_cols=['id'], target_col=None):
        
        ks_df = self._get_ks_of_features(df, id_cols, target_col)

        ks_df[self.ks_stats_col] = pd.Series(
            ["{0:.2f}%".format(val * 100) for val in ks_df[self.ks_stats_col]],
            index = ks_df.index
        )
        
        return ks_df.reset_index(drop=True)
