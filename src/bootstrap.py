from abc import ABC, abstractmethod
from typing import Union

import numpy as np

from pandas import DataFrame, Series, concat
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
from sklearn.metrics import precision_score, recall_score, log_loss
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator
from scipy.stats import t, sem

from tqdm import tqdm


class bootstrap(ABC):

    @abstractmethod
    def get_classification_metrics(y_true: [Series, np.array, list],
                                   y_predicted: [Series, np.array, list],
                                   y_scores: [Series, np.array, list]) -> dict:
        
        # Evaluate binary classification metrics (if multi-class, just evaluate
        # One vs Rest)
        if len(np.unique(y_true)) > 2:
            multi_class = 'ovr'
            metric_avg = 'micro'
        else:
            multi_class = 'raise'
            metric_avg = 'binary'
            y_scores = y_scores[:, 1].ravel()


        return {
            'roc_auc_score': roc_auc_score(y_true, y_scores,
                                           multi_class=multi_class),
            'gini_score': 2 * roc_auc_score(y_true, y_scores, 
                                            multi_class=multi_class) - 1,
            'f1_score': f1_score(y_true, y_predicted, average=metric_avg),
            'average_precision_score': average_precision_score(y_true,
                                                               y_scores),
            'precision_score': precision_score(y_true,
                                               y_predicted, 
                                               average=metric_avg),
            'recall_score': recall_score(y_true, y_predicted,
                                         average=metric_avg),
            'log_loss': log_loss(y_true, y_scores),
            'accuracy_score': accuracy_score(y_true, y_predicted),
        }


    # def get_bs_samples(X, y, bootstrap_samples):
    #     return train_test_split(X, y, test_size= 1/bootstrap_samples)
    @abstractmethod
    def eval_bs_metrics(model: BaseEstimator,
                        X: DataFrame,
                        y: [Series, np.array, list]=None,
                        bootstrap_samples: int = 10, 
                        n_iteractions: int = 1000) -> DataFrame:

        # y_classes = np.unique(y)

        df_metrics = DataFrame(
            columns=['iteraction', 'roc_auc_score', 'gini_score', 'f1_score',
                    'average_precision_score', 'precision_score', 'recall_score',
                    'log_loss', 'accuracy_score']
        )

        for i in tqdm(range(n_iteractions)):
            # Evaluate BootStrap Samples:
            X_train, X_test, y_train, y_test = train_test_split(
                X, 
                y, 
                test_size= 1/bootstrap_samples,
                shuffle=True,
                stratify=y
                )
            
            y_train = y_train.values.ravel()
            y_test = y_test.values.ravel()
            
            model.fit(X_train, y_train);

            y_predicted = model.predict(X_test)
            y_scores = model.predict_proba(X_test)

            _metrics = bootstrap.get_classification_metrics(y_test,
                                                            y_predicted,
                                                            y_scores)
            _metrics['iteraction'] = i

            df_metrics = concat([
                df_metrics,
                DataFrame(_metrics, index=[0])
            ])

        return df_metrics
    

    @abstractmethod
    def get_metrics_ci(model: BaseEstimator,
                       X: DataFrame,
                       y: [Series, np.array, list]=None,
                       return_evaluations_df: bool = True,
                       bootstrap_samples: int = 10, 
                       n_iteractions: int = 1000) -> [dict, DataFrame]:
        
        bs_metrics_eval = bootstrap.eval_bs_metrics(model,
                                                    X,
                                                    y,
                                                    bootstrap_samples,
                                                    n_iteractions)
        
        metrics_ci = {
            'roc_auc_score': [],
            'gini_score': [],
            'f1_score': [],
            'average_precision_score': [],
            'precision_score': [],
            'recall_score': [],
            'log_loss': [],
            'accuracy_score': [],
        }

        for m in tqdm(metrics_ci.keys()):
            metric_mean = np.mean(bs_metrics_eval[m])
            metric_sem = sem(bs_metrics_eval[m])


            # Evaluate confidence interval
            lower_limit, upper_limit = t.interval(
                confidence=.95,
                df=n_iteractions,
                loc=metric_mean,
                scale=metric_sem
            )

            metrics_ci[m].append((lower_limit, upper_limit))
            metrics_ci[m].append(metric_mean)
            metrics_ci[m].append(metric_sem)

        metrics_ci['format_description'] = [('lower_limit', 'upper_limit'), 
                                            'avg', 'std']
        
        if return_evaluations_df:
            return metrics_ci, bs_metrics_eval
        else:
            return metrics_ci
        