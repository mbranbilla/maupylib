from sklearn.utils import compute_sample_weight
import xgboost as xgb
import optuna
from numpy import where, array, inf, abs, exp
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sksurv.metrics import concordance_index_censored
from sklearn.utils.class_weight import compute_sample_weight

class optunaModelOptimizer:
    def __init__(self,
                 objective:str = "binary:logistic",   
                 algorithm:str = "XGBoost",           # Use 'XGBoost' or 'LGBM'
                 eval_metric:str = 'auc',
                 valid_size=0.05,
                 rnd_seed=42,
                 n_trials=1000,
                 num_boost_round=31,
                 params=None) -> None:
        
        self.objective = objective
        self.algorithm = algorithm
        self.eval_metric = eval_metric
        self.valid_size = valid_size
        self.rnd_seed = rnd_seed
        self.n_trials = n_trials
        self.num_boost_round = num_boost_round

        self.surv_objc_cox = 'survival:cox'
        self.surv_objc_aft = 'survival:aft'
        self.params = params
    
    def build_xgb_cox_DMatrix(self,
                              X: DataFrame, 
                              y: array,
                              weights: array=None) -> xgb.DMatrix:
        
        """
        Build a data matrix for XGBoost implementation using objective=survival:cox

        Inputs:
            X: pandas dataframe with independent variables of model
            y: structured array with first field as "event" and second fiels as 
            duration (time)
            weights (optional): sample weights to be considered in training set
                                during the fit process. 

        Return:
            d_matrix: a XGBoost data matrix for use in survival:cox implementation
        """
        
        # Split y into event E and time T, without structured array
        event_field, time_field = y.dtype.names
        E, T = y[event_field], y[time_field]


        # Construct xgb.DMatrix for objective survival:cox
        target = where(E, T, -T)
        d_matrix = xgb.DMatrix(X, label=target)

        if weights is not None:
            d_matrix.set_weight(weights)

        return d_matrix
        

    def build_xgb_aft_DMatrix(self,
                              X: DataFrame, 
                              y: array,
                              weights: array=None) -> xgb.DMatrix:
        
        """
        Build a data matrix for XGBoost implementation using objective=survival:aft

        Inputs:
            X: pandas dataframe with independent variables of model
            y: structured array with first field as "event" and second fiels as 
            duration (time)
            weights (optional): sample weights to be considered in training set
                                during the fit process. 

        Return:
            d_matrix: a XGBoost data matrix for use in survival:aft implementation
        """
        
        # Split y into event E and time T, without structured array
        event_field, time_field = y.dtype.names
        E, T = y[event_field], y[time_field]


        # Construct xgb.DMatrix for objective survival:aft
        d_matrix = xgb.DMatrix(X)

        y_lower_bound = T
        y_upper_bound = where(E, T, inf)

        d_matrix.set_float_info("label_lower_bound", y_lower_bound.copy())
        d_matrix.set_float_info("label_upper_bound", y_upper_bound.copy())
        
        if weights is not None:
            d_matrix.set_weight(weights)

        return d_matrix


    def build_DMatrix(self,
                      X: DataFrame, 
                      y: array,
                      sample_weights=None) -> xgb.DMatrix:
        
        # Treating Survival cases
        if self.objective in [self.surv_objc_cox, self.surv_objc_aft]:
            event_field, _ = y.dtype.names
            stratify_y = y[event_field]
        else:
            stratify_y = y


        # Split train data in order to create a small validation set, for
        # using to evaluate metric during parameter optimization
        X_train, X_valid, y_train, y_valid = train_test_split(
            X,
            y,
            random_state=self.rnd_seed,
            stratify=stratify_y
            )
        
        # Set weights

        # Build XGB Data Matrix
        if self.objective == self.surv_objc_cox:
            w_train = compute_sample_weight(sample_weights,
                                            y_train[event_field])
            dtrain = self.build_xgb_cox_DMatrix(X_train,
                                                y_train,
                                                weights=w_train)
            
            dvalid = self.build_xgb_cox_DMatrix(X_valid, y_valid)

        elif self.objective == self.surv_objc_aft:
            w_train = compute_sample_weight(sample_weights,
                                            y_train[event_field])
            dtrain = self.build_xgb_aft_xgb(X_train,
                                            y_train,
                                            weights=w_train)
            
            dvalid = self.build_xgb_aft_xgb(X_valid, y_valid)

        else:
            w_train = compute_sample_weight(sample_weights,
                                            y_train)
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dvalid = xgb.DMatrix(X_valid, label=y_valid)

            if w_train is not None:
                dtrain.set_weight(w_train)


        return dtrain, dvalid, y_train, y_valid


    def penalized_optimization_metric(self, metric_train, metric_valid):

        if (metric_train == metric_valid) & (metric_train == 0.5):
            return 1e6
        elif abs(metric_train - metric_valid) > 0.05:
            return 1e6
        else:
            return exp(1 + abs(metric_train - metric_valid)) \
                        + (1 / (1+metric_train)) \
                        + (1 / (1+metric_valid)) \
                        + (metric_valid - 1)**2 
        

    def set_params(self, trial):
        if self.params is not None:
            params = self.params
        elif self.algorithm == 'XGBoost':
            params = {
                "verbosity": 0,
                "seed": self.rnd_seed,
                "eval_metric": self.eval_metric,
                "tree_method": "hist",
                # defines booster, gblinear for linear functions.
                "booster": "dart",
                # "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
                # L2 regularization weight.
                "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
                # L1 regularization weight.
                "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
                # sampling ratio for training data.
                "subsample": trial.suggest_float("subsample", 0.2, 1.0),
                # sampling according to each tree.
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
            }

            if params["booster"] in ["gbtree", "dart"]:
                # maximum depth of the tree, signifies complexity of the tree.
                params["max_depth"] = trial.suggest_int("max_depth", 3, 9, step=1)
                params["max_depth"] = 2
                # minimum child weight, larger the term more conservative the tree.
                params["min_child_weight"] = trial.suggest_int("min_child_weight", 1, 3, step=1)
                params["eta"] = trial.suggest_float("eta", 1e-8, 0.02, log=True)
                # defines how selective algorithm is.
                params["gamma"] = trial.suggest_float("gamma", 1e-8, 0.1, log=True)
                params["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])

            if params["booster"] == "dart":
                params["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
                params["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
                params["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
                params["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)

        elif self.algorithm == 'LGBM':
            print("Defalt params for LGBM hasn't been implemented yet")
            params = {}

        else:
            params = {}

        return params


    def optimizer_objective_function(self, trial, dtrain, dvalid,
                                     y_train, y_valid):
        # Setting parameters
        params = self.set_params(trial)
        # Define model object for experiment
        if self.algorithm == 'XGBoost':
            study_model = xgb.train(params, dtrain)
        elif self.algorithm == 'LGBM':
            print("LGBM Optimization hasn't been implemented yet.")
        else:
            print("Unknown algorithm")
            return None
        
        preds_train = study_model.predict(dtrain)
        preds_valid = study_model.predict(dvalid)

        if self.objective in [self.surv_objc_cox, self.surv_objc_aft]:
            metric_train = concordance_index_censored(y_train['c1'], 
                                                      y_train['c2'],
                                                      preds_train)[0]
            
            metric_valid = concordance_index_censored(y_valid['c1'], 
                                                      y_valid['c2'], 
                                                      preds_valid)[0]
        else:
            metric_train = roc_auc_score(y_train, preds_train)
            metric_valid = roc_auc_score(y_valid, preds_valid)

        return self.penalized_optimization_metric(metric_train, metric_valid)


    def run_optimization(self, X, y, sample_weights=None, 
                         return_fitted_model=True):

        # Setting training and validation set
        dtrain, dvalid, y_train, y_valid = self.build_DMatrix(X, 
                                                              y, 
                                                              sample_weights)

        study = optuna.create_study(directions=['minimize'])

        def objective_function(trial):
            return self.optimizer_objective_function(trial,
                                                     dtrain,
                                                     dvalid,
                                                     y_train,
                                                     y_valid)
        
        study.optimize(objective_function, n_trials=self.n_trials)
        
        try:
            print("Single metric optimization")
            params = study.best_params
        except:
            print("Multiobjective Optimization")
            print("Best trials:")

            for t in study.best_trials:
                print(t)

            params = study.best_trials[1].params

        
        params['seed'] = self.rnd_seed
        params['objective'] = self.objective
        params['eval_metric'] = self.eval_metric

        if return_fitted_model:
            # Training model
            if self.algorithm == 'XGBoost':
                model = xgb.train(params,
                                  dtrain,
                                  num_boost_round=self.num_boost_round,
                                  evals=[(dtrain, 'train'), (dvalid, 'valid')])
            elif self.algorithm == 'LGBM':
                print("LGBM Optimization hasn't been implemented yet.")
            else:
                print("Unknown algorithm")
                return None
            
            return model, params
        else:
            return params