from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler

class Model:
    def __init__(self):
        xgb = XGBClassifier(
            base_score=0.5, booster='gbtree', colsample_bylevel=1,
            colsample_bynode=1, colsample_bytree=0.5, gamma=0.4, gpu_id=-1,
            importance_type='gain', interaction_constraints='',
            learning_rate=0.15, max_delta_step=0, max_depth=4,
            min_child_weight=1, monotone_constraints='()',
            n_estimators=100, n_jobs=0, num_parallel_tree=1, random_state=42,
            reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
            tree_method='exact', validate_parameters=1, verbosity=None)

        lgbm = LGBMClassifier(
            colsample_bytree=0.8104721917981516, min_child_samples=118,
            min_child_weight=1, num_leaves=30, random_state=42, reg_alpha=0,
            reg_lambda=0.1, subsample=0.8243603847850702)

        hgb = HistGradientBoostingClassifier(
            max_depth=5, max_leaf_nodes=30,
            min_samples_leaf=5, random_state=42)
        
        self.model = VotingClassifier(estimators=[('xgb',xgb), ('hgb', hgb), ('lgbm', lgbm)], voting='soft', flatten_transform=False)
        self.scaler = StandardScaler()

    def fit(self, X, y):
        X_transformed = self.scaler.fit_transform(X)
        self.model = self.model.fit(X_transformed, y)

    def predict(self, X):
        return self.model.predict(self.scaler.transform(X))


