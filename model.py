from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler

class Model:
    def __init__(self):
        xgb = XGBClassifier(
            base_score=0.5, booster='gbtree', colsample_bylevel=1,
            colsample_bynode=1, colsample_bytree=0.5, gamma=0.0, gpu_id=-1,
            importance_type='gain', interaction_constraints='',
            learning_rate=0.1, max_delta_step=0, max_depth=15,
            min_child_weight=1, monotone_constraints='()',
            n_estimators=100, n_jobs=0, num_parallel_tree=1, random_state=42,
            reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
            tree_method='exact', validate_parameters=1, verbosity=None)
        
        rf = RandomForestClassifier(random_state=42, n_estimators=120)

        self.model = VotingClassifier(estimators=[('xgb',xgb), ('rf', rf)],voting='hard')
        self.scaler = StandardScaler()

    def fit(self, X, y):
        ros = RandomOverSampler(random_state=42)
        X_ros, y_ros = ros.fit_resample(X, y)
        X_ros = self.scaler.fit_transform(X_ros)
        self.model = self.model.fit(X_ros, y_ros)

    def predict(self, X):
        X = self.scaler.transform(X)
        return self.model.predict(X)


