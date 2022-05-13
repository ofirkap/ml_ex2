from sklearn.linear_model import LogisticRegression


class Model:
    def __init__(self):
        self.model = LogisticRegression(max_iter=500)

    def fit(self, X, y):
        self.model = self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


