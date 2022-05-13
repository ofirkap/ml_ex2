import pickle
from sklearn.metrics import accuracy_score
from model import Model


def sanity_test():
    with open('data/ass2.pickle', 'rb') as handle:
        data = pickle.load(handle)

    X_train, y_train = data['train']
    X_dev, y_dev = data['dev']
    clf = Model()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_dev)
    print('Dev error: ', accuracy_score(y_pred, y_dev))


if __name__ == "__main__":
    sanity_test()
