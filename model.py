from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.model_selection import GridSearchCV


def get_RFC(n_estimators=150, max_depth=6, random_state=1):
    model = RandomForestClassifier(n_estimators, max_depth=max_depth, random_state=random_state)
    return model


class mySVM():
    def __init__(self):
        self.svc = svm.SVC()

    def fit(self, data, target, 
            parameters={'kernel': ('linear', 'poly', 'rbf', 'sigmoid'), 
                        'C': list(range(1, 11)), 
                        'gamma': [x / 200 for x in range(400)],
                        'degree': list(range(3, 6)),
                        'coef0': [x / 200 for x in range(400)]}):
        self.clf = GridSearchCV(self.svc, parameters, n_jobs=-1)
        self.clf.fit(data, target)

    def predict(self, data):
        return self.clf.predict(data)


def get_svm():
    model = mySVM()
    return model


if __name__ == "__main__":
    pass