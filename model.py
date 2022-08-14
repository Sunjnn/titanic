from sklearn.ensemble import RandomForestClassifier


def get_RFC(n_estimators=100, max_depth=5, random_state=1):
    model = RandomForestClassifier(n_estimators, max_depth=max_depth, random_state=random_state)
    return model


if __name__ == "__main__":
    pass