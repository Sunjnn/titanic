import pandas as pd


from data import read_traintest
from model import get_svm as get_model


def main():
    train_data, test_data = read_traintest()
    train_data.Age = train_data.Age.fillna(0)
    test_data.Age = test_data.Age.fillna(0)
    test_data.Fare = test_data.Fare.fillna(0)

    y = train_data['Survived']

    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
    X = pd.get_dummies(train_data[features])
    X_test = pd.get_dummies(test_data[features])

    age_max = max(X.Age.max(), X_test.Age.max())
    X.Age = X.Age / age_max
    X_test.Age = X_test.Age / age_max

    fare_max = max(X.Fare.max(), X_test.Fare.max())
    X.Fare = X.Fare / fare_max
    X_test.Fare = X_test.Fare / fare_max

    model = get_model()
    model.fit(X, y)
    pred = model.predict(X_test)

    output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': pred})
    output.to_csv('submission.csv', index=False)
    print("Your submission was successfully saved!")


if __name__ == '__main__':
    main()