import pandas as pd


from data import read_traintest
from model import get_RFC


def main():
    train_data, test_data = read_traintest()

    y = train_data['Survived']

    features = ['Pclass', 'Sex', 'SibSp', 'Parch']
    X = pd.get_dummies(train_data[features])
    X_test = pd.get_dummies(test_data[features])

    model = get_RFC()
    model.fit(X, y)
    pred = model.predict(X_test)

    output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': pred})
    output.to_csv('submission.csv', index=False)
    print("Your submission was successfully saved!")


if __name__ == '__main__':
    main()