import pandas as pd


def read_csv(file='train.csv'):
    data = pd.read_csv(file)
    return data


def read_traintest(files=['train.csv', 'test.csv']):
    train_data = read_csv(files[0])
    test_data = read_csv(files[1])
    return train_data, test_data


if __name__ == '__main__':
    train_data = read_csv()
    test_data = read_csv('test.csv')
    # women = women = train_data[train_data.Sex == 'female']['Survived']
    # sum(women) / len(women)
    print('a')