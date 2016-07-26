import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import svm
import warnings
import csv as csv

from sklearn.utils import check_array

warnings.filterwarnings("ignore")

train_data = pd.read_csv("../input/train.csv")
train_lables = train_data['Survived'].copy()
train_data['Gender'] = train_data['Sex'].map({'female': 0, 'male': 1}).astype(int)
#TODO: add cabin and see if it makes difference.
train_data = train_data.drop(['Name', 'Sex', 'Ticket', 'PassengerId','Survived','Cabin'], axis=1)

#used some of the code from example of random forest.
Ports = train_data.Embarked.unique()
Ports = list(enumerate(Ports))    # determine all values of Embarked,
Ports_dict = { name : i for i, name in Ports }              # set up a dictionary in the form  Ports : index
train_data.Embarked = train_data.Embarked.map( lambda x: Ports_dict[x]).astype(int)     # Convert all Embark strings to int

# All the ages with no input -> make the median of all Ages
median_age = train_data['Age'].dropna().median()
if len(train_data.Age[ train_data.Age.isnull() ]) > 0:
    train_data.loc[ (train_data.Age.isnull()), 'Age'] = median_age


#preparing for perceptron

train_lables[train_lables == 0] = -1
train_inputs = train_data


print('number of rows {0}'.format(len(train_data.columns)))

X = train_inputs
Y = train_lables


clf = svm.SVC(C= 10)
clf.fit(X, Y)
print("score is {0}".format(clf.score(X,Y)))






with open("submission.csv", "w") as predictions_file:
    predictions_file_object = predictions_file
    predictions_file_object.write('PassengerId,Survived\r\n')

    test_data = pd.read_csv("../input/test.csv")
    test_passengerId = test_data['PassengerId'].copy()
    test_data['Gender'] = test_data['Sex'].map({'female': 0, 'male': 1}).astype(int)
    #TODO: add cabin and see if it makes difference.
    test_data = test_data.drop(['Name', 'Sex', 'Ticket', 'PassengerId','Cabin'], axis=1)
    test_data.Embarked = test_data.Embarked.map( lambda x: Ports_dict[x]).astype(int)     # Convert all Embark strings to int

    if len(test_data.Age[ test_data.Age.isnull() ]) > 0:
        test_data.loc[ (test_data.Age.isnull()), 'Age'] = median_age

    test_inputs = test_data

    for ix in range(len(test_data.index)):
        inputVector = test_inputs.irow(ix)
        inputVector = np.nan_to_num(inputVector)
        inputVector = check_array(inputVector, accept_sparse='csr', dtype=np.float64, order="C")
        estimatedLabel = output = clf.predict(inputVector)
        if estimatedLabel[0] == -1 :
            estimatedLabel[0] = 0
        predictions_file_object.write("{0},{1}\r\n".format(test_passengerId[ix],estimatedLabel[0]))

# Close out the files
    predictions_file.close()