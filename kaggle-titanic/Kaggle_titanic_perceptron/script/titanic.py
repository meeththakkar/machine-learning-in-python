import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
#from sklearn import svm, metrics
import warnings
import csv as csv
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

weights = np.zeros(len(train_data.columns))

weights.fill(0)

learningRate = 0.01

Iterations = 400
errors = []
for numIter in range(Iterations):
    print("iteration {0}".format(numIter))
    err = 0  # reset the error counter
    # For each handwritten digit in training set,
    for ix in range(len(train_data.index)):
        label = train_lables[ix]
        inputVector = train_inputs.irow(ix)
        dotproduct = np.dot(inputVector, weights)  # take the dot product of input and weight
        estimatedLabel = np.sign(dotproduct)
        #perceptron adjustment
        if estimatedLabel == 0:
            estimatedLabel = -1
        actualLabel = train_lables[ix];
        if actualLabel != estimatedLabel:
            weights = weights + actualLabel*learningRate*inputVector
            err+=1

    errors.append( (err*1.0) / len(train_inputs) )  # track the error after each pass through the training set



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
        dotproduct = np.dot(inputVector, weights)  # take the dot product of input and weight
        estimatedLabel = np.sign(dotproduct)
        if estimatedLabel == -1:
            estimatedLabel = 0
        else:
            estimatedLabel = 1
        predictions_file_object.write("{0},{1}\r\n".format(test_passengerId[ix],estimatedLabel))

# Close out the files
    predictions_file.close()


