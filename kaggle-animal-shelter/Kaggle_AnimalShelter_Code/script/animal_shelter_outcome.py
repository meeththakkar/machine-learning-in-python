import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import svm
from animal_shelter_outcome_helper import dataset_transformation
import warnings
import seaborn as sns
import csv as csv
from sklearn.linear_model import SGDClassifier
from sklearn.utils import check_array
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.cluster import KMeans
from sklearn import linear_model



warnings.filterwarnings("ignore")

desired_width = 500
pd.set_option('display.width', desired_width)

data_train = pd.read_csv("../input/train.csv")
data_test = pd.read_csv('../input/test.csv')


#print(data_train[0:100])


data_train  = dataset_transformation(data_train)
#data_test  = dataset_transformation(data_test)



classifiers = []
weight = []

X_all = data_train.drop(['AnimalID', 'OutcomeType', 'OutcomeSubtype','Name'], axis=1)
y_all = data_train['OutcomeType']


print(X_all.isnull().values.any())
print(y_all.isnull().values.any())

print(X_all.head(10))


num_test = 0.10
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=num_test, random_state=23)


clf = svm.SVC()
classifiers.append(clf)
classifiers[0].fit(X_train, y_train)
predictions = classifiers[0].predict(X_test)
accuracy = accuracy_score(y_test, predictions)
weight.append(accuracy)
print("Total accuracy of SVM: {0}".format(accuracy))

with open("submission.csv", "w") as predictions_file:
    predictions_file_object = predictions_file
    predictions_file_object.write('AnimalID,Adoption,Died,Euthanasia,Return_to_owner,Transfer\r\n')

    test_data = pd.read_csv("../input/test.csv")
    test_passengerId = test_data['PassengerId'].copy()

    test_data = process_gender(test_data)
    test_data = process_Cabin(test_data)
    test_data = process_ports(test_data)
    test_data = process_Ticket(test_data)
    test_data = process_age(test_data)
    test_data = process_names(test_data)
    test_data = drop_Features(test_data)
    # TODO: add cabin and see if it makes difference.
    test_data = test_data.drop(['Name', 'PassengerId'], axis=1)

    test_inputs = test_data

    for ix in range(len(test_data.index)):
        inputVector = test_inputs.irow(ix)
        inputVector = np.nan_to_num(inputVector)
        inputVector = check_array(inputVector, accept_sparse='csr', dtype=np.float64, order="C")
        estimatedLabel = output = getMajority(classifiers, weight, inputVector)
        if estimatedLabel == -1:
            estimatedLabel = 0
        predictions_file_object.write("{0},{1}\r\n".format(test_passengerId[ix], estimatedLabel))

    predictions_file.close()


