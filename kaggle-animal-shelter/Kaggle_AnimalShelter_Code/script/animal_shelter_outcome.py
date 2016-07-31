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


[data_train,data_test] = dataset_transformation(data_train,data_test)


classifiers = []
weight = []

X_all = data_train.drop(['AnimalID', 'OutcomeType', 'OutcomeSubtype'], axis=1)
y_all = data_train['OutcomeType']


print(X_all.isnull().values.any())
print(y_all.isnull().values.any())


print(data_test.isnull().values.any())

print(X_all.head(10))


num_test = 0.001
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=num_test, random_state=23)



data_test.to_csv("data_test_after_transform.csv", sep=',', encoding='utf-8')



clf = svm.SVC()
classifiers.append(clf)
classifiers[0].fit(X_train, y_train)
predictions = classifiers[0].predict(X_test)
accuracy = accuracy_score(y_test, predictions)
weight.append(accuracy)
print("Total accuracy of SVM: {0}".format(accuracy))





with open("submission.csv", "w") as predictions_file:
    predictions_file_object = predictions_file
    predictions_file_object.write('ID,Adoption,Died,Euthanasia,Return_to_owner,Transfer\r\n')


    test_data = data_test

    test_ID = test_data['ID'].copy()

    test_inputs = test_data.drop(['ID'], axis =1 )

    for ix in range(len(test_data.index)):
        inputVector = test_inputs.irow(ix)
        inputVector = np.nan_to_num(inputVector)
        inputVector = check_array(inputVector, accept_sparse='csr', dtype=np.float64, order="C")
        estimatedLabel = output = clf.predict(inputVector)
        estimatedLabel= estimatedLabel[0]
        #print(estimatedLabel)

        Adoption = 1 if estimatedLabel == 'Adoption' else 0
        Died = 1 if estimatedLabel == 'Died' else 0
        Euthanasia = 1 if estimatedLabel == 'Euthanasia' else 0
        Return_to_owner = 1 if estimatedLabel == 'Return_to_owner' else 0
        Transfer = 1 if estimatedLabel == 'Transfer' else 0

        #Adoption, Died, Euthanasia, Return_to_owner, Transfer


        predictions_file_object.write("{0},{1},{2},{3},{4},{5}\r\n".format(test_ID[ix], Adoption,Died,Euthanasia,Return_to_owner,Transfer))

    predictions_file.close()


