import pandas as pd
from sklearn import preprocessing



dog_median_age = 0
cat_median_age = 0
Lastnumber= 1
Lastmultuplier= 1

def calculateAge(val: str):

    number= Lastnumber
    multuplier = Lastmultuplier

    if not (type(val) is str):
        return number*multuplier

    arr = val.split(" ")
    number = int(arr[0])
    monthToWeek = 4.34524
    yearToWeeks = 52.1429

    charc = arr[1][0]

    if charc == 'w' or charc == 'W':
        multuplier = 1
    elif charc == 'm' or charc == 'M':
        multuplier = monthToWeek
    elif charc == 'y' or charc == 'Y':
        multuplier = yearToWeeks
    elif charc == 'd' or charc == 'D':
        multuplier = 1/7

    if multuplier == 0 :
        print("EXCEPTION cant find multiplier")

    print(number*multuplier)
    return number*multuplier

def setMissingAge(row):

    row['AgeuponOutcome'] = dog_median_age if row['AnimalType'] == 'Dog' else cat_median_age
    return row

def processName(val):
    return val


def encodeFeature(df, featureID):
    le = preprocessing.LabelEncoder()
    if len(df[featureID][df[featureID].isnull()]) > 0:
        df.loc[(df[featureID].isnull()), featureID] = 'UNKNOWN'
    df[featureID] = le.fit_transform(df[featureID])
    return le

def encodeFeature(df, featureID,le = preprocessing.LabelEncoder()):
    if len(df[featureID][df[featureID].isnull()]) > 0:
        df.loc[(df[featureID].isnull()), featureID] = 'UNKNOWN'
    df[featureID] = le.fit_transform(df[featureID])
    return le


def dataset_transformation(df : pd.DataFrame, testData : pd.DataFrame):
    df['AgeuponOutcome'] =  df['AgeuponOutcome'].apply(calculateAge)
    testData['AgeuponOutcome'] = testData['AgeuponOutcome'].apply(calculateAge)

    df['Name'] = df['Name'].apply(processName)
    testData['Name'] = testData['Name'].apply(processName)

    #df = df.apply(setMissingAge, axis=1)
    #testData = testData.apply(setMissingAge, axis=1)

    le =  encodeFeature(df,'OutcomeType')
    encodeFeature(df,'OutcomeSubtype')
    encodeFeature(df,'AnimalType')
    encodeFeature(df,'SexuponOutcome')
    encodeFeature(df,'Breed')
    encodeFeature(df,'Color')
    #encodeFeature(df,'Breed1')
    #encodeFeature(df,'Breed2')
    #encodeFeature(df,'Breedcount')
    #encodeFeature(df,'Name')

    df =  df.drop(['DateTime'], axis=1)
    df = df.drop(['Breed'], axis=1)

    dog_median_age = df.query('AnimalType == "Dog"')['AgeuponOutcome'].dropna().mean()
    cat_median_age = df.query('AnimalType == "Cat"')['AgeuponOutcome'].dropna().mean()

    return df