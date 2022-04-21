import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelBinarizer
from math import sqrt
from tensorflow.keras.utils import to_categorical
from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend,
                              title, subplot, show, grid)
from sklearn import model_selection

pd.set_option('display.max_columns', None)
def preprocess(df):
    #Drop several columns
    df.drop('school', inplace=True, axis=1)
    df.drop('sex', inplace=True, axis=1)
    df.drop('address', inplace=True, axis=1)
    df.drop('famsize', inplace=True, axis=1)
    df.drop('Pstatus', inplace=True, axis=1)
    df.drop('guardian', inplace=True, axis=1)
    df.drop('traveltime', inplace=True, axis=1)
    df.drop('studytime', inplace=True, axis=1)
    df.drop('failures', inplace=True, axis=1)
    df.drop('schoolsup', inplace=True, axis=1)
    df.drop('famsup', inplace=True, axis=1)
    df.drop('paid', inplace=True, axis=1)
    df.drop('activities', inplace=True, axis=1)
    df.drop('higher', inplace=True, axis=1)
    df.drop('internet', inplace=True, axis=1)
    df.drop('nursery', inplace=True, axis=1)
    df.drop('romantic', inplace=True, axis=1)
    df.drop('famrel', inplace=True, axis=1)
    N, M = df.shape
    #Categorize data
    attributeNames = np.asarray(df.columns[range(0, M)])
    for attribute in attributeNames:
        if not isinstance(df[attribute][0], np.integer):
            df[attribute] = df[attribute].astype('category').cat.codes

    column_to_move_1 = df.pop('G1')
    column_to_move_2 = df.pop('G2')
    column_to_move_3 = df.pop('G3')
    column_to_move_4 = df.pop('Mjob')
    column_to_move_5 = df.pop('Fjob')
    column_to_move_6 = df.pop('reason')
    df = (df - df.mean()) / df.std()
    df = df.assign(Mjob=column_to_move_4,
                   Fjob=column_to_move_5,
                   reason=column_to_move_6)
    df = pd.get_dummies(data=df, columns=['Mjob', 'Fjob', 'reason'])
    df = df.assign(G1 = column_to_move_1,
              G2 = column_to_move_2,
              G3 = column_to_move_3)

    df.insert(0, 'Offset', 1)
    return df

def calculateGrade(grade):
    if grade < 13:
        return 0
    elif grade >= 13:
        return 1

df = preprocess(pd.read_csv(r'student-por.csv',";"))
X = df.iloc[:,0:-3]

y = df.iloc[:,-1]


y_cat = to_categorical(df.iloc[:,-1].apply(lambda x: calculateGrade(x)))


N, M = X.shape

attributeNames = df.columns[range(0, M)]
df.to_csv('preprocessed_data_v2.csv')
print('preprocessing runned')